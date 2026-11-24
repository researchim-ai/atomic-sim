from copy import deepcopy
import json
import numpy as np
import pytest
from atomic_model_multiphysics.provenance import seal,unseal
from coupling import core_fidelity as f


def cost(cheap=1.,reference=2.):
    return seal(dict(format='core-fidelity-cost/v1',source=f.identity(),measurements={
        c:{'2':[cheap,cheap*1.1,cheap*.9],'4':[reference,reference*1.1,reference*.9]} for c in f.CASES},construction_timed=True,repeats=3))


@pytest.mark.parametrize('cheap,reference,eligible',[(1.,2.,True),(2.,1.,False),(1.,1.,False)])
def test_cost_gate_computes_medians_instead_of_trusting_labels(cheap,reference,eligible):
    d=f.cost_decision(unseal(cost(cheap,reference)),'insertion')
    assert d['cheap_eligible'] is eligible and d['speed_ratio']==pytest.approx(reference/cheap)


@pytest.mark.parametrize('value',[0.,-1.,float('nan'),True])
def test_invalid_cost_cannot_authorize_reduction(value):
    c=unseal(cost());c['measurements']['insertion']['2'][0]=value
    with pytest.raises(ValueError):f.cost_decision(c,'insertion')


def test_missing_timing_repeats_and_stale_source_refused():
    c=unseal(cost());c['measurements']['insertion']['2'].pop()
    with pytest.raises(ValueError,match='Three'):f.cost_decision(c,'insertion')
    c=unseal(cost());c['source']={}
    with pytest.raises(ValueError,match='source'):f.cost_decision(c,'insertion')


@pytest.mark.parametrize('cases',[{},dict(insertion={},return_case={})])
def test_complete_resolution_time_scenario_matrix_required(cases):
    with pytest.raises(ValueError,match='Complete'):f.assess(cases)


def test_incomplete_or_stale_scientific_envelope_refused():
    with pytest.raises(ValueError):f.validate_science(seal({}))
    with pytest.raises(ValueError):f.validate_science(seal(dict(policy_source={},cases={},report={})))


def test_only_fixed_scenarios_and_resolutions_are_supported():
    with pytest.raises(ValueError):f.protocol('scram',.2)
    with pytest.raises(ValueError):f.session(3)
    assert sum(s['duration_s'] for s in f.protocol('insertion',.2))==2.
    assert sum(s['duration_s'] for s in f.protocol('return',.2))==3.


@pytest.fixture(scope='module')
def cheap_seed():
    return f.session(2)


@pytest.fixture
def mocked_science_dispatcher(monkeypatch,cheap_seed):
    # Pure dispatcher test: full numerical evidence is exercised by the
    # qualification driver's separate policy_proof, not forged here.
    monkeypatch.setattr(f,'validate_science',lambda science,**kwargs:(None,{n:deepcopy(cheap_seed) for n in f.MESHES}))
    monkeypatch.setattr(f,'session',lambda cells,preset='fine':deepcopy(cheap_seed))
    return f.QualifiedCoreRun(seal({}),cost(),case='insertion')


def test_unqualified_command_is_rejected_before_any_state_change(mocked_science_dispatcher):
    run=mocked_science_dispatcher;before=run.checkpoint()
    with pytest.raises(ValueError,match='Outside'):run.advance(.1,{'rod_target':0.})
    assert run.checkpoint()==before
    with pytest.raises(ValueError,match='Outside'):run.advance(True)
    assert run.checkpoint()==before


def test_qualified_dispatcher_json_restart_and_evidence_binding(mocked_science_dispatcher):
    run=mocked_science_dispatcher;steps=f.protocol('insertion',run.observation()['snapshot']['actuator_status']['rod_position'])
    run.advance(**steps[0]);saved=json.loads(json.dumps(run.checkpoint()))
    restored=f.QualifiedCoreRun.restore(saved,seal({}),cost())
    assert restored.checkpoint()==saved
    assert restored.advance(**steps[1])==run.advance(**steps[1])
    with pytest.raises(ValueError,match='evidence changed'):f.QualifiedCoreRun.restore(saved,seal({}),cost(cheap=1.1))


def test_equivalent_integer_duration_is_recorded_canonically_for_restart(mocked_science_dispatcher):
    run=mocked_science_dispatcher;steps=f.protocol('insertion',run.observation()['snapshot']['actuator_status']['rod_position'])
    for step in steps[:3]:run.advance(**step)
    run.advance(1,None)
    saved=json.loads(json.dumps(run.checkpoint()))
    assert type(unseal(saved)['session']['payload']['history'][-1]['duration_s']) is float
    restored=f.QualifiedCoreRun.restore(saved,seal({}),cost())
    assert restored.checkpoint()==saved


def test_failed_overall_report_cannot_authorize_cheap_run():
    evidence=seal(dict(policy_source=f.identity(),cases={},report=dict(passed=False)))
    with pytest.raises(ValueError,match='Accepted report'):f.validate_science(evidence)


@pytest.mark.parametrize('change',[dict(construction_timed=False),dict(repeats=1),dict(repeats=True)])
def test_incompatible_cost_protocol_refused(change):
    c=unseal(cost());c.update(change)
    with pytest.raises(ValueError,match='Complete-episode'):f.cost_decision(c,'insertion')


def test_source_matching_evidence_still_requires_current_initial_recipe(monkeypatch,cheap_seed):
    assessment=dict(passed=True)
    controls={k:True for k in ('power_bias','sixth_precursor_bias','energy_failure','restart_failure','temperature_bias','rollback_failure')}
    initial=cheap_seed.snapshot();bad=deepcopy(initial);bad['controls']['rod_target']=0.
    cases={case:{str(n)+'_fine':dict(initial=deepcopy(bad)) for n in f.MESHES} for case in f.CASES}
    evidence=seal(dict(policy_source=f.identity(),cases=cases,report=dict(passed=True,assessment=assessment,negative_controls=dict(passed=True,checks=controls))))
    monkeypatch.setattr(f,'assess',lambda c:assessment)
    monkeypatch.setattr(f,'session',lambda n:deepcopy(cheap_seed))
    with pytest.raises(ValueError,match='default recipe anchor'):f.validate_science(evidence)
