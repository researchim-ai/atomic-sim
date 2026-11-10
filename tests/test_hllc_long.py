"""Arbitrary histories, late replay protocol and independent window reference."""
from copy import deepcopy
import numpy as np
import pytest
from scripts import graph_history as history
from scripts import verify_hllc_long as long
from scripts import long_hllc_reference as reference
from scripts import benchmark_modelica_network as legacy
from scripts.verify_heterogeneous_network import as_peer

TIMES=(0.,1e-5,2e-5,3e-5)

@pytest.fixture(scope='module')
def packet():return long.capture(long.request_for('pressure_thermal_step',2,long.CAPS[-1]),times=TIMES,restart_index=2)


@pytest.mark.parametrize('times',[[],[1.,2.],[0.,0.],[0.,float('nan')],[0.,float('inf')],[0.,.1,-1.],[False,.1]])
def test_invalid_grids_rejected(times):
    with pytest.raises(ValueError):history.validate_times(times)


def test_arbitrary_history_and_restart_without_changing_legacy_grid(packet):
    old=legacy.TIMES
    result=history.compare(packet,as_peer(packet['observations']),times=TIMES)
    assert result['passed'] and packet['record']['passed']
    assert packet['record']['restart_time_s']==TIMES[2]
    assert max(result['maximum_errors'].values())==0.
    assert legacy.TIMES==old


@pytest.mark.parametrize('damage',['length','time','nan','identity','ledger'])
def test_incomplete_or_inconsistent_history_cannot_pass(packet,damage):
    native=deepcopy(packet);peer=as_peer(native['observations'])
    if damage=='length':peer.pop()
    elif damage=='time':peer[1]['time']+=.001
    elif damage=='nan':peer[1]['pressure[1]']=float('nan')
    elif damage=='identity':native['observations'][1]['cells'][0]['id']='other'
    else:native['observations'][1]['local_ledgers']['integrated_face_mass_kg'].pop()
    with pytest.raises(ValueError):history.compare(native,peer,times=TIMES)


@pytest.mark.parametrize('index',[0,3,True])
def test_bad_restart_boundary_rejected_before_model_creation(index):
    with pytest.raises(ValueError):long.capture({},times=TIMES,restart_index=index)


def comparisons():
    # Convergence harness only; these numbers are never physical evidence.
    return [dict(cap_s=cap,method=method,passed=True,maximum_errors=dict(pressure_pa=1/8**index,momentum_kg_m_s=1e-6/8**index))
            for index,cap in enumerate(long.CAPS) for method in long.REFERENCE_METHODS]


@pytest.mark.parametrize('damage',['missing','duplicate','nonfinite','negative'])
def test_temporal_matrix_rejects_invalid_evidence(damage):
    rows=comparisons()
    if damage=='missing':rows.pop()
    elif damage=='duplicate':rows.append(deepcopy(rows[0]))
    elif damage=='nonfinite':rows[0]['maximum_errors']['pressure_pa']=float('nan')
    else:rows[0]['maximum_errors']['pressure_pa']=-1.
    with pytest.raises(ValueError):long.refinement(rows)


def test_refinement_accuracy_and_roundoff_floors_have_separate_meanings():
    rows=comparisons();assert long.refinement(rows)['passed']
    rows[0]['passed']=False
    assert long.refinement(rows)['passed'] # coarse caps are explicitly exploratory
    rows[-1]['passed']=False
    assert not long.refinement(rows)['passed']
    rows=comparisons();rows[2]['maximum_errors']['pressure_pa']=10.
    assert not long.refinement(rows)['passed']
    rows=comparisons()
    for row in rows:row['maximum_errors']={k:0. for k in long.ERROR_FLOORS}
    assert long.refinement(rows)['passed'] # floor resolved; no claimed measured order


def test_real_window_reference_preserves_geometry_and_cross_agrees():
    request=long.request_for('pressure_thermal_step',2,long.CAPS[-1])
    refs=[reference.solve(request,TIMES,method,max_step=1e-5) for method in long.REFERENCE_METHODS]
    assert all(sum(w['evaluations'] for w in r['windows'])==r['evaluations'] for r in refs)
    assert all(0<w['evaluations']<=20000 for r in refs for w in r['windows'])
    native=long.reference_packet(request,refs[0])
    assert 'volume_m3' not in refs[0]['observations'][0]['cells'][0]
    result=history.compare(native,as_peer(refs[1]['observations']),times=TIMES,budgets=long.peer.REFERENCE_BUDGETS)
    assert result['passed']


def test_one_pass_iterable_time_grid_is_not_consumed_during_validation():
    assert history.validate_times(iter(TIMES))==TIMES

@pytest.mark.parametrize('rtol,atol',[(float('nan'),1e-10),(1e-11,0.)])
def test_bad_reference_tolerances_rejected_before_assembly(rtol,atol):
    with pytest.raises(ValueError):reference.solve({},TIMES,'BDF',max_step=1e-5,rtol=rtol,atol=atol)


def test_wide_bdf_profile_tightens_without_relaxing_other_cases():
    original=long.reference_accuracy('pressure_thermal_step','Radau')
    assert original==dict(rtol=2e-11,atol=1e-10)
    tighter=long.reference_accuracy('wide_pressure_step','BDF')
    assert tighter['rtol']==original['rtol']/2 and tighter['atol']==original['atol']/2


def test_refinement_cannot_reuse_foreign_source(tmp_path):
    import json
    from scripts import refine_hllc_long
    from atomic_model_multiphysics.provenance import seal
    parent=tmp_path/'parent';parent.mkdir()
    (parent/'protocol.json').write_text(json.dumps(seal(dict(source={}))))
    (parent/'report.json').write_text(json.dumps(seal(dict(source_unchanged=True))))
    with pytest.raises(ValueError):refine_hllc_long.run(tmp_path/'out',parent)
    assert not (tmp_path/'out').exists()
