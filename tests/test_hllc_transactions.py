"""Control atomicity is checked against real evolving networks and exact replay."""
from copy import deepcopy
import pytest
from atomic_model_hem_contact import ContactHEMNetwork
from scripts import verify_hllc_transactions as transactions
from scripts import verify_hllc_branching as branch

@pytest.fixture
def evolved():
    request=branch.request_for('branch_reversal',2,branch.CAPS[-1])
    model=ContactHEMNetwork(request,scheme='hllc-local');model.advance(1e-5)
    return model

@pytest.mark.parametrize('fault',transactions.FAULTS)
def test_rejection_preserves_every_checkpoint_and_observation_field(evolved,fault):
    result=transactions.reject_unchanged(evolved,fault)
    assert result['rejected'] and result['unchanged']
    assert result['stage_injections']==(1 if fault=='stage_failure' else 0)

@pytest.mark.parametrize('scheme,cells',[(s,c) for s in transactions.SCHEMES for c in transactions.MESHES])
def test_commands_replay_and_continue_identically(scheme,cells):
    result=transactions.trajectory(scheme,cells,times=(0.,1e-5,2e-5))
    assert result['passed']
    assert len(result['faults'])==2*len(transactions.FAULTS)
    assert [r['revision'] for r in result['restarts']]==[1,2]


def test_canonical_comparison_keeps_signed_zero():
    assert transactions.canonical({'x':0.})!=transactions.canonical({'x':-0.})
    with pytest.raises(ValueError):transactions.canonical({'x':float('nan')})

@pytest.mark.parametrize('scheme,cells',[('unknown',2),('hllc',3),('hllc-local',True)])
def test_undeclared_matrix_refused(scheme,cells):
    with pytest.raises(ValueError):transactions.trajectory(scheme,cells)


def test_refused_stage_restores_callable_and_next_command(evolved):
    clean=ContactHEMNetwork(evolved.request,scheme=evolved.scheme);clean.restore(evolved.checkpoint())
    transactions.reject_unchanged(evolved,'stage_failure')
    assert 'evaluate' not in evolved.__dict__
    controls=deepcopy(evolved.snapshot()['controls'])
    arguments=dict(expected_revision=1,controller_state={'tick':2})
    assert transactions.canonical(evolved.step(controls,1e-5,**arguments))==transactions.canonical(clean.step(controls,1e-5,**arguments))
