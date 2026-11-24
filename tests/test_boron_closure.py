"""Independent heat reference and finite closing-tail acceptance contracts."""
import math
from types import SimpleNamespace

import pytest
from coupling.regenerator import counterflow
from scripts import qualify_boron_closure as closure
from scripts import qualify_boron_duration as duration


def inputs(flow,hot_cp=5240.,cold_cp=4180.):
    return counterflow(flow_kg_s=flow,ua_w_k=500000.,hot_k=563.,cold_k=350.6,
        hot_cp_j_kg_k=hot_cp,cold_cp_j_kg_k=cold_cp)


@pytest.mark.parametrize('flow',[1.,math.exp(-.5),math.exp(-1.),math.exp(-2.5),math.exp(-5.)])
@pytest.mark.parametrize('hot_cp,cold_cp',[(5240.,4180.),(4180.,5240.),(4180.,4180.),(4180.*(1+1e-12),4180.)])
def test_stable_reference_resolves_both_capacity_orderings_and_closing_tail(flow,hot_cp,cold_cp):
    model=inputs(flow,hot_cp,cold_cp);ref=closure.tail_reference(model,flow)
    assert ref['heat_w']==pytest.approx(model['heat_w'],rel=1e-8)
    assert ref['hot_outlet_k']==pytest.approx(model['hot_outlet_capacity_temperature_k'],abs=1e-7)
    assert ref['cold_outlet_k']==pytest.approx(model['cold_outlet_capacity_temperature_k'],abs=1e-7)
    if flow>=.1:
        assert ref['heat_w']==pytest.approx(duration.gap_reference(model,flow)['heat_w'],rel=1e-8)


def test_reverse_heat_and_temperature_translation():
    q=math.exp(-5.);x=inputs(q);before=closure.tail_reference(x,q)
    y=dict(x,hot_inlet_temperature_k=x['hot_inlet_temperature_k']+20.,cold_inlet_temperature_k=x['cold_inlet_temperature_k']+20.)
    after=closure.tail_reference(y,q)
    assert after['heat_w']==pytest.approx(before['heat_w'],rel=1e-12)
    assert after['hot_outlet_k']==pytest.approx(before['hot_outlet_k']+20.)
    reverse=dict(x,hot_inlet_temperature_k=x['cold_inlet_temperature_k'],cold_inlet_temperature_k=x['hot_inlet_temperature_k'])
    assert closure.tail_reference(reverse,q)['heat_w']==pytest.approx(-before['heat_w'],rel=1e-12)


@pytest.mark.parametrize('mode',['flow','ua','temperature'])
def test_degenerate_limits(mode):
    x=inputs(1.);q=1.
    if mode=='flow':q=0.
    elif mode=='ua':x['ua_w_k']=0.
    else:x['hot_inlet_temperature_k']=x['cold_inlet_temperature_k']
    assert closure.tail_reference(x,q)['heat_w']==0.


@pytest.mark.parametrize('time',[-1.,61.01,float('nan'),True])
def test_mass_reference_refuses_outside_declared_protocol(time):
    with pytest.raises(ValueError,match='0..61'):closure.reference_mass(time)


def test_tail_mass_is_not_clipped_at_command_or_assigned_its_infinite_limit():
    assert closure.reference_mass(60.)==pytest.approx(59.8)
    assert closure.reference_mass(61.)==pytest.approx(60.-.2*math.exp(-5.),abs=1e-12)
    assert 59.8<closure.reference_mass(61.)<60.
    assert closure.STEPS[:5]==duration.STEPS
    assert closure.STEPS[5]['patch']=={'boron_flow_target_kg_s':0.}
    assert sum(s['duration_s'] for s in closure.STEPS)==61.


@pytest.mark.parametrize('budget',[True,1,10001])
def test_invalid_reference_budget(budget):
    with pytest.raises(ValueError,match='budget'):closure.tail_reference({},1.,max_nodes=budget)


def test_unsuccessful_or_over_budget_reference_is_not_evidence(monkeypatch):
    monkeypatch.setattr(closure,'solve_ivp',lambda *a,**k:SimpleNamespace(success=False))
    with pytest.raises(ValueError,match='converge'):closure.tail_reference(inputs(1.),1.)


def test_solved_reference_exceeding_accepted_mesh_budget_is_refused():
    with pytest.raises(ValueError,match='budget'):closure.tail_reference(inputs(math.exp(-5.)),math.exp(-5.),max_nodes=8)


@pytest.mark.parametrize('field,value',[('ua_w_k',-1.),('hot_cp_j_kg_k',True),('cold_inlet_temperature_k',float('nan'))])
def test_invalid_capacity_inputs(field,value):
    x=inputs(1.);x[field]=value
    with pytest.raises(ValueError,match='inputs'):closure.tail_reference(x,1.)


def test_missing_chemical_time_matrix_is_refused():
    with pytest.raises(ValueError,match='Complete'):closure.assess({})
