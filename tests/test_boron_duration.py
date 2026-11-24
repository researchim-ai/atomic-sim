from types import SimpleNamespace
import pytest

from coupling.regenerator import counterflow
from scripts import qualify_boron_duration as duration


def reference_inputs(flow=1.,hot_cp=5240.,cold_cp=4180.):
    return counterflow(flow_kg_s=flow,ua_w_k=500000.,hot_k=563.,cold_k=350.6,
        hot_cp_j_kg_k=hot_cp,cold_cp_j_kg_k=cold_cp)


@pytest.mark.parametrize('flow,hot_cp,cold_cp',[(.1,5240.,4180.),(.3934693402873666,5240.,4180.),
    (1.,5240.,4180.),(50.,5240.,4180.),(1.,4180.,5240.),(1.,4180.,4180.),(1.,4180.*(1+1e-12),4180.)])
def test_normalized_boundary_problem_matches_capacity_model_including_small_flow(flow,hot_cp,cold_cp):
    model=reference_inputs(flow,hot_cp,cold_cp);ref=duration.gap_reference(model,flow)
    assert ref['heat_w']==pytest.approx(model['heat_w'],rel=1e-8)
    assert ref['hot_outlet_k']==pytest.approx(model['hot_outlet_capacity_temperature_k'],abs=1e-7)
    assert ref['cold_outlet_k']==pytest.approx(model['cold_outlet_capacity_temperature_k'],abs=1e-7)


def test_reference_preserves_temperature_translation_and_reverse_transfer():
    x=reference_inputs();before=duration.gap_reference(x,1.)
    y=dict(x,hot_inlet_temperature_k=x['hot_inlet_temperature_k']+20.,cold_inlet_temperature_k=x['cold_inlet_temperature_k']+20.)
    after=duration.gap_reference(y,1.)
    assert after['heat_w']==pytest.approx(before['heat_w'],rel=1e-12)
    assert after['cold_outlet_k']==pytest.approx(before['cold_outlet_k']+20.,abs=1e-8)
    reversed_inputs=dict(x,hot_inlet_temperature_k=x['cold_inlet_temperature_k'],cold_inlet_temperature_k=x['hot_inlet_temperature_k'])
    assert duration.gap_reference(reversed_inputs,1.)['heat_w']==pytest.approx(-before['heat_w'],rel=1e-12)


@pytest.mark.parametrize('kind',['flow','ua','temperature'])
def test_no_reference_heat_in_degenerate_limits(kind):
    x=reference_inputs();flow=1.
    if kind=='flow':flow=0.
    elif kind=='ua':x['ua_w_k']=0.
    else:x['hot_inlet_temperature_k']=x['cold_inlet_temperature_k']
    assert duration.gap_reference(x,flow)['heat_w']==0.


@pytest.mark.parametrize('budget',[True,1,10001])
def test_reference_node_budget_is_bounded(budget):
    with pytest.raises(ValueError,match='budget'):duration.gap_reference({},1.,max_nodes=budget)


def test_failed_gap_reference_cannot_be_used_as_evidence(monkeypatch):
    monkeypatch.setattr(duration,'solve_bvp',lambda *args,**kwargs:SimpleNamespace(success=False))
    with pytest.raises(ValueError,match='did not converge'):duration.gap_reference(reference_inputs(),1.)


@pytest.mark.parametrize('cases',[{},dict(dilution={},boration={}),dict(dilution={},unknown={})])
def test_incomplete_minute_matrix_is_refused(cases):
    with pytest.raises(ValueError,match='Complete'):duration.assess(cases)


def test_minute_inventory_reference_and_protocol():
    assert duration.reference_mass(0.)==0.
    assert duration.reference_mass(60.)==pytest.approx(59.8,abs=1e-12)
    assert sum(x['duration_s'] for x in duration.STEPS)==60.
    assert duration.STEPS[0]['patch']=={'boron_flow_target_kg_s':1.}


@pytest.mark.parametrize('field,value',[('ua_w_k',-1.),('hot_cp_j_kg_k',True),('cold_inlet_temperature_k',float('nan'))])
def test_invalid_capacity_reference_inputs_are_refused(field,value):
    x=reference_inputs();x[field]=value
    with pytest.raises(ValueError,match='inputs'):duration.gap_reference(x,1.)
