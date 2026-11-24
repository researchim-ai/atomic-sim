import numpy as np
import pytest
from scipy.integrate import solve_bvp

from coupling.regenerator import counterflow


def evaluate(**changes):
    return counterflow(**dict(dict(flow_kg_s=10.,ua_w_k=50000.,hot_k=560.,cold_k=350.,
        hot_cp_j_kg_k=5200.,cold_cp_j_kg_k=4180.),**changes))


@pytest.mark.parametrize('hot_cp,cold_cp,ua',[(4180.,4180.,50000.),(5200.,4180.,50000.),
    (4180.,5200.,50000.),(5200.,4180.,500000.),(4180.,4180.*(1.+1e-12),50000.)])
def test_heat_matches_independent_counterflow_boundary_value_problem(hot_cp,cold_cp,ua):
    ch=10.*hot_cp;cc=10.*cold_cp
    def rhs(x,y):
        difference=y[0]-y[1]
        return np.vstack((-ua/ch*difference,-ua/cc*difference))
    def boundary(a,b):return np.array([a[0]-560.,b[1]-350.])
    x=np.linspace(0.,1.,8);guess=np.vstack((np.full(8,560.),np.full(8,350.)))
    solution=solve_bvp(rhs,boundary,x,guess,tol=1e-9,max_nodes=2000)
    assert solution.success
    r=evaluate(hot_cp_j_kg_k=hot_cp,cold_cp_j_kg_k=cold_cp,ua_w_k=ua)
    assert r['hot_outlet_capacity_temperature_k']==pytest.approx(solution.y[0,-1],abs=1e-7)
    assert r['cold_outlet_capacity_temperature_k']==pytest.approx(solution.y[1,0],abs=1e-7)
    assert r['heat_w']==pytest.approx(ch*(560.-solution.y[0,-1]),rel=1e-9)


@pytest.mark.parametrize('ua,flow',[(0.,10.),(50000.,0.),(0.,0.)])
def test_no_heat_without_conductance_or_flow(ua,flow):
    r=evaluate(ua_w_k=ua,flow_kg_s=flow)
    assert r['heat_w']==r['effectiveness']==r['capacity_entropy_generation_w_k']==0.
    assert r['hot_outlet_capacity_temperature_k']==560.
    assert r['cold_outlet_capacity_temperature_k']==350.


@pytest.mark.parametrize('flow,ua',[(1e-300,1e8),(100.,1e-9),(1.,1e8),(10.,50000.)])
def test_capacity_balance_endpoint_bounds_and_entropy(flow,ua):
    r=evaluate(flow_kg_s=flow,ua_w_k=ua)
    assert 0<=r['effectiveness']<=1.
    assert 350.<=r['hot_outlet_capacity_temperature_k']<=560.
    assert 350.<=r['cold_outlet_capacity_temperature_k']<=560.
    assert r['heat_w']==pytest.approx(r['hot_capacity_w_k']*(560.-r['hot_outlet_capacity_temperature_k']),abs=1e-7)
    assert r['heat_w']==pytest.approx(r['cold_capacity_w_k']*(r['cold_outlet_capacity_temperature_k']-350.),abs=1e-7)
    assert r['capacity_entropy_generation_w_k']>=-1e-8


def test_reverse_temperature_difference_transfers_heat_passively():
    r=evaluate(hot_k=350.,cold_k=560.)
    assert r['heat_w']<0 and r['capacity_entropy_generation_w_k']>0


@pytest.mark.parametrize('changes',[dict(flow_kg_s=True),dict(ua_w_k=-1.),dict(ua_w_k=float('inf')),
    dict(hot_k=float('nan')),dict(cold_k=0.),dict(hot_cp_j_kg_k=0.),dict(cold_cp_j_kg_k=True)])
def test_bad_inputs_are_rejected(changes):
    with pytest.raises(ValueError):evaluate(**changes)
