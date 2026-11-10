"""Full nonlinear wave sampling and conservative spatial-reference checks."""
import math
from copy import deepcopy
import numpy as np
import pytest
from atomic_model_hem_qualified import eos
from scripts.liquid_riemann_profile import Profile,Sample
from scripts import verify_hllc_spatial as spatial


def profile(reverse=False):
    a,b=eos.pt(14e6,441.),eos.pt(2e6,439.)
    return Profile(*( (b,a) if reverse else (a,b) ),pressure_bracket=(1e6,15e6))


@pytest.mark.parametrize('reverse',[False,True])
def test_rarefaction_interior_preserves_entropy_and_characteristic(reverse):
    p=profile(reverse);s=p.solution
    xi=(s.right_inner_speed_m_s+s.right_outer_speed_m_s)/2 if reverse else (s.left_inner_speed_m_s+s.left_outer_speed_m_s)/2
    sample=p.sample(xi);initial=p.right.initial if reverse else p.left.initial
    assert sample.zone.endswith('_fan')
    assert abs(sample.state.entropy_j_kg_k-initial.entropy_j_kg_k)<2e-8
    sign=1 if reverse else -1
    assert abs(sample.velocity_m_s+sign*eos.characteristic_speeds(sample.state)[1]-xi)<2e-7
    assert min(s.pressure_pa,initial.pressure_pa)<sample.state.pressure_pa<max(s.pressure_pa,initial.pressure_pa)


@pytest.mark.parametrize('xi',[-3000.,-1450.,-1000.,0.,1000.,1450.,3000.])
def test_full_profile_reflects_without_changing_thermodynamic_state(xi):
    a=profile().sample(xi);b=profile(True).sample(-xi)
    assert abs(a.state.pressure_pa-b.state.pressure_pa)<1e-4
    assert abs(a.state.temperature_k-b.state.temperature_k)<1e-8
    assert abs(a.velocity_m_s+b.velocity_m_s)<1e-8


@pytest.mark.parametrize('pl,pr',[(4.02e6,3.98e6),(14e6,2e6),(2e6,14e6)])
def test_integral_conservation_before_boundary_arrival(pl,pr):
    p=Profile(eos.pt(pl,441.),eos.pt(pr,439.),pressure_bracket=(1e6,15e6))
    first=p.cell_average(0.,10.,0.,origin=5.,energy_reference=7e5)
    final=p.cell_average(0.,10.,.001,origin=5.,energy_reference=7e5)
    delta=final-first
    assert abs(delta[0])<2e-9
    assert abs(delta[1]-.001*(pl-pr)/10)<2e-6
    assert abs(delta[2])<2e-5


def test_cell_average_is_additive_and_not_a_midpoint_sample():
    p=profile();a,b=3.,5.;time=.001
    whole=p.cell_average(a,b,time,origin=5.,energy_reference=7e5)
    parts=.5*(p.cell_average(a,4.,time,origin=5.,energy_reference=7e5)+p.cell_average(4.,b,time,origin=5.,energy_reference=7e5))
    np.testing.assert_allclose(whole,parts,rtol=1e-11,atol=2e-6)
    assert np.max(abs(whole-p.sample((4.-5.)/time).conserved(7e5)))>1.


def test_moving_contact_cell_fraction_is_exact_at_initial_and_positive_time():
    left,right=eos.pt(4e6,441.),eos.pt(4e6,439.)
    p=Profile(left,right,10.,10.,pressure_bracket=(1e6,15e6))
    for t in (0.,.001):
        fraction=.37+10*t
        expected=fraction*Sample(left,10.,'left').conserved(7e5)+(1-fraction)*Sample(right,10.,'right').conserved(7e5)
        np.testing.assert_allclose(p.cell_average(0.,1.,t,origin=.37,energy_reference=7e5),expected,rtol=1e-13,atol=1e-7)


@pytest.mark.parametrize('damage',['nan','order','time'])
def test_invalid_sampling_is_refused(damage):
    p=profile()
    with pytest.raises(ValueError):
        if damage=='nan':p.sample(float('nan'))
        elif damage=='order':p.cell_average(1.,0.,.001)
        else:p.cell_average(0.,1.,-.001)


def fixture_rows():
    # Harness-only numbers: never used as physical evidence.
    rows=[]
    for case in spatial.CASES:
        for cells in spatial.MESHES:
            for scheme in spatial.SCHEMES:
                value=(1. if scheme=='hllc' else .8)/cells
                rows.append(dict(case=case,cells=cells,scheme=scheme,cap_s=spatial.CAP,
                    l1=dict(pressure_pa=value,momentum_kg_m2_s=value),curves=[dict(pressure_pa=[1.,1.])]))
        rows.append(dict(case=case,cells=spatial.MESHES[-1],scheme='hllc-local',cap_s=spatial.CAP/2,
            l1=dict(pressure_pa=.8/spatial.MESHES[-1],momentum_kg_m2_s=.8/spatial.MESHES[-1]),curves=[dict(pressure_pa=[1.,1.])]))
    return rows


@pytest.mark.parametrize('damage',['missing','duplicate','nonfinite'])
def test_spatial_matrix_cannot_fabricate_convergence(damage):
    rows=fixture_rows()
    if damage=='missing':rows.pop()
    elif damage=='duplicate':rows.append(deepcopy(rows[0]))
    else:rows[0]['l1']['pressure_pa']=float('nan')
    with pytest.raises(ValueError):spatial.assess(rows)


def test_spatial_and_time_failures_remain_failures():
    rows=fixture_rows();assert spatial.assess(rows)['passed']
    rows[2]['l1']['pressure_pa']=10.
    assert not spatial.assess(rows)['passed']
    rows=fixture_rows();rows[-1]['curves'][0]['pressure_pa']=[2.,2.]
    assert not spatial.assess(rows)['passed']


def test_real_graph_reference_balances_and_opposite_edge_orientation():
    ref=spatial.reference('moderate',16)
    result=spatial.execute('moderate',16,'hllc-local',spatial.CAP,ref)
    assert result['passed'] and result['checks']['boundary_drift']
    first,last=result['curves']
    np.testing.assert_allclose(first['conserved'],last['conserved'],rtol=2e-5,atol=.02)
