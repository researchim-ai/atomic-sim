from dataclasses import asdict
from decimal import Decimal,localcontext
import math
import json
from importlib.resources import files
from pathlib import Path
import numpy as np
import pytest
from core.status import DomainError,ModelLimitError
from atomic_model_station_mixed import *
from atomic_model_station_mixed.reference import collocation


@pytest.mark.parametrize('direction,branch,ratio',[
 ('upward','upper','0.9'),('upward','middle','0.5'),('upward','inverted','0.4'),('downward','unique','1.2')])
def test_named_root_against_60_digit_inverse_reference(direction,branch,ratio):
    with localcontext() as ctx:
        ctx.prec=60;r=Decimal(ratio);a=Decimal('.46');p=Decimal('2.1')
        factor=(1-r**(1/a)) if branch in ('upper','middle') else ((1+r**(1/a)) if branch=='inverted' else (r**(1/a)-1))
        bo=float(r**p*factor/Decimal('100000'))
    result=solve_ratio(bo,direction=direction,branch=branch)
    assert result.ratio==pytest.approx(float(ratio),rel=2e-12)
    assert result.algebraic_residual<1e-10 and result.original_equation_residual<1e-8
    assert not result.station_usable and not result.validated
    json.dumps(asdict(result),allow_nan=False)


def test_three_simultaneous_upward_roots_cannot_be_silently_interchanged():
    results=[solve_ratio(FOLD_BO/2,direction='upward',branch=b) for b in ('upper','middle','inverted')]
    assert results[0].ratio>FOLD_RATIO>results[1].ratio>results[2].ratio>0
    assert results[0].derivative_per_bo<0<results[1].derivative_per_bo
    with pytest.raises(TypeError):solve_ratio(FOLD_BO/2,direction='upward')


@pytest.mark.parametrize('branch',['upper','middle'])
def test_fold_is_a_branch_endpoint_not_a_silent_projection(branch):
    r=solve_ratio(FOLD_BO,direction='upward',branch=branch)
    assert r.tangent and r.ratio==FOLD_RATIO and r.derivative_per_bo is None
    with pytest.raises(ModelLimitError) as info:
        solve_ratio(math.nextafter(FOLD_BO,math.inf),direction='upward',branch=branch)
    assert info.value.criterion=='mixed_branch_fold'
    assert solve_ratio(2*FOLD_BO,direction='upward',branch='inverted').ratio>0


@pytest.mark.parametrize('direction,branch',[('upward','upper'),('downward','unique')])
def test_zero_buoyancy_has_exact_forced_limit(direction,branch):
    assert solve_ratio(0.,direction=direction,branch=branch).ratio==1.


@pytest.mark.parametrize('kwargs',[dict(buoyancy_parameter=-1),dict(buoyancy_parameter=True),dict(buoyancy_parameter=float('nan')),
 dict(buoyancy_parameter=1e-12),dict(buoyancy_parameter=.1),dict(direction='horizontal'),dict(branch='automatic')])
def test_unknown_direction_branch_and_unbounded_parameter_are_rejected(kwargs):
    args=dict(buoyancy_parameter=FOLD_BO/2,direction='upward',branch='upper');args.update(kwargs)
    with pytest.raises((ValueError,DomainError)):solve_ratio(**args)


@pytest.mark.parametrize('branch',['middle','inverted'])
def test_spurious_zero_root_of_transformed_equation_is_not_admitted(branch):
    with pytest.raises(DomainError):solve_ratio(0.,direction='upward',branch=branch)


@pytest.mark.parametrize('buoyancy',[-40.,0.,40.,200.])
def test_fv_profiles_converge_to_an_independent_collocation_solution(buoyancy):
    c=LaminarCase(buoyancy=buoyancy);reference=collocation(c)
    errors=[]
    for n in (64,128,256):
        r=solve_laminar(LaminarCase(buoyancy=buoyancy,cells=n))
        errors.append(abs(r['nusselt']/reference['nusselt']-1))
        assert max(r['checks'].values())<1e-8
        assert r['thermal_gradient_integral']>0 and r['darcy_friction_factor']>0
    assert errors[0]>3*errors[1]>9*errors[2]
    assert errors[-1]<2e-4
    assert reference['maximum_collocation_residual']<3e-10


def test_poiseuille_heat_flux_and_friction_have_analytic_limits():
    r=solve_laminar(LaminarCase(cells=512));x=np.array(r['radius_fraction'])
    assert np.max(np.abs(np.array(r['velocity_ratio'])-2*(1-x*x)))<2e-5
    assert r['nusselt']==pytest.approx(48/11,rel=2e-5)
    assert r['darcy_friction_factor']*r['reynolds']==pytest.approx(64,rel=2e-5)
    assert r['pressure_wall_reference']==pytest.approx(8,rel=2e-5)


def test_laminar_and_turbulent_aiding_responses_are_not_substitutes():
    a=solve_laminar(LaminarCase(buoyancy=40));f=solve_laminar();o=solve_laminar(LaminarCase(buoyancy=-40))
    assert a['nusselt']>f['nusselt']>o['nusselt']
    assert solve_ratio(FOLD_BO/2,direction='upward',branch='upper').ratio<1
    with pytest.raises(DomainError):LaminarCase(reynolds=5693.)


@pytest.mark.parametrize('buoyancy',[-100.,500.])
def test_local_reversal_is_rejected_despite_positive_prescribed_mean(buoyancy):
    assert min(collocation(LaminarCase(buoyancy=buoyancy))['velocity_ratio'])<0
    with pytest.raises(ModelLimitError) as info:solve_laminar(LaminarCase(buoyancy=buoyancy))
    assert info.value.criterion=='local_forward_velocity' and info.value.observed<0


@pytest.mark.parametrize('kwargs',[dict(cells=True),dict(cells=4),dict(cells=513),dict(reynolds=0),dict(reynolds=3000),
 dict(prandtl=float('inf')),dict(buoyancy=True),dict(buoyancy=1001)])
def test_laminar_reference_requires_a_bounded_declared_domain(kwargs):
    with pytest.raises((ValueError,DomainError)):LaminarCase(**kwargs)


def test_dimensional_groups_have_independent_radius_and_diameter_scaling():
    args=dict(diameter_m=.02,mass_flux_kg_m2_s=20.,density_kg_m3=1000.,viscosity_pa_s=.001,
              conductivity_w_m_k=.6,cp_j_kg_k=4200.,expansion_per_k=.0003,heat_flux_w_m2=100.,direction='upward')
    g=tube_groups(**args);d=tube_groups(**{**args,'direction':'downward'})
    assert g['reynolds']==400. and g['prandtl']==pytest.approx(7.,rel=2e-15)
    assert g['laminar_buoyancy']==pytest.approx(9.80665*.0003*100*.01**3/(.6*1e-6*.02))
    assert d['laminar_buoyancy']==-g['laminar_buoyancy']
    assert d['iaea_heat_flux_buoyancy']==g['iaea_heat_flux_buoyancy']
    with pytest.raises(ValueError):tube_groups(**{**args,'heat_flux_w_m2':True})


def test_reported_experimental_box_and_production_eligibility_are_separate():
    args=dict(direction='downward',reynolds_bulk=1e4,prandtl_bulk=3.,reynolds_film=1e4,
              prandtl_film=2.,grashof_film=2e7,rayleigh_film=4e7)
    accepted=assess_opposing(**args);up=assess_opposing(**{**args,'direction':'upward'})
    assert all(r['eligible_for_reported_box_comparison'] for r in accepted.values())
    assert all(not r['station_usable'] and not r['validated'] for r in accepted.values())
    assert all('opposing_direction' in r['failed_checks'] for r in up.values())
    low=assess_opposing(**{**args,'prandtl_bulk':.84,'rayleigh_film':3e5})
    assert not any(r['eligible_for_reported_box_comparison'] for r in low.values() if r['source_table']==2 and r is not low['churchill'])


def test_frozen_station_sample_is_traceable_without_being_a_measurement():
    f=json.loads(files('atomic_model_station_mixed').joinpath('data/station_v63_boundary.json').read_text())
    public=json.loads((Path(__file__).resolve().parents[1]/'docs/benchmarks/station_low_flow_v63.json').read_text())['payload']
    assert public['evidence_sha256'][f['origin']]==f['origin_sha256']
    assert f['time_s']==1475. and not f['station_usable'] and not f['validated']


@pytest.mark.parametrize('branch',['upper','middle','inverted'])
def test_branch_sensitivity_matches_independent_centered_difference(branch):
    bo=FOLD_BO/2;step=bo*1e-4
    derivative=(solve_ratio(bo+step,direction='upward',branch=branch).ratio-solve_ratio(bo-step,direction='upward',branch=branch).ratio)/(2*step)
    assert solve_ratio(bo,direction='upward',branch=branch).derivative_per_bo==pytest.approx(derivative,rel=2e-7)


def test_failed_independent_reference_cannot_become_a_passing_comparison(monkeypatch):
    from types import SimpleNamespace
    from core.status import SolverError
    import atomic_model_station_mixed.reference as reference
    monkeypatch.setattr(reference,'solve_bvp',lambda *a,**k:SimpleNamespace(success=False))
    with pytest.raises(SolverError):reference.collocation()


def test_branches_resolve_the_selected_numerical_window_without_nonfinite_json():
    for bo in np.geomspace(1e-10,1e-3,40):
        branches=('upper','middle','inverted') if bo<FOLD_BO else ('inverted',)
        for branch in branches:
            r=solve_ratio(float(bo),direction='upward',branch=branch)
            assert r.original_equation_residual<1e-8
            json.dumps(asdict(r),allow_nan=False)
        r=solve_ratio(float(bo),direction='downward',branch='unique')
        assert r.ratio>1
    r=solve_ratio(math.nextafter(FOLD_BO,0.),direction='upward',branch='upper')
    assert r.ratio>=FOLD_RATIO


@pytest.mark.parametrize('change',[dict(diameter_m=1e308),dict(viscosity_pa_s=1e-300)])
def test_unrepresentable_dimensional_scales_are_explicit_input_errors(change):
    args=dict(diameter_m=.02,mass_flux_kg_m2_s=20.,density_kg_m3=1000.,viscosity_pa_s=.001,
              conductivity_w_m_k=.6,cp_j_kg_k=4200.,expansion_per_k=.0003,heat_flux_w_m2=100.,direction='upward')
    with pytest.raises(ValueError):tube_groups(**{**args,**change})
