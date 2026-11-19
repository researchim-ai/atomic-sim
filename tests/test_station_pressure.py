import json
from pathlib import Path
import numpy as np
import pytest
from atomic_model_station_pressure import pressure_operators,solve_pressure,manufactured_pressure
from atomic_model_station_quadrature.data import load_fixture
from atomic_model_multiphysics.provenance import unseal
from scripts import qualify_station_pressure as study


def radius():return np.array(load_fixture()['rules']['32']['radius'])


@pytest.mark.parametrize('m,power',[(0,4),(1,1),(2,2),(4,4)])
def test_polynomial_cylindrical_operators_against_analytic_derivatives(m,power):
    r=radius();op=pressure_operators(r,3,m,.628319);p=r**power
    first=power*r**(power-1);second=power*(power-1)*r**(power-2)
    exact=(power*power-m*m)*r**(power-2)-(3*.628319)**2*p
    np.testing.assert_allclose(op.radial_first@p,first,atol=1e-10)
    np.testing.assert_allclose(op.radial_second@p,second,atol=1e-8)
    np.testing.assert_allclose(op.laplacian_fd@p,exact,atol=1e-8)
    np.testing.assert_allclose(op.laplacian_composed@p,exact,atol=1e-8)


@pytest.mark.parametrize('m,k',[(0,0),(1,3),(2,-3),(4,3)])
def test_compatible_pressure_solve_recovers_manufactured_pressure(m,k):
    r=radius();op=pressure_operators(r,k,m,.628319);p,dr,lap=manufactured_pressure(r,m,k,.628319)
    rhs=op.laplacian_composed@p;boundary=dict(wall_derivative=(op.radial_first@p)[-1]) if m or k else dict(wall_gauge=p[-1])
    new,audit=solve_pressure(op,rhs,composed=True,**boundary)
    np.testing.assert_allclose(new,p,atol=1e-9)
    assert audit['interior_div_grad_residual']<1e-8
    assert not audit['no_slip_preservation_qualified'] and not audit['time_integrator_qualified']


def test_native_style_laplacian_does_not_cancel_discrete_gradient_divergence():
    r=radius();op=pressure_operators(r,3,2,.628319);p,_,_=manufactured_pressure(r,2,3,.628319)
    rhs=op.laplacian_composed@p;boundary=dict(wall_derivative=(op.radial_first@p)[-1])
    original,a=solve_pressure(op,rhs,composed=False,**boundary)
    compatible,b=solve_pressure(op,rhs,composed=True,**boundary)
    assert a['linear_residual']<1e-8 and a['interior_div_grad_residual']>1e-5
    assert b['interior_div_grad_residual']<1e-9
    assert max(abs(original-p))>1e-8 and max(abs(compatible-p))<1e-10


def test_neumann_pressure_does_not_imply_tangential_no_slip():
    op=pressure_operators(radius(),3,2,.628319)
    p,audit=solve_pressure(op,np.ones(len(op.radius)),composed=True)
    gradient=(op.gradient@p).reshape(3,-1)
    assert abs(gradient[0,-1])<1e-9 and max(abs(gradient[1:,-1]))>1e-3
    assert not audit['no_slip_preservation_qualified']


def test_mean_pressure_gauge_is_explicit_and_gradient_is_gauge_invariant():
    op=pressure_operators(radius(),0,0,.628319);p,_,_=manufactured_pressure(radius(),0,0,.628319)
    rhs=op.laplacian_composed@p
    a,_=solve_pressure(op,rhs,composed=True,wall_gauge=0.)
    b,_=solve_pressure(op,rhs,composed=True,wall_gauge=2.)
    np.testing.assert_allclose(b-a,2,atol=1e-9)
    np.testing.assert_allclose(op.gradient@a,op.gradient@b,atol=1e-9)


@pytest.mark.parametrize('change',[dict(k=True),dict(k=128),dict(m=-1),dict(m=1.),dict(alpha=0),dict(alpha=float('nan')),dict(alpha=True)])
def test_invalid_modes_are_rejected(change):
    args=dict(k=1,m=1,alpha=.628319);args.update(change)
    with pytest.raises(ValueError):pressure_operators(radius(),**args)


@pytest.mark.parametrize('bad',['nan','shape','choice','boundary_nan','conflicting_gauge'])
def test_invalid_pressure_problem_is_rejected(bad):
    op=pressure_operators(radius(),3,2,.628319);rhs=np.ones(len(op.radius));args=dict(composed=True)
    if bad=='nan':rhs[0]=np.nan
    if bad=='shape':rhs=rhs[:-1]
    if bad=='choice':args['composed']=1
    if bad=='boundary_nan':args['wall_derivative']=np.nan
    if bad=='conflicting_gauge':args['wall_gauge']=1
    with pytest.raises(ValueError):solve_pressure(op,rhs,**args)


def test_pressure_export_and_byte_exact_repeat(tmp_path):
    a=study.run(tmp_path/'source');b=study.run(tmp_path/'repeat',tmp_path/'source')
    assert a==b and a['passed'] and a['manufactured_cases']==12 and a['native_meshes']==3
    assert a['maximum_relative_native_coefficient_error']<1e-9
    assert a['maximum_relative_compatible_interior_residual']<1e-9
    assert not a['native_fd_equals_composed'] and not a['external_time_integrator_modified']
    assert not a['station_usable'] and not a['dynamic_continuity_qualified']
    assert a['closed_tasks']==[]


def test_missing_or_duplicate_native_coefficients_are_rejected():
    from importlib.resources import files
    data=unseal(json.loads(files('atomic_model_station_pressure').joinpath('data/native_operators.json').read_text()))
    row=data['records']['32'];bad=dict(row,bands=row['bands'][:-1])
    with pytest.raises(ValueError):study.native_matrices(bad,1)
    bad=dict(row,bands=[*row['bands'],row['bands'][0]])
    with pytest.raises(ValueError):study.native_matrices(bad,1)
