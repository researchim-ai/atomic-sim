from copy import deepcopy
import numpy as np
import pytest
from scipy.linalg import eig
from atomic_model_spatial import Model
from atomic_model_spatial.cases import slab
from atomic_model_spatial.reference import dense_operators,dense_k
from atomic_model_spatial_response import solve_pair,sensitivities,station_core_response
from atomic_model_spatial_response import model as response


def case(n=5):
    config=slab(n,heterogeneous=True)
    config['faces_m']=(np.linspace(0,1,n+1)**1.3).tolist()
    return config


@pytest.mark.parametrize('n',[1,3,7])
def test_pair_matches_independent_dense_assembly_and_generalized_adjoint(n):
    model=Model(case(n));before=model.config();r=solve_pair(model,tolerance=1e-12)
    a,f,_=dense_operators(model);values,left,right=eig(f,a,left=True,right=True)
    index=int(np.argmax(values.real));k=values[index].real
    phi=right[:,index].real;phi*=1/(np.sum(f@phi))
    psi=left[:,index].real;psi/=float(psi@(f@phi))
    assert r['k']==pytest.approx(k,rel=1e-10)
    np.testing.assert_allclose(np.ravel(r['flux']),phi,rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(np.ravel(r['adjoint']),psi,rtol=1e-9,atol=1e-10)
    assert model.config()==before and r['fission_pairing']==pytest.approx(1.,abs=1e-12)


@pytest.mark.parametrize('field',['nu_fission_per_m','absorption_per_m','scatter_per_m','diffusion_m'])
@pytest.mark.parametrize('boundaries',[['zero_flux','zero_flux'],['reflecting','zero_flux'],['reflecting','reflecting']])
def test_every_local_gradient_matches_independently_assembled_dense_differences(field,boundaries):
    config=case(4);config['boundaries']=boundaries;model=Model(config)
    result=sensitivities(model,tolerance=1e-12);analytic=np.asarray(result['gradients'][field])
    values=np.asarray(config[field])
    for index in np.ndindex(values.shape):
        if field=='scatter_per_m' and index[0]==index[1]:continue
        delta=max(values[index],.001)*2e-5
        plus=deepcopy(config);minus=deepcopy(config)
        a=np.array(plus[field]);b=np.array(minus[field]);a[index]+=delta;b[index]-=delta
        plus[field]=a.tolist();minus[field]=b.tolist()
        difference=(dense_k(Model(plus))-dense_k(Model(minus)))/(2*delta)
        assert analytic[index]==pytest.approx(difference,rel=2e-5,abs=1e-7),(field,index)


def test_scaling_identities_for_birth_and_complete_loss_operator():
    result=sensitivities(Model(case(6)),tolerance=1e-12);e=result['elasticities']
    assert np.sum(e['nu_fission_per_m'])==pytest.approx(1.,abs=1e-10)
    assert sum(np.sum(e[k]) for k in ('absorption_per_m','scatter_per_m','diffusion_m'))==pytest.approx(-1.,abs=1e-10)


def test_single_cell_one_group_matches_closed_form_boundary_leakage_response():
    config=slab(1);config.update(faces_m=[0.,2.],diffusion_m=[[.04]],absorption_per_m=[[1.2]],
        nu_fission_per_m=[[.9]],scatter_per_m=[[[0.]]],speed_m_s=[[100.]],chi_prompt=[[1.]],chi_delayed=[[1.]])
    r=sensitivities(Model(config));denominator=1.2+4*.04/2**2
    assert r['k']==pytest.approx(.9/denominator,rel=1e-12)
    assert r['gradients']['diffusion_m'][0][0]==pytest.approx(-.9*4/2**2/denominator**2,rel=1e-12)
    assert r['gradients']['absorption_per_m'][0][0]==pytest.approx(-.9/denominator**2,rel=1e-12)


def test_power_normalization_preserves_gradients_and_adjoint_pairing():
    model=Model(case(4));plain=sensitivities(model,tolerance=1e-12)
    weights=np.arange(1,9,dtype=float).reshape(2,4)*1e6
    power=sensitivities(model,tolerance=1e-12,power_weights_w=weights,power_w=3e9)
    assert np.sum(weights*np.array(power['flux']))==pytest.approx(3e9,rel=1e-12)
    assert power['fission_pairing']==pytest.approx(1.,abs=1e-12)
    for key in plain['gradients']:
        np.testing.assert_allclose(power['gradients'][key],plain['gradients'][key],rtol=1e-10,atol=1e-10)


def test_reflecting_homogeneous_shape_has_no_diffusion_response():
    model=Model(slab(5,boundaries=('reflecting','reflecting')))
    result=sensitivities(model,tolerance=1e-12)
    np.testing.assert_allclose(result['gradients']['diffusion_m'],0.,atol=1e-20)


def test_disconnected_modes_are_refused_instead_of_arbitrary_derivatives():
    config=slab(2);config['scatter_per_m']=np.zeros((2,2,2)).tolist()
    config['chi_prompt']=[[1.,1.],[0.,0.]];config['chi_delayed']=[[1.,1.],[0.,0.]]
    with pytest.raises(ValueError,match='Irreducible'):solve_pair(Model(config))


@pytest.mark.parametrize('kwargs',[dict(tolerance=True),dict(max_iterations=0),dict(power_w=1.),
    dict(power_weights_w=[[1.]]),dict(power_weights_w=np.zeros((2,3)),power_w=1.)])
def test_invalid_solver_or_normalization_parameters_refused(kwargs):
    with pytest.raises(ValueError):solve_pair(Model(case(3)),**kwargs)


def test_iteration_exhaustion_is_not_promoted_to_accepted_response():
    with pytest.raises(ValueError,match='budget exhausted'):solve_pair(Model(case(5)),max_iterations=1)


def test_station_adapter_matches_actual_declared_power_projection():
    from atomic_model_station.core import DiffusionCore,CoreConfig
    from physics.neutronics.point_kinetics import PointKinetics
    core=DiffusionCore(CoreConfig(kind='diffusion',cells=4),PointKinetics.thermal_u235(power_nominal_w=3e9))
    before=core.operator.copy();r=station_core_response(core)
    assert r['k']==pytest.approx(1.,abs=1e-10)
    assert sum(r['fission_by_cell_w'])==pytest.approx(core.kinetics.power_nominal_w,rel=1e-12)
    assert r['relative_power_projection_defect']<=1e-12 and not r['local_thermal_feedback']
    np.testing.assert_array_equal(before,core.operator)


def test_nonzero_station_bias_cannot_use_zero_bias_reference():
    from atomic_model_station.core import DiffusionCore,CoreConfig
    from physics.neutronics.point_kinetics import PointKinetics
    core=DiffusionCore(CoreConfig(kind='diffusion',cells=2,axial_bias_reactivity=1e-5),PointKinetics.thermal_u235())
    with pytest.raises(ValueError,match='zero-bias'):station_core_response(core)


def test_source_change_refuses_pair(monkeypatch):
    versions=iter([1,2]);monkeypatch.setattr(response,'identity',lambda:next(versions))
    with pytest.raises(ValueError,match='source changed'):solve_pair(Model(case(3)))
