import json
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from atomic_model_station import CoreConfig,PWRCoreSession,PWRPowerSession,PointCore,DiffusionCore
from physics.neutronics.point_kinetics import PointKinetics
from atomic_model_spatial.reference import dense_operators


@pytest.mark.parametrize('cells',[2,3,4,6,8])
def test_uniform_spatial_point_limit_and_local_precursor_units(cells):
    k=PointKinetics.thermal_u235(power_nominal_w=3e9);point=PointCore(k)
    spatial=DiffusionCore(CoreConfig(kind='diffusion',cells=cells),k);p=k.initial_state()
    y=spatial.lift(p);np.testing.assert_allclose(spatial.project(y),p,rtol=1e-14)
    for rho in (-.02,0.,1e-4):
        np.testing.assert_allclose(spatial.project(spatial.rhs(y,rho)),point.rhs(p,rho),atol=1e-10,rtol=1e-11)
    assert spatial.size==7*cells and spatial.eigenpair['k']==pytest.approx(1.,abs=1e-12)


def test_arbitrary_spatial_profile_preserves_global_neutron_precursor_balance():
    k=PointKinetics.thermal_u235();s=DiffusionCore(CoreConfig(kind='diffusion',cells=4,axial_bias_reactivity=1e-4),k)
    y=s.lift(k.initial_state());y[:4]=[.7,1.2,.9,1.1]
    assert s.inventory_balance(y,-.003)['relative_defect']<1e-11


def test_frozen_spatial_generator_matches_independently_assembled_dense_reference():
    k=PointKinetics.thermal_u235();s=DiffusionCore(CoreConfig(kind='diffusion',cells=4,axial_bias_reactivity=1e-4),k)
    _,_,matrix=dense_operators(s.model);rho=-.002
    matrix[np.arange(4),np.arange(4)]+=(rho+s.bias)/k.generation_time_s
    y=s.lift(k.initial_state());reference=expm(matrix*.1)@y
    solved=solve_ivp(lambda t,y:s.rhs(y,rho),(0.,.1),y,method='Radau',rtol=1e-10,atol=1e-12).y[:,-1]
    np.testing.assert_allclose(solved,reference,rtol=2e-8,atol=1e-11)


def test_point_adapter_coupled_limit_matches_existing_station_without_duplicate_heat():
    a=PWRCoreSession(max_step_s=.2);b=PWRPowerSession(max_step_s=.2)
    x=a.advance(.3,{'load_fraction':.95});y=b.advance(.3,{'load_fraction':.95})
    np.testing.assert_allclose(x['pwr_state'],y['pwr_state'],rtol=2e-6,atol=1e-5)
    assert x['powers']['nuclear_heat_w']==pytest.approx(y['powers']['nuclear_heat_w'],rel=1e-7)
    assert x['energy_relative']<1e-8 and x['core']['fission_projection_residual_w']==0.


def test_spatial_feedback_changes_profile_and_preserves_total_thermal_source():
    s=PWRCoreSession(core=CoreConfig(kind='diffusion',cells=4,axial_bias_reactivity=1e-4),max_step_s=.2)
    r=s.advance(.2);c=r['core']
    assert c['fission_by_cell_w'][0]<c['fission_by_cell_w'][-1]
    assert sum(c['fission_by_cell_w'])==pytest.approx(c['fission_total_w'],rel=1e-13)
    assert abs(c['fission_projection_residual_w'])/c['fission_total_w']<1e-12
    assert r['energy_relative']<1e-8 and r['mass_relative']<1e-10
    assert c['neutron_precursor_inventory']['relative_defect']<1e-10
    assert c['nuclear_data_family']=='illustrative' and not c['validated']


def test_owned_core_states_and_derived_aliases_survive_json_restart_and_power_event():
    s=PWRCoreSession(core=CoreConfig(kind='diffusion',cells=2),max_step_s=.2)
    s.advance(.1,{'grid_available':False});saved=json.loads(json.dumps(s.checkpoint()));t=PWRCoreSession.restore(saved)
    r=s.advance(.1);assert t.advance(.1)==r
    np.testing.assert_allclose(r['pwr_state'][:7],r['core']['derived_point_state'],rtol=0,atol=0)
    assert any(mode['name']=='legacy_neutron_aliases' for mode in r['exact_mode_reduction'])
    assert len(r['feed_inventory'])==7
    with pytest.raises((ValueError,KeyError)):PWRPowerSession.restore(saved)


@pytest.mark.parametrize('kwargs',[{'kind':'unknown'},{'cells':2},{'kind':'diffusion','cells':1},{'kind':'diffusion','cells':True},{'axial_bias_reactivity':1e-4}])
def test_invalid_core_configuration_refused(kwargs):
    with pytest.raises(ValueError):CoreConfig(**kwargs)


def test_negative_local_population_cannot_be_hidden_by_positive_aggregate():
    s=PWRCoreSession(core=CoreConfig(kind='diffusion',cells=2));y=s.state.copy();y[s.core_index]=-.1
    y=s._reconstruct_state(y)
    assert y[0]>0.
    with pytest.raises(ValueError,match='population'):s._admit(y)


def test_core_registry_and_checkpoint_bind_imported_spatial_sources():
    from atomic_model_station.foundation import validate_registry
    from atomic_model_multiphysics.provenance import unseal,seal
    s=PWRCoreSession(core=CoreConfig(kind='diffusion',cells=2));assert validate_registry(s.parameter_registry())
    saved=unseal(s.checkpoint());assert 'spatial' in saved['core_source']
    saved['core_source']['spatial']['sources']['model.py']='0'*64
    with pytest.raises(ValueError,match='source'):PWRCoreSession.restore(seal(saved))
