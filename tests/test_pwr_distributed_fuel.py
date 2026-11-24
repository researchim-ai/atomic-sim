from dataclasses import replace
import json
import numpy as np
import pytest
from core.status import DomainError
from coupling.pwr_fuel import FuelGrid,DistributedFuelSession,local_heat
from atomic_model_station import PWRCoreSession,CoreConfig
from atomic_model_multiphysics.provenance import seal,unseal
from physics.depletion.chains import DecayHeat


def test_local_decay_equations_sum_to_existing_global_model():
    decay=DecayHeat.illustrative();q=np.array([1e8,4e8,2e8]);h=np.arange(1,10).reshape(3,3)*1e9
    heat,rate=local_heat(decay,q,h)
    np.testing.assert_allclose(rate.sum(axis=1),decay.derivatives(h.sum(axis=1),q.sum()),rtol=1e-14)
    assert heat.sum()==pytest.approx(decay.total_power_w(q.sum(),h.sum(axis=1)),rel=1e-14)
    assert q.sum()-heat.sum()==pytest.approx(rate.sum(),rel=1e-14)


def test_zero_fission_retains_spatial_decay_and_matches_exponential_release():
    decay=DecayHeat.illustrative();initial=np.array([[1e9,2e9],[3e9,4e9],[5e9,6e9]])
    t=10.;stored=initial*np.exp(-decay.decay_constants[:,None]*t)
    heat,rate=local_heat(decay,[0,0],stored)
    np.testing.assert_allclose(heat,(decay.decay_constants[:,None]*stored).sum(axis=0))
    np.testing.assert_allclose(rate,-decay.decay_constants[:,None]*stored)
    assert heat.sum()>0 and np.ptp(heat)>0


@pytest.mark.parametrize('q,h',[([-1],[[1],[1],[1]]),([1],[[1],[1]]),([1],[[1],[-1],[1]]),
    ([np.nan],[[1],[1],[1]]),([1],[[1+0j],[1],[1]])])
def test_invalid_local_heat_states_refused(q,h):
    with pytest.raises((ValueError,DomainError)):local_heat(DecayHeat.illustrative(),q,h)


@pytest.mark.parametrize('faces',[(),(0,),[0,4],(0,0,4),(0,np.nan,4),tuple(np.linspace(0,4,18))])
def test_invalid_fuel_geometry_refused(faces):
    with pytest.raises((ValueError,DomainError)):FuelGrid(faces)


@pytest.mark.parametrize('cells,faces',[(2,(0.,.7,2.2,4.)),(4,(0.,4.)),(4,(0.,.7,2.2,4.)),(4,(0.,.2,.9,2.1,3.,4.))])
def test_connected_pwr_has_single_local_owners_and_global_fission_boundary(cells,faces):
    s=DistributedFuelSession(core=CoreConfig(kind='diffusion',cells=cells,axial_bias_reactivity=1e-4),fuel_grid=FuelGrid(faces))
    r=s.advance(.02);f=r['distributed_fuel']
    assert r['energy_relative']<1e-9 and r['mass_relative']<1e-10 and r['boron_relative']<1e-10
    assert abs(r['instantaneous_energy_defect_w'])/f['fission_w']<1e-10
    assert abs(f['nuclear_storage_identity_residual_w'])/f['fission_w']<1e-12
    assert abs(f['fuel_transfer_residual_w'])/f['thermal_w']<1e-12
    assert r['core']['neutron_precursor_inventory']['relative_defect']<1e-10
    assert len(f['temperature_k'])==len(faces)-1
    assert len(f['decay_stored_j'])==3 and len(f['decay_stored_j'][0])==len(faces)-1
    np.testing.assert_array_equal(r['pwr_state'][7:10],np.asarray(f['decay_stored_j']).sum(axis=1))
    assert r['pwr_state'][10]==s.fuel_weights@f['temperature_k']
    assert any(m['name']=='lumped_fuel_and_decay_aliases' for m in r['exact_mode_reduction'])
    assert all(not s._passive_column(i) for i in range(s.decay_index,s.exchange_index))
    assert s._passive_column(s.exchange_index)
    assert not r['distributed_fluid_channels'] and not f['validated']


def test_uniform_limit_preserves_inherited_thermal_hydraulic_trajectory():
    cfg=CoreConfig(kind='diffusion',cells=2)
    a=DistributedFuelSession(core=cfg,rtol=2e-8,max_step_s=.05)
    b=PWRCoreSession(core=cfg,rtol=2e-8,max_step_s=.05)
    x=a.advance(.05,{'load_fraction':.99});y=b.advance(.05,{'load_fraction':.99})
    np.testing.assert_allclose(x['pwr_state'],y['pwr_state'],rtol=2e-7,atol=1e-5)
    assert x['turbine_energy_j']==pytest.approx(y['turbine_energy_j'],rel=1e-8)
    assert x['powers']['nuclear_heat_w']==pytest.approx(y['powers']['nuclear_heat_w'],rel=1e-8)
    assert a._stored()-PWRCoreSession._stored(a)==pytest.approx(x['distributed_fuel']['decay_stored_total_j'],rel=1e-13)


def test_bias_creates_local_feedback_and_scram_draws_stored_heat():
    s=DistributedFuelSession(core=CoreConfig(kind='diffusion',cells=4,axial_bias_reactivity=1e-4),rtol=2e-8,max_step_s=.05)
    s.advance(.05);r=s.advance(.05,{'scram':True});f=r['distributed_fuel']
    assert np.ptp(f['temperature_k'])>1e-4
    assert np.ptp(r['core']['local_fuel_feedback_k'])>1e-4
    assert f['thermal_w']>f['fission_w'] and f['nuclear_storage_rate_w']<0
    assert f['decay_power_w']>0 and abs(f['decay_exchange_residual_j'])<.1
    assert r['energy_relative']<1e-9


def test_negative_local_reservoir_cannot_hide_in_positive_sum():
    s=DistributedFuelSession();y=s.state.copy();y[s.decay_index]=-1
    y=s._reconstruct_state(y)
    assert y[7]>0
    with pytest.raises(ValueError,match='reservoir'):s._admit(y)


def test_extreme_local_fuel_temperature_cannot_hide_in_bulk_mean():
    s=DistributedFuelSession();y=s.state.copy();y[s.fuel_index]=3000
    y=s._reconstruct_state(y)
    assert y[10]<2800
    with pytest.raises(ValueError,match='temperature'):s._admit(y)
    with pytest.raises(ValueError,match='thermal state'):s._rhs(0.,y)


def test_json_restart_across_scram_is_exact():
    s=DistributedFuelSession(core=CoreConfig(kind='diffusion',cells=2))
    s.advance(.02);saved=json.loads(json.dumps(s.checkpoint()));t=DistributedFuelSession.restore(saved)
    assert t.advance(.02,{'scram':True})==s.advance(.02,{'scram':True})
    assert t.checkpoint()==s.checkpoint()


def test_late_rejection_does_not_commit_controls_events_or_states(monkeypatch):
    s=DistributedFuelSession();saved=s.checkpoint();original=DistributedFuelSession._admit
    def reject(self,y):
        if self.time>0:raise ValueError('deliberate final rejection')
        return original(self,y)
    monkeypatch.setattr(DistributedFuelSession,'_admit',reject)
    with pytest.raises(ValueError,match='final rejection'):s.advance(.02,{'scram':True})
    assert s.checkpoint()==saved


@pytest.mark.parametrize('kind',['operator','format','extra','history','snapshot'])
def test_corrupted_resealed_checkpoint_is_refused(kind):
    s=DistributedFuelSession();p=unseal(s.checkpoint())
    if kind=='operator':p['distributed_fuel_source']['exchange_sha256']='0'*64
    elif kind=='format':p['format']='other'
    elif kind=='extra':p['unknown']=True
    elif kind=='history':p['history']=[{'duration_s':.01,'bad':True}]
    else:p['snapshot']['distributed_fuel']['temperature_k'][0]+=1.
    with pytest.raises(ValueError):DistributedFuelSession.restore(seal(p))


def test_registry_exposes_local_capacities_and_does_not_claim_calibration():
    from atomic_model_station.foundation import validate_registry
    s=DistributedFuelSession();registry=s.parameter_registry()
    assert validate_registry(registry)
    local={k:v for k,v in registry['parameters'].items() if k.startswith('distributed_fuel.')}
    assert len(local)==3*s.fuel_cells and all(v['family']=='illustrative' for v in local.values())


def test_stationary_neutron_balance_retains_absolute_error_and_finite_scale():
    r=DistributedFuelSession().snapshot()['core']['neutron_precursor_inventory']
    assert r['absolute_defect']<1e-12 and r['relative_defect']<1e-10
    assert r['balance_scale']>abs(r['rate'])


def test_neutron_leak_is_detected_despite_stationary_normalization():
    s=DistributedFuelSession();s.neutron_core.operator[0,0]+=.001
    r=s.snapshot()['core']['neutron_precursor_inventory']
    assert r['absolute_defect']>1e-8 and r['relative_defect']>1e-10


def test_staged_backend_keeps_film_header_and_local_decay_connected():
    from coupling.pwr_fuel import StagedDistributedFuelSession
    s=StagedDistributedFuelSession(core=CoreConfig(kind='diffusion',cells=2))
    r=s.advance(.01)
    assert r['distributed_fuel_backend']=='staged' and r['finite_steam_header_connected']
    assert r['staged_turbine'] and r['energy_relative']<1e-9
    saved=json.loads(json.dumps(s.checkpoint()));t=StagedDistributedFuelSession.restore(saved)
    assert s.advance(.01,{'scram':True})==t.advance(.01,{'scram':True})
    with pytest.raises(ValueError):DistributedFuelSession.restore(saved)


def test_core_checkpoint_cannot_be_replayed_as_staged_backend():
    from coupling.pwr_fuel import StagedDistributedFuelSession
    with pytest.raises(ValueError):StagedDistributedFuelSession.restore(DistributedFuelSession().checkpoint())


def test_staged_heated_length_must_match_neutron_fuel_coordinate():
    from coupling.pwr_fuel import StagedDistributedFuelSession
    with pytest.raises(ValueError,match='heated length'):
        StagedDistributedFuelSession(core=CoreConfig(kind='diffusion',cells=2,length_m=6),fuel_grid=FuelGrid((0.,3.,6.)))
