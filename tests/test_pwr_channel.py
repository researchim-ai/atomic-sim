from dataclasses import replace
import json

import numpy as np
import pytest
from core.status import DomainError
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_station import CoreConfig
from physics.hydraulics.network import G
from coupling.pwr_channel import ChannelPWRSession,FluidGrid
from coupling.pwr_cladding import DistributedCladdingSession


@pytest.mark.parametrize('faces',[(0.,4.),(0.,1.,2.,4.),(0.,.3,1.,2.3,3.,4.)])
def test_local_fluid_is_only_owner_with_conserved_global_inventories(faces):
    s=ChannelPWRSession(fluid_grid=FluidGrid(faces));r=s.advance(.005)
    f=r['distributed_fluid'];owned=np.asarray(f['inventory_by_cell'])
    np.testing.assert_array_equal(r['pwr_state'][s.core_alias_index:s.core_alias_index+3],owned.sum(axis=0))
    assert any(m['name']=='lumped_core_fluid_aliases' for m in r['exact_mode_reduction'])
    assert all(not s._passive_column(i) for i in range(s.fluid_index,len(s.state)))
    assert sum(f['volume_m3'])==pytest.approx(18.)
    assert r['energy_relative']<1e-9 and r['mass_relative']<1e-10 and r['boron_relative']<1e-10
    assert abs(r['instantaneous_energy_defect_w'])<1e-3
    assert abs(f['local_boundary_balance_residual'][1])<1e-4
    assert r['distributed_fluid_channels'] and r['local_cladding_connected']
    assert not r['channel_initialization']['stationary_channel_solve']


def test_single_fluid_cell_reduces_to_previous_full_cladding_pwr():
    spec=dict(core=CoreConfig(kind='diffusion',cells=2),max_step_s=.01,rtol=2e-9)
    s=ChannelPWRSession(**spec,fluid_grid=FluidGrid((0.,4.)));old=DistributedCladdingSession(**spec)
    for patch in (None,{'load_fraction':.99},{'scram':True}):
        a=s.advance(.01,patch);b=old.advance(.01,patch)
        np.testing.assert_allclose(a['pwr_state'],b['pwr_state'],rtol=2e-6,atol=1e-5)
        np.testing.assert_allclose(a['distributed_fuel']['temperature_k'],b['distributed_fuel']['temperature_k'],rtol=1e-8)
        assert a['turbine_energy_j']==pytest.approx(b['turbine_energy_j'],rel=1e-8)


def test_internal_hydraulic_kinetic_energy_is_stored_once_and_thermally_compensated():
    s=ChannelPWRSession();r=s.advance(.005)
    expected=sum(.5*j.length_m/j.area_m2*s.state[s.internal_flow_index+i]**2/s.internal_density_ref[i]
        for i,j in enumerate(s.channel.junctions.values()))
    assert s._stored()-DistributedCladdingSession._stored(s)==pytest.approx(expected,abs=1e-4)
    assert max(abs(v) for v in r['distributed_fluid']['internal_kinetic_rate_w'])>1.
    assert abs(r['energy_residual_j'])<.1


def test_full_pwr_boundary_momentum_uses_actual_local_endpoint_states():
    s=ChannelPWRSession();y=s.state.copy();y[s.fluid_index+1]+=1e5;y[s.internal_flow_index-2]-=1e5
    y=s._reconstruct_state(y);rate=s._rhs(0.,y)
    first=s.channel.volumes['cell0'];last=s.channel.volumes[f'cell{s.fluid_cells-1}']
    cold=s.plant.network.volumes['cold_leg'];hot=s.plant.network.volumes['hot_leg']
    for name,up,down,index in [('pump',cold,first,s.pump_index),('core_out',last,hot,s.out_index)]:
        j=s.plant.network.junctions[name]
        expected=(up.pressure_pa-down.pressure_pa-.5*(up.thermo.rho_kg_m3+down.thermo.rho_kg_m3)*G*(down.elevation_m-up.elevation_m)
            -j.friction_dp(up.thermo.rho_kg_m3)+j.pump_pressure_pa)*j.area_m2/j.length_m
        assert rate[index]==pytest.approx(expected,rel=1e-10,abs=1e-6)
    assert abs(s.instantaneous_energy_defect_w)<1e-3


def test_boron_and_enthalpy_outlet_use_last_cell_not_the_core_mean():
    s=ChannelPWRSession();y=s.state.copy();fluid=s._fluid(y)
    fluid[-1,2]*=1.001;fluid[-1,1]+=1e5;y=s._reconstruct_state(y);s._rhs(0.,y)
    f=s.fluid_terms;last=s.channel.volumes[f'cell{s.fluid_cells-1}'];q=y[s.out_index]
    assert f['port_outlet_rates'][1]==pytest.approx(q*last.thermo.h_j_kg)
    assert f['port_outlet_rates'][2]==pytest.approx(q*last.boron_fraction)
    assert f['hot_leg_correction'][2]!=0
    assert abs(f['local_boundary_balance_residual'][2])<1e-10


def test_local_coolant_gradients_feed_wall_and_fuel_dynamics():
    s=ChannelPWRSession();r=s.advance(.01)
    assert np.ptp(r['distributed_fluid']['temperature_k'])>.1
    assert np.ptp(r['distributed_cladding']['temperature_k'])>1e-3
    assert np.ptp(r['core']['local_fuel_feedback_k'])>1e-6
    assert r['distributed_fluid']['contact_entropy_w_k']>=0


@pytest.mark.parametrize('kind',['mass','boron','energy','reverse_pump','reverse_internal','alias'])
def test_local_domain_violation_cannot_hide_in_aggregate(kind):
    s=ChannelPWRSession();y=s.state.copy()
    if kind=='mass':y[s.fluid_index]=-1
    elif kind=='boron':y[s.fluid_index+2]=-1e-8
    elif kind=='energy':y[s.fluid_index+1]*=.01
    elif kind=='reverse_pump':y[s.pump_index]=-1.
    elif kind=='reverse_internal':y[s.internal_flow_index]=-1.
    else:y[s.core_alias_index]+=.1
    if kind!='alias':y=s._reconstruct_state(y)
    with pytest.raises((DomainError,ValueError)):s._admit(y)


@pytest.mark.parametrize('grid',[FluidGrid((0.,3.)),[0.,4.],None])
def test_fluid_geometry_and_explicit_type(grid):
    if grid is None:
        assert ChannelPWRSession().fluid_cells==3
    else:
        with pytest.raises((DomainError,ValueError)):ChannelPWRSession(fluid_grid=grid)


@pytest.mark.parametrize('faces,k',[([0.,4.],0.),((0.,1.,1.,4.),0.),((0.,4.),1.),((0.,1.,4.),-1.)])
def test_invalid_fluid_grid_refused(faces,k):
    with pytest.raises((DomainError,ValueError)):FluidGrid(faces,k)


def test_json_restart_is_exact_after_spatial_cooldown_and_scram():
    s=ChannelPWRSession(core=CoreConfig(kind='diffusion',cells=2))
    s.advance(.005);saved=json.loads(json.dumps(s.checkpoint()));t=ChannelPWRSession.restore(saved)
    assert s.advance(.005,{'scram':True})==t.advance(.005,{'scram':True})
    assert s.checkpoint()==t.checkpoint()
    with pytest.raises(ValueError):DistributedCladdingSession.restore(saved)
    with pytest.raises(ValueError):ChannelPWRSession.restore(DistributedCladdingSession().checkpoint())


@pytest.mark.parametrize('kind',['source','extra','format','grid','history','snapshot','initial'])
def test_resealed_checkpoint_corruption_refused(kind):
    s=ChannelPWRSession();p=unseal(s.checkpoint())
    if kind=='source':p['source']['adapter_sha256']='0'*64
    elif kind=='extra':p['extra']=True
    elif kind=='format':p['format']='other'
    elif kind=='grid':p['fluid_grid']['extra']=1
    elif kind=='history':p['history']=[{'duration_s':.01,'bad':True}]
    elif kind=='initial':
        old=DistributedCladdingSession();old.advance(.001);p['initial_pwr']=old.checkpoint()
    else:p['snapshot']['distributed_fluid']['temperature_k'][0]+=1
    with pytest.raises(ValueError):ChannelPWRSession.restore(seal(p))


def test_late_rejection_rolls_back_coupled_fluids_commands_and_events(monkeypatch):
    s=ChannelPWRSession();saved=s.checkpoint();original=ChannelPWRSession._admit
    def reject(self,y):
        original(self,y)
        if self.time>0:raise DomainError('deliberate coupled rejection')
    monkeypatch.setattr(ChannelPWRSession,'_admit',reject)
    with pytest.raises(DomainError,match='deliberate'):s.advance(.005,{'scram':True})
    assert s.checkpoint()==saved


def test_channel_registry_discloses_constructed_geometry():
    from atomic_model_station.foundation import validate_registry
    registry=ChannelPWRSession().parameter_registry();assert validate_registry(registry)
    assert registry['parameters']['distributed_fluid.flow_area']['family']=='illustrative'
