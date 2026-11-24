"""Local thermal storage: independent balances, uniform limit and strict replay."""
import json

import numpy as np
import pytest
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_station import CoreConfig
from coupling.pwr_cladding import DistributedCladdingSession
from coupling.pwr_fuel import DistributedFuelSession,FuelGrid
from physics.materials.iapws import saturation_temperature_k


@pytest.mark.parametrize('faces',[(0.,4.),(0.,.7,2.2,4.),(0.,.2,.9,2.1,3.,4.)])
def test_local_storage_is_not_counted_twice_and_alias_is_not_integrated(faces):
    s=DistributedCladdingSession(fuel_grid=FuelGrid(faces));r=s.advance(.02)
    c=r['distributed_cladding']
    assert c['stored_energy_j']==pytest.approx(s.config.clad_capacity_j_k*r['clad_temperature_k'],rel=1e-14)
    assert s._stored()==DistributedFuelSession._stored(s)
    assert s.cladding_capacities.sum()==pytest.approx(s.config.clad_capacity_j_k)
    assert s.coolant_conductances.sum()==pytest.approx(s.plant.clad_coolant_ua)
    assert any(m['name']=='lumped_cladding_alias' for m in r['exact_mode_reduction'])
    assert all(not s._passive_column(i) for i in range(s.cladding_index,len(s.state)))
    assert r['energy_relative']<1e-9 and abs(r['instantaneous_energy_defect_w'])<1e-3
    assert not r['distributed_fluid_channels']


def test_nonuniform_local_thermal_rhs_matches_independent_resistance_network():
    s=DistributedCladdingSession();y=s.state.copy()
    y[s.fuel_index:s.decay_index]+=[20.,-10.,3.]
    y[s.cladding_index:]+=[1.,-.8,.3]
    y=s._reconstruct_state(y);rate=s._rhs(0,y)
    m=s.fuel_cells;w=s.fuel_weights
    # Independent 2M-node capacitance/resistance network with a prescribed
    # common coolant and current nuclear sources. No adapter rate is used.
    matrix=np.zeros((2*m,2*m));forcing=np.zeros(2*m)
    for j in range(m):
        cf=s.plant.fuel_capacity*w[j];cc=s.config.clad_capacity_j_k*w[j]
        a=s.plant.fuel_clad_ua*w[j];b=s.plant.clad_coolant_ua*w[j]
        matrix[j,j]=-a/cf;matrix[j,m+j]=a/cf
        matrix[m+j,j]=a/cc;matrix[m+j,m+j]=-(a+b)/cc
        forcing[j]=s.fuel_terms['thermal_source_by_cell_w'][j]/cf
        forcing[m+j]=b*s.plant.network.volumes['core'].temperature_k/cc
    z=np.r_[y[s.fuel_index:s.decay_index],y[s.cladding_index:]]
    expected=matrix@z+forcing
    actual=np.r_[rate[s.fuel_index:s.decay_index],rate[s.cladding_index:]]
    np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-9)
    capacities=np.r_[s.fuel_capacities,s.cladding_capacities]
    assert capacities@actual+sum(s.cladding_terms['clad_to_coolant_by_cell_w'])==pytest.approx(s.fuel_terms['thermal_w'],abs=1e-5)
    assert min(s.cladding_terms['conductive_entropy_generation_w_k'])>=0
    assert max(abs(np.array(s.cladding_terms['local_energy_residual_w'])))<1e-5


def test_uniform_limit_matches_previous_pwr_over_load_change_and_scram():
    spec=dict(core=CoreConfig(kind='diffusion',cells=2),max_step_s=.02,rtol=5e-9)
    a=DistributedCladdingSession(**spec);b=DistributedFuelSession(**spec)
    for patch in (None,{'load_fraction':.99},{'scram':True}):
        x=a.advance(.03,patch);y=b.advance(.03,patch)
        np.testing.assert_allclose(x['pwr_state'],y['pwr_state'],rtol=2e-6,atol=1e-5)
        assert x['clad_temperature_k']==pytest.approx(y['clad_temperature_k'],rel=1e-8)
        assert x['energy_relative']<1e-9


def test_bias_generates_a_local_cladding_profile_and_heat_storage():
    s=DistributedCladdingSession(core=CoreConfig(kind='diffusion',cells=4,axial_bias_reactivity=1e-4))
    r=s.advance(.05);c=r['distributed_cladding']
    assert np.ptp(c['temperature_k'])>1e-6
    assert np.ptp(c['fuel_to_clad_by_cell_w'])>1e4
    assert r['clad_to_coolant_w']==pytest.approx(sum(c['clad_to_coolant_by_cell_w']),rel=1e-14)
    assert c['alias_residual_k']==0 and r['energy_relative']<1e-9


@pytest.mark.parametrize('value',[299.,float('nan'),float('inf'),'saturation'])
def test_local_wall_limit_cannot_hide_behind_admissible_mean(value):
    s=DistributedCladdingSession();y=s.state.copy()
    if value=='saturation':value=saturation_temperature_k(s.plant.network.volumes['core'].pressure_pa*1e-6)+.1
    y[s.cladding_index]=value;y=s._reconstruct_state(y)
    with pytest.raises(ValueError):s._admit(y)
    with pytest.raises(ValueError):s._rhs(0,y)


def test_alias_mismatch_is_rejected():
    s=DistributedCladdingSession();y=s.state.copy();y[s.base_size+1]+=.01
    with pytest.raises(ValueError,match='alias'):s._admit(y)


def test_json_restart_exact_and_fuel_only_restore_refuses_cladding_payload():
    s=DistributedCladdingSession(core=CoreConfig(kind='diffusion',cells=2,axial_bias_reactivity=1e-4))
    s.advance(.02);saved=json.loads(json.dumps(s.checkpoint()));t=DistributedCladdingSession.restore(saved)
    assert s.advance(.02,{'scram':True})==t.advance(.02,{'scram':True})
    assert s.checkpoint()==t.checkpoint()
    with pytest.raises(ValueError):DistributedFuelSession.restore(saved)
    with pytest.raises(ValueError):DistributedCladdingSession.restore(DistributedFuelSession().checkpoint())


@pytest.mark.parametrize('kind',['source','format','extra','nested_source','nested_format','nested_extra','config','grid','history','snapshot'])
def test_resealed_corruption_is_rejected(kind):
    s=DistributedCladdingSession();p=unseal(s.checkpoint())
    if kind=='source':p['source']['cladding_sha256']='0'*64
    elif kind=='format':p['format']='other'
    elif kind=='extra':p['bad']=True
    else:
        inner=unseal(p['fuel_checkpoint'])
        if kind=='nested_source':inner['distributed_fuel_source']['adapter_sha256']='0'*64
        elif kind=='nested_format':inner['format']='other'
        elif kind=='nested_extra':inner['extra']=0
        elif kind=='config':inner['config'].pop('clad_capacity_j_k')
        elif kind=='grid':inner['fuel_grid']['extra']=0
        elif kind=='history':inner['history']=[{'duration_s':.01,'bad':0}]
        else:inner['snapshot']['distributed_cladding']['temperature_k'][0]+=1
        p['fuel_checkpoint']=seal(inner)
    with pytest.raises(ValueError):DistributedCladdingSession.restore(seal(p))


def test_late_failure_rolls_back_local_states_controls_and_events(monkeypatch):
    s=DistributedCladdingSession();saved=s.checkpoint();original=DistributedCladdingSession._admit
    def reject(self,y):
        original(self,y)
        if self.time>0:raise ValueError('deliberate local wall rejection')
    monkeypatch.setattr(DistributedCladdingSession,'_admit',reject)
    with pytest.raises(ValueError,match='deliberate'):s.advance(.01,{'scram':True})
    assert s.checkpoint()==saved


def test_registry_records_local_storage_and_closure_as_illustrative():
    from atomic_model_station.foundation import validate_registry
    s=DistributedCladdingSession();r=s.parameter_registry()
    assert validate_registry(r)
    entries=[v for k,v in r['parameters'].items() if k.startswith('distributed_cladding.')]
    assert len(entries)==2*s.fuel_cells
    assert all(v['family']=='illustrative' and v['uncertainty'] is None for v in entries)
