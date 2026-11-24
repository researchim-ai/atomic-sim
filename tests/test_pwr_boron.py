from copy import deepcopy
from dataclasses import replace
import json
import math

import numpy as np
import pytest
from atomic_model_multiphysics.provenance import seal,unseal
from coupling.pwr_boron import BoronExchangeConfig,BoronExchangePWRSession,tank_state
from coupling.refinement import coupled_pwr_mesh


@pytest.mark.parametrize('change',[dict(supply_mass_kg=True),dict(supply_boron_ppm=-1.),dict(receiver_boron_ppm=10001.),
    dict(pump_efficiency=0.),dict(maximum_flow_kg_s=float('inf')),dict(gas_volume_m3=0.),dict(temperature_k=400.,pressure_pa=1e5),
    dict(minimum_mass_kg=1000.,supply_mass_kg=100.)])
def test_invalid_exchange_parameters_rejected(change):
    with pytest.raises(ValueError):replace(BoronExchangeConfig(),**change)


def tank(m=5000.,b=2.5):
    return tank_state([m,m*1e6,b],initial_mass=5000.,gas_volume=5.,pressure=2e5,rho=1000.,u_ref=1e6,temperature=350.)


def test_adiabatic_cushion_and_reference_liquid_match_hand_values():
    t=tank(4500.)
    assert t['gas_volume_m3']==5.5 and t['temperature_k']==350.
    assert t['pressure_pa']==pytest.approx(2e5*(5/5.5)**1.4)
    assert t['gas_energy_j']==pytest.approx(t['pressure_pa']*5.5/.4)
    step=.01
    derivative=(tank(4500.+step)['gas_energy_j']-tank(4500.-step)['gas_energy_j'])/(2*step)
    assert derivative==pytest.approx(t['pressure_pa']/1000.,rel=1e-8)


@pytest.mark.parametrize('mass,boron',[(0.,0.),(10000.,1.),(5000.,-1.),(5000.,51.)])
def test_tank_dry_gas_exhaustion_or_bad_boron_rejected(mass,boron):
    with pytest.raises(ValueError):tank(mass,boron)


@pytest.fixture(scope='module')
def seed():
    return BoronExchangePWRSession(**coupled_pwr_mesh(2),max_step_s=.05,rtol=2e-9)


def test_inactive_component_preserves_base_primary_state_and_declares_capability(seed):
    r=seed.snapshot();base=unseal(seed.initial_moderator)['snapshot']
    np.testing.assert_array_equal(r['coupled_state'][:seed.boron_index],base['coupled_state'])
    assert r['powers']['nuclear_fission_w']==base['powers']['nuclear_fission_w']
    assert r['actuator_status']['boron']['available'] and not r['actuator_status']['boron']['direct_inventory_assignment_allowed']
    assert r['finite_boron_exchange']['electrical_power_w']==0.
    assert all(v['family']=='illustrative' and v['uncertainty'] is None for k,v in seed.parameter_registry()['parameters'].items() if k.startswith('boron_exchange.'))


def test_transfer_rhs_has_finite_tank_owners_and_closed_mass_boron_energy(seed):
    s=deepcopy(seed);s.state[s.boron_index+6]=10.;rate=s._rhs(0.,s.state);r=s.snapshot();t=r['finite_boron_exchange']
    assert rate[s.boron_index]==-10. and rate[s.boron_index+3]==10.
    assert rate[s.boron_index+2]==0. and rate[s.boron_index+5]>0.
    assert t['electrical_power_w']>t['hydraulic_power_w']>0.
    assert abs(t['instantaneous_energy_residual_w'])<1e-6
    assert abs(t['instantaneous_boron_residual_kg_s'])<1e-15
    assert s._passive_column(s.boron_index+7) and not s._passive_column(s.boron_index)


def test_channel_face_diagnostics_exclude_appended_exchange_states(seed):
    s=deepcopy(seed);s.state[s.boron_index+6]=10.
    row=s.snapshot();fluid=row['distributed_fluid']
    expected=np.r_[s.state[s.pump_index],s.state[s.internal_flow_index:s.boron_index],s.state[s.out_index]]
    np.testing.assert_array_equal(fluid['face_flow_kg_s'],expected)
    assert len(fluid['face_flow_kg_s'])==len(fluid['face_acceleration_kg_s2'])==s.fluid_cells+1
    from scripts.qualify_fluid_geometry import audit_geometry,coefficients
    assert audit_geometry(row,coefficients(s))['passed']


def test_public_flow_command_respects_actuator_lag_and_inventory_ledgers(seed):
    s=deepcopy(seed);before=s.snapshot();r=s.advance(.1,{'boron_flow_target_kg_s':50.})
    t=r['finite_boron_exchange']
    assert t['flow_kg_s']==pytest.approx(50.*(1-np.exp(-.1/.2)))
    assert t['tanks']['supply']['mass_kg']<before['finite_boron_exchange']['tanks']['supply']['mass_kg']
    assert t['tanks']['receiver']['boron_kg']>before['finite_boron_exchange']['tanks']['receiver']['boron_kg']
    assert r['mass_relative']<1e-10 and r['boron_relative']<1e-10 and abs(r['energy_residual_j'])<.1
    assert any(m['name']=='boron_flow_actuator' for m in r['exact_mode_reduction'])


@pytest.mark.parametrize('patch',[{},True,{'boron_flow_target_kg_s':True},{'boron_flow_target_kg_s':51.},{'boron_ppm':500.}])
def test_bad_commands_do_not_change_state_or_history(seed,patch):
    s=deepcopy(seed);before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(.1,patch)
    assert s.checkpoint()==before


def test_reserve_failure_rolls_back_whole_step(seed):
    s=BoronExchangePWRSession(**coupled_pwr_mesh(2),boron=BoronExchangeConfig(supply_mass_kg=1000.,minimum_mass_kg=999.),max_step_s=.05,rtol=2e-9)
    before=s.checkpoint()
    with pytest.raises(ValueError,match='reserve'):s.advance(1.,{'boron_flow_target_kg_s':50.})
    assert s.checkpoint()==before


def test_json_restart_reproduces_exchange_and_rejects_config_tampering(seed):
    s=deepcopy(seed);s.advance(.05,{'boron_flow_target_kg_s':50.});saved=json.loads(json.dumps(s.checkpoint()))
    restored=BoronExchangePWRSession.restore(saved)
    assert restored.advance(.05)==s.advance(.05)
    p=unseal(saved);p['config']['supply_boron_ppm']=2000.
    with pytest.raises(ValueError):BoronExchangePWRSession.restore(seal(p))


def test_exact_actuator_mass_integral_includes_closing_tail():
    from scripts.qualify_boron_exchange import transferred_mass
    at_close=50*(2.-.2*(1-math.exp(-10.)))
    tail=50*(1-math.exp(-10.))*.2*(1-math.exp(-2.5))
    assert transferred_mass(0.)==0.
    assert transferred_mass(2.5)==pytest.approx(at_close+tail)


@pytest.mark.parametrize('time',[-1.,True,float('inf')])
def test_invalid_mass_integral_time_rejected(time):
    from scripts.qualify_boron_exchange import transferred_mass
    with pytest.raises(ValueError):transferred_mass(time)


def test_incomplete_chemical_comparison_cannot_pass():
    from scripts.qualify_boron_exchange import assess
    with pytest.raises(ValueError):assess({'dilution_fine':{}})


def test_unsupported_loss_of_pump_power_rolls_back_without_inventing_energy(seed):
    s=deepcopy(seed);s.advance(.05,{'boron_flow_target_kg_s':50.});before=s.checkpoint()
    with pytest.raises(ValueError,match='external grid'):s.advance(.1,{'grid_available':False})
    assert s.checkpoint()==before


def test_restore_reuses_verified_base_instead_of_solving_stationarity_twice(seed,monkeypatch):
    from coupling.pwr_moderator import LocalModeratorPWRSession
    s=deepcopy(seed);s.advance(.05,{'boron_flow_target_kg_s':50.});saved=s.checkpoint()
    calls=[];original=LocalModeratorPWRSession.__init__
    def counted(self,*args,**kwargs):
        calls.append(True);return original(self,*args,**kwargs)
    monkeypatch.setattr(LocalModeratorPWRSession,'__init__',counted)
    restored=BoronExchangePWRSession.restore(saved)
    assert len(calls)==1
    assert restored.advance(.05)==s.advance(.05)


def test_regenerator_moves_heat_internally_without_adding_pump_work_or_boron(seed):
    plain=deepcopy(seed);plain.state[plain.boron_index+6]=10.
    active=deepcopy(plain);active.boron_config=replace(active.boron_config,regenerator_ua_w_k=200000.)
    before=plain._rhs(0.,plain.state);after=active._rhs(0.,active.state)
    r=active.snapshot();x=r['finite_boron_exchange'];heat=x['regenerator']['heat_w']
    assert heat>0 and x['regenerator']['capacity_entropy_generation_w_k']>0
    cold_index=13+3*active.plant.network.volume_names().index('cold_leg')+1
    expected=before.copy();expected[cold_index]+=heat;expected[active.boron_index+4]-=heat
    np.testing.assert_allclose(after,expected,rtol=1e-14,atol=1e-6)
    assert x['electrical_power_w']==plain.boron_terms['electrical_power_w']
    assert abs(x['instantaneous_energy_residual_w'])<1e-6
    assert abs(x['instantaneous_boron_residual_kg_s'])<1e-15


@pytest.mark.parametrize('ua',[-1.,True,float('nan')])
def test_invalid_regenerator_conductance_rejected(ua):
    with pytest.raises(ValueError):replace(BoronExchangeConfig(),regenerator_ua_w_k=ua)


@pytest.mark.parametrize('mass',[True,-1.,float('nan')])
def test_bad_protocol_mass_reference_cannot_be_used_as_evidence(seed,mass):
    from scripts.qualify_boron_exchange import audit
    meta=dict(cold_index=13+3*seed.plant.network.volume_names().index('cold_leg'),rho=seed.boron_rho)
    with pytest.raises(ValueError,match='reference transferred mass'):audit(seed.snapshot(),meta,expected_transferred_mass=mass)
