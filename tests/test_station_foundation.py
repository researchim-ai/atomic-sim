from copy import deepcopy
import json
import numpy as np
import pytest
from atomic_model_station.contracts import (FluidState,fluid_connection,thermal_connection,mechanical_connection,electrical_connection)
from atomic_model_station.mixed import Variable,ResidualComponent,MixedSystem
from atomic_model_station.mixed_session import MixedStationSession
from atomic_model_station.nonlinear import build,guess,continuation
from atomic_model_station.model import StationConfig
from atomic_model_multiphysics.provenance import seal,unseal


def test_fluid_upstream_reversal_and_zero_transport():
    a=FluidState(1e6,1e5,2.,(('boron',.001),));b=FluidState(1e6,3e5,-2.,(('boron',.002),))
    r=fluid_connection(a,b);assert r['enthalpy_a_to_b_w']==2e5 and r['species_a_to_b_kg_s']['boron']==.002
    a=FluidState(1e6,1e5,-2.,(('boron',.001),));b=FluidState(1e6,3e5,2.,(('boron',.002),))
    r=fluid_connection(a,b);assert r['upstream']=='b' and r['enthalpy_a_to_b_w']==-6e5
    assert fluid_connection(FluidState(1e6,1e5,0.),FluidState(1e6,3e5,0.))['enthalpy_a_to_b_w']==0.


def test_effort_flow_and_disconnected_contracts():
    assert thermal_connection(500.,10.,500.,-10.)['flow_residual_w']==0.
    assert mechanical_connection(100.,2.,100.,-2.)['power_residual_w']==0.
    assert electrical_connection(0.,False,0.,False)['power_residual_w']==0.
    with pytest.raises(ValueError):electrical_connection(1.,False,-1.,False)
    with pytest.raises(ValueError):fluid_connection(FluidState(1e6,1e5,1.),FluidState(1e6,1e5,1.))


@pytest.mark.parametrize('args',[(True,1e5,0.),(1e6,float('nan'),0.),(1e6,1e5,0.,(('x',1.1),)),(1e6,1e5,0.,(),'unknown')])
def test_invalid_fluid_contract_refused(args):
    with pytest.raises(ValueError):FluidState(*args)


def simple_system(duplicate=False,bad_unit=False):
    x=Variable('a','x','K',True,1.,-100.,100.);z=Variable('b','z','K',False,1.,-100.,100.)
    a=ResidualComponent('a',[x],{'b.z':'W' if bad_unit else 'K'},[1.],lambda t,y,d,u:[d['a.x']+y['a.x']-y['b.z']])
    b=ResidualComponent('b',[z],{},[1.],lambda t,y,d,u:[0. if duplicate else y['b.z']-2.])
    return MixedSystem([a,b])


def test_generic_consistency_keeps_differential_stock_and_detects_rank():
    s=simple_system();y,rate,r=s.consistent(0.,np.array([1.,0.]),{})
    np.testing.assert_array_equal(y,[1.,2.]);np.testing.assert_allclose(rate,[1.,0.]);assert r['rank']==2
    y,_,_=s.step(0.,y,.1,{});assert abs(y[0]-(1+.1*2)/1.1)<1e-9
    with pytest.raises(ValueError):simple_system(duplicate=True).consistent(0.,np.array([1.,0.]),{})
    with pytest.raises(ValueError):simple_system(bad_unit=True)


def test_nonlinear_load_continuation_and_wall_adapter():
    rows=continuation(StationConfig());assert len(rows)==3
    assert all(r['receipt']['residual']<1e-8 and r['receipt']['rank']==11 for r in rows)
    assert rows[0]['state'][4]>rows[-1]['state'][4]


def test_topology_reinitializes_algebraic_values_without_inventory_projection():
    s=MixedStationSession();old=s.y[:5].copy();s.set_command({'generator_connected':False})
    np.testing.assert_array_equal(s.y[:5],old);assert abs(s.y[9])<1e-6 and s.revision==1


def test_root_event_localized_and_reinitialized():
    s=MixedStationSession(max_step_s=.2);s.set_command({'source_power_w':1.1e6})
    threshold=float(s.y[0]+.01);s.guard('trip','stock.primary',threshold,1,{'generator_connected':False})
    s.advance(1.)
    root=[e for e in s.events if e['kind']=='root'];assert len(root)==1 and 0<root[0]['time_s']<1.
    assert s.guards[0]['latched'] and s.revision==1 and root[0]['localization_tolerance_s']<=1e-6


def test_guard_already_active_trips_immediately():
    s=MixedStationSession();s.guard('trip','stock.primary',float(s.y[0]-1.),1,{'generator_connected':False})
    assert s.guards[0]['latched'] and s.events[0]['time_s']==0.


def test_queue_rng_measurements_regulator_restart_new_history():
    s=MixedStationSession(seed=71);m=s.observe();s.control(606.,m['wall_temperature_k'],.1)
    s.schedule(1.,{'generator_connected':False});s.guard('late','stock.primary',850.,1,{'source_power_w':0.})
    s.advance(.4);saved=json.loads(json.dumps(s.checkpoint()));r=MixedStationSession.restore(saved)
    assert r.observe()==s.observe();assert r.advance(1.)==s.advance(1.)
    assert r.snapshot()['queue']==[] and r.revision==1


def test_invalid_step_rolls_back_rng_events_and_all_inventories():
    s=MixedStationSession();before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(float('nan'))
    assert before==s.checkpoint()
    with pytest.raises(ValueError):s.set_command({'load_fraction':0.})
    assert before==s.checkpoint()


def test_resealed_checkpoint_corruption_refused():
    s=MixedStationSession();p=unseal(s.checkpoint());p['snapshot']['state'][0]+=1.
    with pytest.raises(ValueError):MixedStationSession.restore(seal(p))


def test_generic_map_rejects_duplicate_owner_and_complex_guess():
    s=simple_system()
    with pytest.raises(ValueError):MixedSystem([s.components[0],s.components[0]])
    with pytest.raises(ValueError):s.steady(np.array([1+0j,2+0j]),{})


def test_registry_and_reuse_inventory_are_explicit_and_versioned():
    from atomic_model_station.foundation import parameter_registry,validate_registry,reuse_audit,passport,acceptance
    registry=parameter_registry();assert validate_registry(registry)
    bad=deepcopy(registry);bad['parameters']['engine_efficiency']['family']='validated'
    with pytest.raises(ValueError):validate_registry(bad)
    assert len(reuse_audit())>=10 and not passport()['validated']
    assert not acceptance()['levels']['R2']['accepted']


def test_duplicate_queued_events_and_failed_late_transaction_preserve_state():
    s=MixedStationSession();s.schedule(1.,{'generator_connected':False});before=s.checkpoint()
    with pytest.raises(ValueError):s.schedule(1.,{'generator_connected':True})
    assert s.checkpoint()==before
    # The queued change executes in the trial, then the artificial call budget fails.
    s.history=[dict(op='observe',args={})]*256;before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(2.)
    assert s.checkpoint()==before


def test_local_component_diagnostics_and_event_variable_ownership():
    s=MixedStationSession();r=s.system.diagnostics(s.time,s.y,s.rate,s.u)
    assert set(r)=={'stock','flux'}
    assert all(v['scaled_max']<1e-8 for v in r.values())
    assert all(name.startswith(owner+'.') for owner,v in r.items() for name in v['event_variables'])


def test_reuse_audit_test_links_exist_in_source_checkout():
    from pathlib import Path
    from atomic_model_station.foundation import reuse_audit
    root=Path(__file__).resolve().parents[1]
    assert all((root/r['test']).is_file() for r in reuse_audit())


def test_mixed_system_composes_with_existing_residual_framework():
    from core.residual import ResidualSystem
    model=simple_system();wrapped=ResidualSystem([model])
    assert wrapped.size==2
    wrapped.require_consistent(0.,np.array([1.,2.]),np.array([1.,0.]),{})
