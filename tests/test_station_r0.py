from copy import deepcopy
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_station import StationConfig,StationResidual,StationSession,Port,validate_connections


def test_steady_initialization_and_global_balance():
    s=StationSession();before=s.snapshot();r=s.advance(30.)
    np.testing.assert_allclose(r['state'],before['state'],rtol=1e-12)
    assert abs(r['step_energy_residual_j'])<.001 and r['initialization_residual']<1e-8
    assert r['electrical_export_j']>0 and r['cooling_rejection_j']>0


def test_load_disconnect_matches_independent_ode_solver():
    s=StationSession();initial=s.snapshot()['state'];m=s.model;a,b=m.operators(1e6,False)
    reference=solve_ivp(lambda t,y:a@y+b,(0,2.),initial,method='DOP853',rtol=1e-11,atol=1e-8)
    actual=s.advance(2.,generator_connected=False)
    np.testing.assert_allclose(actual['state'],reference.y[:,-1],rtol=1e-10)
    assert actual['electrical_export_j']==0.


def test_restart_exact_and_detached():
    s=StationSession();s.advance(2.,source_power_w=1.05e6);saved=s.checkpoint();r=StationSession.restore(saved)
    saved['payload']['snapshot']['state'][0]=0.
    assert r.advance(1.)==s.advance(1.)


@pytest.mark.parametrize('damage',['duration','power','bool','domain'])
def test_rejected_transaction_keeps_state_and_history(damage):
    s=StationSession(StationConfig(rotor_inertia_kg_m2=100.)) if damage=='domain' else StationSession()
    before=s.checkpoint()
    with pytest.raises(ValueError):
        if damage=='duration':s.advance(float('nan'))
        elif damage=='power':s.advance(1.,source_power_w=-1.)
        elif damage=='bool':s.advance(1.,generator_connected=1)
        else:s.advance(60.,generator_connected=False)
    assert s.checkpoint()==before


def test_resealed_restart_corruption_refused():
    s=StationSession();p=unseal(s.checkpoint());p['snapshot']['state'][0]+=1.
    with pytest.raises(ValueError):StationSession.restore(seal(p))


@pytest.mark.parametrize('value',[True,float('inf'),-1.])
def test_config_refuses_bad_values(value):
    with pytest.raises(ValueError):StationConfig(engine_efficiency=value)


def test_parameter_registry_units_and_scope():
    c=StationConfig();r=c.registry()
    assert r['rotor_inertia_kg_m2']['unit']=='kg*m^2'
    assert all(v['family']=='illustrative' for v in r.values())
    assert 'Fluid mass and momentum' in StationSession().snapshot()['passport']['unsupported']


def ports():return [Port('a','out','thermal','K','W'),Port('b','in','thermal','K','W')]


def test_ports_require_owned_unique_compatible_complete_connections():
    p=ports();assert validate_connections(p,[(('a','out'),('b','in'))])
    for declaration,connections in [(p,[]),(p+p,[]),(p,[(('a','out'),('a','out'))]),(p,[(('missing','out'),('b','in'))])]:
        with pytest.raises(ValueError):validate_connections(declaration,connections)


def test_port_family_and_units_refused():
    with pytest.raises(ValueError):Port('a','x','thermal','degC','W')
    p=[ports()[0],Port('b','in','mechanical','rad/s','N*m')]
    with pytest.raises(ValueError):validate_connections(p,[(('a','out'),('b','in'))])


def test_split_steps_preserve_same_physical_solution():
    a=StationSession();b=StationSession()
    a.advance(20.,source_power_w=1.05e6)
    b.advance(10.,source_power_w=1.05e6);b.advance(10.)
    np.testing.assert_allclose(a.snapshot()['state'],b.snapshot()['state'],rtol=1e-12)
    assert abs(a.snapshot()['electrical_export_j']-b.snapshot()['electrical_export_j'])<1e-6


@pytest.mark.parametrize('connections',[[(1,2)],[(('a',),('b','in'))],iter([])])
def test_malformed_or_unbounded_connection_input_refused(connections):
    with pytest.raises(ValueError):validate_connections(ports(),connections)


def test_checkpoint_survives_json_roundtrip():
    import json
    s=StationSession();s.advance(3.,source_power_w=1.05e6)
    saved=json.loads(json.dumps(s.checkpoint()))
    restored=StationSession.restore(saved)
    assert restored.advance(2.)==s.advance(2.)
