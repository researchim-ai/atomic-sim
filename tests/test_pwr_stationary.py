from copy import deepcopy
import json

import numpy as np
import pytest
from core.status import SolverError
from atomic_model_multiphysics.provenance import seal,unseal
from coupling.pwr_channel import ChannelPWRSession
from coupling.pwr_stationary import StationaryChannelPWRSession,StationaryConfig,solve


@pytest.fixture(scope='module')
def equilibrium():return StationaryChannelPWRSession()


def test_full_rank_and_all_physical_equations_are_stationary(equilibrium):
    s=deepcopy(equilibrium);report=s.stationary_report;rate=s._rhs(0.,s.state)
    assert report['constrained_rank']==report['constrained_unknowns']==46
    ratio=abs(rate[report['physical_indices']])/report['physical_tolerances']
    assert max(ratio)<=1 and report['max_physical_tolerance_ratio']<=1
    assert not report['parameter_changes'] and not report['pressure_control_settings_changed']
    assert s.plant._pressurizer_heater_w==0
    for name in ('surge','spray','dump'):
        i=s.junction_offset+s.plant.network.junction_names().index(name)
        assert abs(rate[i])<1e-6


def test_stationary_constraints_preserve_mass_and_boron_without_an_entropy_claim(equilibrium):
    s=equilibrium;initial=unseal(s.initial_channel)['snapshot']['coupled_state'];old=np.asarray(initial)
    names=s.plant.network.volume_names();idx={n:13+3*i for i,n in enumerate(names)}
    primary=('core','hot_leg','sg_primary','cold_leg','pressurizer')
    assert sum(s.state[idx[n]] for n in primary)==pytest.approx(sum(old[idx[n]] for n in primary),abs=1e-5)
    assert sum(s.state[idx[n]+2] for n in primary)==pytest.approx(sum(old[idx[n]+2] for n in primary),abs=1e-10)
    new_secondary=s.state[idx['sg_secondary']]+s.state[s.base_size-3]+s.state[s.feed_index]
    old_secondary=old[idx['sg_secondary']]+old[s.base_size-3]+old[s.feed_index]
    assert new_secondary==pytest.approx(old_secondary,abs=1e-4)
    report=s.stationary_report
    assert not report['entropy_preserved'] and not report['transient_preparation_simulated']
    assert report['original_pressurizer_entropy_j_k']!=report['final_pressurizer_entropy_j_k']
    assert report['initial_energy_change_j']!=0


def test_warm_profile_and_equilibrium_reservoirs_are_owned_locally(equilibrium):
    s=deepcopy(equilibrium);r=s.snapshot()
    assert np.ptp(r['distributed_fluid']['temperature_k'])>10
    assert r['channel_initialization']['stationary_channel_solve']
    assert r['stationary_initialization']['equilibrium_heat_history_assumed']
    assert max(abs(np.ravel(r['distributed_fuel']['local_heat_storage_rate_w'])))<1e-6
    assert r['distributed_fuel']['decay_exchange_residual_j']==0.
    assert r['actuator_status']['rod_position']==s.controls['rod_target']
    assert not r['validated'] and not r['plant_referenced']


def test_hold_has_no_artificial_cold_front_or_control_motion(equilibrium):
    s=deepcopy(equilibrium);initial=s.state.copy();r=s.advance(.05)
    active=s.stationary_report['physical_indices']
    error=max(abs(s.state[active]-initial[active])/np.maximum(abs(initial[active]),1.))
    assert error<1e-7
    assert r['energy_relative']<1e-9 and r['mass_relative']<1e-10 and r['boron_relative']<1e-10
    assert s.state[12]==initial[12]


def test_json_restart_of_stationary_start_and_scram_suffix_is_exact(equilibrium):
    s=deepcopy(equilibrium);s.advance(.01)
    saved=json.loads(json.dumps(s.checkpoint()));t=StationaryChannelPWRSession.restore(saved)
    assert s.advance(.01,{'scram':True})==t.advance(.01,{'scram':True})
    assert s.checkpoint()==t.checkpoint()
    with pytest.raises(ValueError):ChannelPWRSession.restore(saved)


@pytest.mark.parametrize('patch',[{'relative_power':True},{'relative_power':.8},{'relative_power':float('nan')},
    {'max_nfev':0},{'max_nfev':True},{'max_rhs_calls':0},{'max_rhs_calls':True}])
def test_invalid_stationary_controls_are_refused(patch):
    with pytest.raises(ValueError):StationaryConfig(**patch)


def test_failed_solver_budget_does_not_change_seed():
    s=ChannelPWRSession();before=s.checkpoint()
    with pytest.raises(SolverError,match='budget'):solve(s,StationaryConfig(max_rhs_calls=64))
    assert s.checkpoint()==before


def test_cannot_reinitialize_a_running_trajectory(equilibrium):
    s=deepcopy(equilibrium);s.advance(.001);before=s.checkpoint()
    with pytest.raises(ValueError,match='Unadvanced'):solve(s)
    assert s.checkpoint()==before


def test_omitted_physical_equation_leak_cannot_pass_optimizer(equilibrium,monkeypatch):
    original=ChannelPWRSession._rhs
    def leaking(self,t,y):
        rate=original(self,t,y);rate[self.feed_index+3]+=100.
        return rate
    monkeypatch.setattr(ChannelPWRSession,'_rhs',leaking)
    with pytest.raises(SolverError,match='Full stationary'):solve(equilibrium)


@pytest.mark.parametrize('kind',['source','format','extra','config','history','initial'])
def test_resealed_checkpoint_metadata_corruption_refused(equilibrium,kind):
    p=unseal(equilibrium.checkpoint())
    if kind=='source':p['source']['initializer_sha256']='0'*64
    elif kind=='format':p['format']='other'
    elif kind=='extra':p['extra']=True
    elif kind=='config':p['stationary']['extra']=True
    elif kind=='history':p['history']='not-history'
    else:p['initial_channel']={}
    with pytest.raises(ValueError):StationaryChannelPWRSession.restore(seal(p))
