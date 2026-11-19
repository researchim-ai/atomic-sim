from copy import deepcopy
from dataclasses import replace
import io
import json

import pytest

from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_station.faults import PWRFaultSession
from atomic_model_station_equilibrium import EquilibriumConfig,solve
from atomic_model_station_faults import FaultSession
from atomic_model_station_readiness import capture


@pytest.mark.parametrize('kw',[dict(power_w=True),dict(power_w=1.),dict(pressure_pa=8e6),
    dict(speed_rad_s=200.),dict(pressure_pa=float('nan'))])
def test_invalid_equilibrium_targets(kw):
    with pytest.raises(ValueError):EquilibriumConfig(**kw)


def test_whole_assembly_solves_without_equipment_or_caller_changes():
    owner=PWRFaultSession();before=owner.checkpoint();params=owner.parameter_registry()
    result,report=solve(owner)
    assert owner.checkpoint()==before and result.parameter_registry()==params
    assert report['constrained_rank']==report['constrained_unknowns']==49
    assert report['full_dimensional_ratio']<=1. and abs(report['external_net_w'])<50.
    assert not report['parameters_changed'] and not report['operating_state_projected']
    assert report['declared_initial_state_solved'] and report['warmup_steps']==0
    assert abs(report['initial_energy_change_j'])>1.
    snap=result.snapshot()
    assert snap['core']['fission_total_w']==pytest.approx(3.15e9,abs=.5)
    assert snap['steam_header']['pressure_pa']==pytest.approx(6.3e6,abs=1.)
    assert snap['mass_relative']<1e-10 and snap['boron_relative']<1e-10 and snap['energy_relative']<1e-10


@pytest.mark.parametrize('power',[3.12e9,3.18e9])
def test_neighboring_operating_points_respect_the_active_branch(power):
    if power==3.12e9:
        from core.status import DomainError
        owner=PWRFaultSession();before=owner.checkpoint()
        with pytest.raises(DomainError,match='zero-spray'):solve(owner,EquilibriumConfig(power_w=power))
        assert owner.checkpoint()==before
        return
    plant=PWRFaultSession(equilibrium=EquilibriumConfig(power_w=power))
    assert plant.station_equilibrium['report']['full_dimensional_ratio']<=1.
    assert plant.snapshot()['core']['fission_total_w']==pytest.approx(power,abs=.5)


def test_live_station_cannot_be_reinitialized_or_have_ledger_erased():
    owner=PWRFaultSession(fault_tick_s=.01);owner.advance(.01);before=owner.checkpoint()
    with pytest.raises(ValueError):solve(owner)
    assert owner.checkpoint()==before


def test_initialized_epoch_cannot_be_initialized_again():
    owner=PWRFaultSession(equilibrium=EquilibriumConfig());before=owner.checkpoint()
    with pytest.raises(ValueError):solve(owner)
    assert owner.checkpoint()==before


def test_diffusion_core_is_explicitly_unsupported_by_this_initializer():
    from atomic_model_station.core import CoreConfig
    owner=PWRFaultSession(core=CoreConfig(kind='diffusion',cells=2))
    with pytest.raises(ValueError):solve(owner)


def test_physical_checkpoint_reconstructs_declared_initialization_then_history():
    owner=PWRFaultSession(equilibrium=EquilibriumConfig(),fault_tick_s=.05)
    owner.advance(.05);cp=json.loads(json.dumps(owner.checkpoint(),sort_keys=True))
    restored=PWRFaultSession.restore(cp)
    assert owner.advance(.05)==restored.advance(.05)
    assert owner.checkpoint()==restored.checkpoint()


def test_all_six_pi_references_match_solved_state_and_native_restart():
    owner=FaultSession(equilibrium_start=True,inventory_deadband_m=.0002,tick_s=.05)
    assert capture(owner)['goals_met']
    result=owner.step(modes={k:'auto' for k in owner.control.loops})
    assert capture(owner)['goals_met'] and not result['receipt']['trip_latched']
    cp=owner.checkpoint();restored=FaultSession.restore(cp)
    assert owner.step()==restored.step() and owner.checkpoint()==restored.checkpoint()


def test_control_session_native_restore_uses_initialized_physical_factory():
    from atomic_model_station_control import ControlSession
    owner=FaultSession(equilibrium_start=True,tick_s=.05).control
    owner.step(modes={k:'auto' for k in owner.loops});saved=owner.checkpoint()
    restored=ControlSession.restore(saved)
    assert owner.step()==restored.step()


def test_changed_initialization_recipe_is_rejected():
    owner=PWRFaultSession(equilibrium=EquilibriumConfig());p=deepcopy(unseal(owner.checkpoint()))
    p['equilibrium_recipe']['source']={}
    with pytest.raises(ValueError):PWRFaultSession.restore(seal(p))


def test_hot_program_can_start_from_solved_state_without_relaxing_span():
    from atomic_model_station_hot import HotConfig,HotSession
    from atomic_model_station_readiness import ReadinessConfig
    cfg=HotConfig(maximum_warmup_s=.03,hold_s=.02,tick_s=.01,equilibrium_start=True,
        readiness=ReadinessConfig(window_s=.02,maximum_gap_s=.01))
    run=HotSession(cfg)
    while not run.finished:run.step()
    assert run.report()['hold_passed'] and not run.report()['canonical_hold_passed']
    assert run.report()['readiness']['maximum_normalized_span']<1e-3


def test_agent_configuration_selects_equilibrium_without_exposing_hidden_state():
    from atomic_model_station_env import StationEnv,EnvConfig
    from atomic_model_station_env.__main__ import serve
    env=StationEnv(EnvConfig(max_steps=2,tick_s=.05,equilibrium_start=True,inventory_deadband_m=.0002))
    output=io.StringIO();serve(io.StringIO('{"method":"reset"}\n{"method":"step"}\n'),output,env)
    rows=[json.loads(line) for line in output.getvalue().splitlines()]
    assert all(row['ok'] for row in rows)
    assert 'equilibrium_initialization' not in rows[0]['result']['observation']
    saved=env.checkpoint();restored=StationEnv.restore(saved)
    assert env.step()==restored.step()


def test_campaign_initialization_flag_survives_anchor_replay():
    from atomic_model_station_campaign import Episode
    from atomic_model_station_campaign.engine import run_episode
    episode=Episode('equilibrium',2,tick_s=.05,equilibrium_start=True,inventory_deadband_m=.0002,checkpoint_ticks=(1,))
    first=run_episode(episode);p=unseal(first)
    assert p['accepted'] and p['completed_ticks']==2
    second=run_episode(episode,p['checkpoints']['1'])
    assert first==second


@pytest.mark.parametrize('factory',[lambda:FaultSession(equilibrium_start=1),lambda:__import__('atomic_model_station_env',fromlist=['EnvConfig']).EnvConfig(equilibrium_start=1)])
def test_boolean_initialization_option_is_strict(factory):
    with pytest.raises(ValueError):factory()
