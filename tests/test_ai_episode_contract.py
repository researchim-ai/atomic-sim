"""End-to-end environment semantics and adversarial input checks."""

import json
from copy import deepcopy

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from agent_envs import EnvConfig, PWRPlantEnv
from core.status import DomainError, ModelLimitError, SolverError
from sdk.api import create_session, export_snapshot, step_session


def env(**kwargs):
    e = PWRPlantEnv(config=EnvConfig(action_dt_s=0.01, episode_horizon_s=0.025, **kwargs))
    e.reset(seed=123)
    return e


@pytest.mark.parametrize(
    "action", [[], [0.0, 0.0], [[0.0]], [True], ["0"], [float("nan")], [float("inf")], [1.01]]
)
def test_invalid_actions_are_atomic(action):
    e = env()
    before = e.export_episode()
    with pytest.raises(ValueError):
        e.step(action)
    assert e.export_episode() == before


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(action_dt_s=0),
        dict(sensor_tau_s=-1),
        dict(sensor_noise=-1),
        dict(action_channels=("bad",)),
        dict(episode_horizon_s=True),
        dict(rtol=float("nan")),
        dict(engine="rbmk"),
    ],
)
def test_bad_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        EnvConfig(**kwargs)


def test_cached_observation_reset_and_rng_are_pure():
    e = env(sensor_noise=0.01)
    initial = e.export_episode()
    for _ in range(3):
        e.read_observation()
        e._observe()
        e.observation_metadata()
    assert e.export_episode() == initial
    assert e.plant.time_s == 0
    assert e.power_sensor.value == pytest.approx(1.0, abs=1e-15)
    e.step([0.0])
    after = e.export_episode()
    assert after["rng"] != initial["rng"]
    e.read_observation()[0] = 0
    assert e.export_episode() == after


def test_exact_physical_horizon_and_step_lifecycle():
    e = env()
    assert e.step([0.0])[3] is False
    assert e.step([0.0])[3] is False
    _, _, terminated, truncated, info = e.step([0.0])
    assert e.plant.time_s == 0.025
    assert truncated and not terminated and info["reason"] == "time_limit"
    with pytest.raises(RuntimeError):
        e.step([0.0])


def test_timeline_is_split_exactly_and_sdk_path_agrees():
    c = EnvConfig(
        action_dt_s=0.02,
        episode_horizon_s=0.02,
        timeline=({"t": 0.01, "commands": {"throttle": 0.99}},),
    )
    e = PWRPlantEnv(config=c)
    e.reset(seed=7)
    e.step([0.0])
    p = create_session()
    step_session(p, 0.01, {"rod_command": 0.0})
    step_session(p, 0.01, {"throttle": 0.99})
    assert export_snapshot(e.plant) == export_snapshot(p)
    assert e.export_episode()["event_log"] == list(c.timeline)


@pytest.mark.parametrize("fault", ["open", "stuck"])
def test_faults_have_valid_finite_observations_and_explicit_quality(fault):
    e = env(sensor_faults=(fault, "", "", ""))
    e.step([0.0])
    assert e.observation_space.contains(e.read_observation())
    meta = e.observation_metadata()
    assert meta["valid"] == [False, True, True, True]
    assert meta["sample_time_s"][0] == 0.0
    assert meta["sample_time_s"][1] == 0.01


def test_episode_snapshot_is_exact_and_continues_noise_and_action_rng():
    e = env(sensor_noise=0.01)
    e.step([0.001])
    payload = json.loads(json.dumps(e.export_episode(), allow_nan=False))
    r = PWRPlantEnv.from_episode(payload)
    assert r.export_episode() == payload
    assert r.action_space.sample() == pytest.approx(e.action_space.sample())
    a = e.step([0.0])
    b = r.step([0.0])
    assert a[0] == pytest.approx(b[0], abs=1e-6)
    assert a[1] == pytest.approx(b[1], abs=1e-12)
    assert a[2:4] == b[2:4]
    assert a[4]["reward_terms"] == pytest.approx(b[4]["reward_terms"], abs=1e-12)
    for key in ("reason", "measurement", "episode_complete"):
        assert a[4][key] == b[4][key]
    assert np.array(e.export_episode()["plant"]["state"]) == pytest.approx(
        r.export_episode()["plant"]["state"], rel=1e-9, abs=1e-5
    )


@pytest.mark.parametrize(
    "exc,reason",
    [
        (SolverError("test"), "solver_error"),
        (DomainError("test"), "domain_error"),
        (ModelLimitError("test", criterion="x", node="y", observed=2, limit=1), "model_limit"),
    ],
)
def test_failed_decision_rolls_back_sensors_events_commands_rng_and_plant(exc, reason):
    e = env(sensor_noise=0.01)
    before = e.export_episode()
    advance = e.plant.advance

    def fail(dt, **kw):
        advance(dt, **kw)
        raise exc

    e.plant.advance = fail
    obs, reward, terminated, truncated, info = e.step([0.1])
    after = e.export_episode()
    assert after["plant"] == before["plant"]
    for key in (
        "sensors",
        "rng",
        "observation",
        "event_index",
        "event_log",
        "action_log",
        "metrics",
    ):
        assert after[key] == before[key]
    assert truncated and not terminated and reward < 0
    assert info["reason"] == reason


def test_programming_error_is_raised_with_atomic_rollback():
    e = env()
    before = e.export_episode()

    def fail(*a, **kw):
        e.plant.rod_command = 0.5
        raise RuntimeError("program")

    e.plant.advance = fail
    with pytest.raises(RuntimeError, match="program"):
        e.step([0.1])
    assert e.export_episode() == before


def test_task_terminal_and_reset_options():
    e = env(task="stop_on_scram")
    e.reset(seed=1, options={"initial_commands": {"scram": True}, "target_power_fraction": 0.0})
    _, _, term, trunc, info = e.step([0.0])
    assert term and not trunc and info["reason"] == "task_terminal"
    before = e.export_episode()
    with pytest.raises(ValueError):
        e.reset(options={"unknown": 1})
    assert e.export_episode() == before


def test_gymnasium_checker_and_independent_worker_seeds():
    e = PWRPlantEnv(config=EnvConfig(action_dt_s=0.01, episode_horizon_s=0.1))
    check_env(e, skip_render_check=True)
    a = env(sensor_noise=0.01)
    b = env(sensor_noise=0.01)
    a.action_space.sample()
    assert a.step([0.0])[0] == pytest.approx(b.step([0.0])[0])


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("steps", -1),
        ("event_index", 200),
        ("valid", [True]),
        ("observation", [0, 0, 0, 0]),
        ("sample_times", [float("nan")] * 4),
    ],
)
def test_corrupt_snapshot_is_rejected(field, value):
    payload = deepcopy(env().export_episode())
    payload[field] = value
    with pytest.raises((ValueError, DomainError)):
        PWRPlantEnv.from_episode(payload)


def test_multisegment_failure_rolls_back_whole_decision():
    e = PWRPlantEnv(
        config=EnvConfig(
            action_dt_s=0.02,
            episode_horizon_s=0.1,
            sensor_noise=0.01,
            timeline=({"t": 0.01, "commands": {"throttle": 0.99}},),
        )
    )
    e.reset(seed=1)
    before = e.export_episode()
    advance = e.plant.advance

    def fail(dt, **kw):
        if e.plant.time_s > 0:
            raise SolverError("second segment")
        return advance(dt, **kw)

    e.plant.advance = fail
    e.step([0.1])
    after = e.export_episode()
    for key in ("plant", "sensors", "rng", "observation", "event_index", "event_log", "metrics"):
        assert before[key] == after[key]
    assert PWRPlantEnv.from_episode(after).export_episode() == after


def test_completion_ranks_before_reward_and_truncation_failure():
    completed = env(target_power_fraction=0)
    for _ in range(3):
        completed.step([0.0])
    failed = env()

    def fail(*args, **kwargs):
        raise SolverError("test")

    failed.plant.advance = fail
    failed.step([0.0])
    assert completed.evaluator_metrics()["ranking_key"] > failed.evaluator_metrics()["ranking_key"]
    assert failed.evaluator_metrics()["eligible_for_ranking"] is False
    assert completed.evaluator_metrics()["eligible_for_ranking"] is True


@pytest.mark.parametrize(
    "modify",
    [
        lambda p: p["sensors"][0].update(tau_s=-1),
        lambda p: p["metrics"].update(action_cost=-1),
        lambda p: p.update(reason="time_limit", done=True),
        lambda p: p["config"].update(rtol=1e-3),
        lambda p: p.update(event_log=[{}]),
        lambda p: p.update(sample_times=[True] * 4),
    ],
)
def test_snapshot_rejects_inconsistent_history_and_physical_metadata(modify):
    p = env().export_episode()
    modify(p)
    with pytest.raises(ValueError):
        PWRPlantEnv.from_episode(p)


def test_real_subprocess_restart_and_no_overwrite(tmp_path):
    import subprocess
    import sys

    from agent_envs.replay import main

    e = env(sensor_noise=0.01)
    e.step([0.001])
    checkpoint = tmp_path / "episode.json"
    checkpoint.write_text(json.dumps(e.export_episode()))
    output = tmp_path / "resumed.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_envs.replay",
            str(checkpoint),
            "--action-json",
            "[0.0]",
            "--output",
            str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text())
    expected = e.step([0.0])
    assert result["observation"] == pytest.approx(expected[0], abs=1e-6)
    assert result["reward"] == pytest.approx(expected[1], abs=1e-12)
    assert result["checkpoint"]["rng"] == e.export_episode()["rng"]
    assert PWRPlantEnv.from_episode(result["checkpoint"]).export_episode() == result["checkpoint"]
    with pytest.raises(SystemExit):
        main([str(checkpoint), "--action-json", "[0.0]", "--output", str(output)])


def test_multiple_sdk_action_channels_and_default_seed_roundtrip():
    e = PWRPlantEnv(
        config=EnvConfig(seed=4, action_channels=("rod_command", "throttle"), action_dt_s=0.01)
    )
    e.reset()
    e.step([0.0, 0.99])
    assert e.plant.throttle == 0.99
    assert e.export_episode()["episode_seed"] == 4
    assert PWRPlantEnv.from_episode(e.export_episode()).export_episode() == e.export_episode()


@pytest.mark.parametrize("bound", ["low", "high"])
def test_declared_float32_action_endpoints_are_legal_sdk_commands(bound):
    from agent_envs.config import ACTION_BOUNDS

    e = PWRPlantEnv(config=EnvConfig(action_channels=tuple(ACTION_BOUNDS)))
    e.reset(seed=1)
    commands = e._action(getattr(e.action_space, bound))
    assert set(commands) == set(ACTION_BOUNDS)
