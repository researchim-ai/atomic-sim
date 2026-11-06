"""Actual-core replay and interface boundary checks for agent episodes."""

from copy import deepcopy

import numpy as np
import pytest

from atomic_model_training.episode import run_episode
from atomic_pwr_agent import Episode


def advance(episode, commands=None):
    return episode.step(commands or {}, observation_id=episode.observe()["observation_id"])


def test_checkpoint_continuation_and_campaign_share_the_actual_core():
    episode = Episode(max_steps=3, seed=17)
    advance(episode, {"throttle": 0.95})
    saved = episode.checkpoint()
    restored = Episode.restore(saved)
    assert advance(episode) == advance(restored)
    assert episode.checkpoint() == restored.checkpoint()
    case = episode.training_case(case_id="agent", family="load")
    replay = run_episode(case)
    assert replay["checkpoint"] == episode.checkpoint()["payload"]["core"]
    assert replay["state"] == episode.observe()["state"]


def test_stale_and_invalid_actions_do_not_change_episode():
    episode = Episode(max_steps=2)
    old = episode.observe()["observation_id"]
    advance(episode)
    saved = episode.checkpoint()
    with pytest.raises(ValueError, match="Stale"):
        episode.step({}, observation_id=old)
    for commands in ({"throttle": float("nan")}, {"unknown": 1}, {"scram": 1}):
        with pytest.raises(ValueError):
            advance(episode, commands)
        assert episode.checkpoint() == saved


def test_observation_units_ownership_and_budget():
    episode = Episode(max_steps=1)
    obs = episode.observe()
    assert len(obs["state"]) == len(obs["state_schema"])
    assert obs["state_schema"][10]["unit"] == "K"
    obs["state"][0] = -123
    assert episode.observe()["state"][0] != -123
    result = advance(episode)["observation"]
    assert result["truncated"] and not result["terminated"]
    with pytest.raises(ValueError, match="finished"):
        advance(episode)


def test_scram_terminates_and_tampering_is_rejected():
    episode = Episode()
    assert advance(episode, {"scram": True})["observation"]["terminated"]
    saved = deepcopy(episode.checkpoint())
    saved["payload"]["config"]["seed"] = 8
    with pytest.raises(ValueError):
        Episode.restore(saved)


@pytest.mark.parametrize(
    "config", [dict(duration_s=True), dict(max_steps=257), dict(seed=-1), dict(tier="fast")]
)
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        Episode(**config)


def test_gymnasium_matches_python_episode_and_obeys_lifecycle():
    from atomic_pwr_agent.gymnasium import ControlledPWREnv

    env = ControlledPWREnv(max_steps=1)
    with pytest.raises(ValueError, match="reset"):
        env.step(1)
    initial, info = env.reset(seed=17)
    assert env.observation_space.contains(initial)
    reference = Episode(max_steps=1, seed=17)
    expected = advance(reference, {"rod_command": 0.0})
    obs, reward, terminated, truncated, info = env.step(1)
    np.testing.assert_array_equal(obs, expected["observation"]["state"])
    assert np.isfinite(reward) and not terminated and truncated
    with pytest.raises(ValueError, match="finished"):
        env.step(1)
    env.reset(seed=17)
    with pytest.raises(ValueError, match="Action"):
        env.step(3)
    env.close()


def test_gymnasium_standard_contract():
    from gymnasium.utils.env_checker import check_env

    from atomic_pwr_agent.gymnasium import ControlledPWREnv

    check_env(ControlledPWREnv(max_steps=3), skip_render_check=True)


def test_rehashed_wrong_duration_checkpoint_is_rejected():
    from atomic_model_workspace.common import digest

    episode = Episode()
    advance(episode)
    saved = episode.checkpoint()
    saved["payload"]["config"]["duration_s"] = 0.2
    saved["sha256"] = digest(saved["payload"])
    with pytest.raises(ValueError, match="configuration mismatch"):
        Episode.restore(saved)
