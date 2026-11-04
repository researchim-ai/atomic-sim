"""Two local JSONL agent policies, native equivalence and explicit-state resume."""

import argparse
import sys
from pathlib import Path

from atomic_model_agents import jobs
from atomic_model_runtime.episode import EOSEpisode
from atomic_model_workspace.common import atomic_json
from atomic_model_workspace.policies import decide, observation_frame


def fixtures():
    scenario = dict(
        schema_version=1,
        config=dict(preset="uniform", action_dt_s=0.01, episode_horizon_s=0.03, max_step_s=0.0005),
        seed=17,
        split="dev",
        objective=dict(sensor="wall.w0", target_k=300, tolerance_k=1),
    )
    hold = dict(kind="hold", heat_power_w={"wall.w0": 500.0})
    feedback = dict(
        kind="temperature-p",
        sensor="wall.w0",
        actuator="wall.w0",
        target_k=300,
        gain_w_k=500,
        bias_w=0,
        limit_w=1000,
    )
    return scenario, hold, feedback


def descriptor(initial_state):
    return dict(
        schema_version=1,
        agent_id="local-baseline",
        agent_version="v1",
        observations=["wall.w0"],
        actions=["wall.w0"],
        initial_state=initial_state,
        decision_timeout_s=2,
        rss_limit_bytes=128 * 1024**2,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "atomic_model_agents.baseline"]
    scenario, hold, feedback = fixtures()
    for name, state in [("hold", hold), ("feedback", feedback)]:
        desc = descriptor(state)
        atomic_json(a.output / (name + "-descriptor.json"), desc)
        jobs.create(scenario, desc, a.output / name, command)
        final = jobs.run(a.output / name, command)["payload"]
        assert final["status"] == "completed" and final["agent_state"]["decisions"] == 3
        jobs.create(scenario, desc, a.output / (name + "-resumed"), command)
        paused = jobs.run(a.output / (name + "-resumed"), command, max_decisions=1)
        assert paused["payload"]["status"] == "paused"
        jobs.resume(
            a.output / (name + "-resumed"), expected_sha256=paused["sha256"], command=command
        )
        resumed = jobs.run(a.output / (name + "-resumed"), command)["payload"]
        assert (
            resumed["episode"] == final["episode"]
            and resumed["agent_state"] == final["agent_state"]
        )
        native = EOSEpisode(scenario["config"])
        if name == "hold":
            policy = dict(
                kind="hold-v1",
                heat_power_w={
                    n: (500.0 if n == "wall.w0" else 0.0) for n in native.config.action_channels
                },
            )
        else:
            policy = dict(
                kind="temperature-p-v1", **{k: v for k, v in state.items() if k != "kind"}
            )
        while not native._done:
            action = decide(policy, observation_frame(native), native.config.action_channels)
            native.step([action[n] for n in native.config.action_channels])
        assert native.export_episode() == final["episode"]
        jobs.export(a.output / name, a.output / (name + "-export"))
        assert jobs.replay(a.output / (name + "-export"))["replay_exact"]
    atomic_json(a.output / "scenario.json", scenario)
    proof = dict(
        schema_version=1,
        agents=2,
        native_builtin_equivalent=True,
        explicit_state_resume_exact=True,
        agent_free_replay_exact=True,
        external_services=False,
        validated=False,
        plant_referenced=False,
    )
    atomic_json(a.output / "proof.json", proof)
    print(proof)


if __name__ == "__main__":
    main()
