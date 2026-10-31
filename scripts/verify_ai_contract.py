"""Run traceable episode, SDK-parity and fresh-process restart qualification."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from agent_envs import EnvConfig, PWRPlantEnv
from sdk.api import apply_commands, create_session, export_snapshot, step_session
from sdk.experiment import _source_fingerprint


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--budget", type=Path, default=Path("validation/ai01_contract_budget_v1.json")
    )
    args = parser.parse_args()
    args.output.mkdir()
    budget = json.loads(args.budget.read_text())
    cases = [
        EnvConfig(seed=7, action_dt_s=0.01, episode_horizon_s=0.05, sensor_noise=0.01),
        EnvConfig(
            seed=9,
            action_dt_s=0.02,
            episode_horizon_s=0.05,
            sensor_noise=0.01,
            timeline=({"t": 0.015, "commands": {"throttle": 0.99}},),
        ),
        EnvConfig(
            seed=11, action_dt_s=0.01, episode_horizon_s=0.03, sensor_faults=("open", "", "", "")
        ),
        EnvConfig(
            seed=13,
            action_dt_s=0.01,
            episode_horizon_s=0.04,
            task="stop_on_scram",
            timeline=({"t": 0.02, "commands": {"scram": True}},),
        ),
    ]
    rows = []
    for index, config in enumerate(cases):
        folder = args.output / str(index)
        folder.mkdir()
        write(folder / "input.json", config.as_dict())
        env = PWRPlantEnv(config=config)
        env.reset()
        twin = PWRPlantEnv(config=config)
        twin.reset()
        plant = create_session()
        event_index = 0
        transitions = []
        max_state_error = 0.0
        while not env._done:
            before = env.export_episode()
            env.read_observation()
            assert env.export_episode() == before
            result = env.step([0.0])
            repeated = twin.step([0.0])
            assert np.array_equal(result[0], repeated[0])
            assert result[1:] == repeated[1:]
            # Independent public SDK command schedule, same integration policy.
            apply_commands(plant, {"rod_command": 0.0})
            target = env.plant.time_s
            while plant.time_s < target:
                while (
                    event_index < len(config.timeline)
                    and config.timeline[event_index]["t"] <= plant.time_s
                ):
                    apply_commands(plant, config.timeline[event_index]["commands"])
                    event_index += 1
                end = target
                if event_index < len(config.timeline):
                    end = min(end, config.timeline[event_index]["t"])
                step_session(plant, end - plant.time_s)
            while (
                event_index < len(config.timeline)
                and config.timeline[event_index]["t"] <= plant.time_s
            ):
                apply_commands(plant, config.timeline[event_index]["commands"])
                event_index += 1
            state = np.array(export_snapshot(plant)["state"])
            actual = np.array(export_snapshot(env.plant)["state"])
            max_state_error = max(max_state_error, float(max(abs(actual - state))))
            assert np.allclose(
                actual, state, rtol=budget["state_relative"], atol=budget["state_absolute"]
            )
            transitions.append(
                dict(
                    observation=result[0].tolist(),
                    reward=result[1],
                    terminated=result[2],
                    truncated=result[3],
                    info=result[4],
                    time_s=env.plant.time_s,
                )
            )
        write(folder / "trajectory.json", transitions)
        write(folder / "checkpoint.json", env.export_episode())
        assert (
            PWRPlantEnv.from_episode(env.export_episode()).export_episode() == env.export_episode()
        )
        rows.append(
            dict(
                case=index,
                steps=len(transitions),
                reason=transitions[-1]["info"]["reason"],
                sdk_state_max_absolute=max_state_error,
                metrics=env.evaluator_metrics(),
                passed=True,
            )
        )
    # Fresh process continuation with a nonzero control history and noise.
    env = PWRPlantEnv(config=cases[0])
    env.reset()
    env.step([0.001])
    write(args.output / "restart_input.json", env.export_episode())
    subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_envs.replay",
            str(args.output / "restart_input.json"),
            "--action-json",
            "[0.0]",
            "--output",
            str(args.output / "restart_output.json"),
        ],
        check=True,
    )
    other = json.loads((args.output / "restart_output.json").read_text())
    result = env.step([0.0])
    expected = env.export_episode()
    observed_error = float(max(abs(np.array(other["observation"]) - result[0])))
    reward_error = abs(other["reward"] - result[1])
    assert (
        observed_error <= budget["observation_absolute"]
        and reward_error <= budget["reward_absolute"]
    )
    assert other["checkpoint"]["rng"] == expected["rng"]
    assert other["checkpoint"]["action_rng"] == expected["action_rng"]
    assert np.allclose(
        other["checkpoint"]["plant"]["state"],
        expected["plant"]["state"],
        rtol=budget["state_relative"],
        atol=budget["state_absolute"],
    )
    write(args.output / "budget.json", budget)
    report = dict(
        passed=True,
        scope="Short nominal/small-throttle/sensor-fault/SCRAM task contract; not full-horizon physical qualification",
        source_sha256=_source_fingerprint(),
        cases=rows,
        subprocess_restart=True,
        observation_error=observed_error,
        reward_error=reward_error,
        budget_sha256=hashlib.sha256(args.budget.read_bytes()).hexdigest(),
        validated=False,
        plant_referenced=False,
    )
    write(args.output / "report.json", report)
    hashes = {
        str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.rglob("*")
        if p.is_file()
    }
    write(args.output / "manifest.json", dict(sha256=hashes))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
