"""Real local JSONL client; deterministic baseline, no LLM service required."""

import json
import subprocess
import sys


def main():
    with subprocess.Popen(
        [sys.executable, "-m", "atomic_agent_tools"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
    ) as process:
        counter = 0

        def call(tool, arguments):
            nonlocal counter
            counter += 1
            process.stdin.write(
                json.dumps(
                    dict(request_id=f"r{counter}", tool=tool, arguments=arguments), allow_nan=False
                )
                + "\n"
            )
            process.stdin.flush()
            response = json.loads(process.stdout.readline())
            if not response["ok"]:
                raise RuntimeError(response["error"])
            return response["result"]

        def guard(v):
            return dict(
                session_id=v["session_id"],
                expected_revision=v["revision"],
                observation_id=v["observation_id"],
            )

        description = call("describe_simulator", {})
        initial = call(
            "create_episode",
            dict(
                session_id="episode",
                config=dict(seed=42, sensor_noise=0.01, episode_horizon_s=0.03),
            ),
        )
        saved = call("save_checkpoint", guard(initial))
        first = call("step_episode", dict(**guard(initial), action={"rod_command": 0.001}))
        restored = call(
            "restore_checkpoint", dict(**guard(first), checkpoint_id=saved["checkpoint_id"])
        )
        replayed = call("step_episode", dict(**guard(restored), action={"rod_command": 0.001}))
        assert (
            first["observation"] == replayed["observation"]
            and first["last_transition"] == replayed["last_transition"]
        )
        cancelled = call("cancel_episode", guard(replayed))
        call("close_episode", guard(cancelled))
        call("forget_checkpoint", {"checkpoint_id": saved["checkpoint_id"]})
        process.stdin.close()
        process.wait(timeout=60)
        if process.returncode:
            raise RuntimeError("Worker failed")
        print(
            json.dumps(
                dict(
                    protocol_version=description["protocol_version"],
                    tools=[t["name"] for t in description["tools"]],
                    model=description["model"]["model_id"],
                    exact_checkpoint_replay=True,
                    first_observation=first["observation"],
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
