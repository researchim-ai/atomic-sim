"""Verify installed Python/Gym/JSONL controlled-PWR interfaces outside checkout."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from atomic_agent_tools import AgentToolServer
from atomic_agent_tools.__main__ import publish
from atomic_agent_tools.server import provenance
from atomic_pwr_agent import Episode
from atomic_pwr_agent.gymnasium import ControlledPWREnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    server = AgentToolServer()
    view = server.invoke(
        "create_controlled_episode", dict(session_id="pwr", config=dict(seed=17, max_steps=2))
    )
    view = server.invoke(
        "step_controlled_episode",
        dict(
            session_id="pwr",
            expected_revision=view["revision"],
            observation_id=view["observation_id"],
            action={"throttle": 0.95},
        ),
    )
    saved = server.export_session("pwr")
    restored = AgentToolServer().import_session("restored", saved)
    if restored["observation"] != view["observation"]:
        raise RuntimeError("Tool restart differs")
    native = Episode(seed=17, max_steps=2)
    native.step({"throttle": 0.95}, observation_id=native.observe()["observation_id"])
    if native.checkpoint() != saved["payload"]["episode"]["checkpoint"]:
        raise RuntimeError("Python and JSONL tool core differ")
    gym = ControlledPWREnv(max_steps=1)
    obs, _ = gym.reset(seed=17)
    if not gym.observation_space.contains(obs) or not gym.step(1)[3]:
        raise RuntimeError("Gymnasium lifecycle differs")
    gym.close()
    with tempfile.TemporaryDirectory(prefix="atomic-agent-cli-") as temporary:
        checkpoint = Path(temporary) / "checkpoint.json"
        publish(checkpoint, saved)
        request = dict(
            request_id="inspect", tool="inspect_episode", arguments=dict(session_id="pwr")
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "atomic_agent_tools",
                "--resume",
                str(checkpoint),
                "--session-id",
                "pwr",
            ],
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            timeout=60,
            check=True,
        )
        reply = json.loads(result.stdout)
        if not reply["ok"] or reply["result"]["observation"] != view["observation"]:
            raise RuntimeError("Fresh JSONL process replay differs")
    publish(
        args.output,
        dict(
            schema_version=1,
            provenance=provenance(),
            python_tool_equal=True,
            host_restart_equal=True,
            jsonl_restart_equal=True,
            gym_lifecycle_passed=True,
            validated=False,
            plant_referenced=False,
        ),
    )
    print("Installed PWR agent Python/Gymnasium/JSONL checks passed")


if __name__ == "__main__":
    main()
