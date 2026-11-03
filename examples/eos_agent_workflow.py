"""One EOS scenario through public Python, agent tools, files and exact resume."""

import argparse
import json
import tempfile
from pathlib import Path

from atomic_agent_tools import AgentToolServer
from atomic_model_runtime import create_eos_episode
from atomic_model_runtime.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    root = (
        a.output if a.output is not None else Path(tempfile.mkdtemp(prefix="atomic-eos-workflow-"))
    )
    root.mkdir(parents=True, exist_ok=True)
    cfg = dict(preset="mixed_ports", episode_horizon_s=0.03, action_dt_s=0.01, max_step_s=0.0005)
    direct = create_eos_episode(cfg)
    powers = direct._model.snapshot()["active_heat_power_w"].copy()
    heated = powers.copy()
    heated["wall.w0"] = 500.0
    spec = dict(
        schema_version=1,
        model_id=direct.model_id,
        config=cfg,
        timeline=[dict(decision=2, heat_power_w=heated)],
    )
    (root / "input.json").write_text(json.dumps(spec, indent=2) + "\n")
    server = AgentToolServer()
    view = server.invoke("create_eos_episode", dict(session_id="eos", config=cfg))
    for index in range(3):
        commands = powers if index == 0 else heated
        direct.step([commands[k] for k in direct.config.action_channels])
        view = server.invoke(
            "step_episode",
            dict(
                session_id="eos",
                expected_revision=view["revision"],
                observation_id=view["observation_id"],
                action=commands,
            ),
        )
        assert [x["value"] for x in view["observation"]] == direct.read_observation().tolist()
    assert server.export_session("eos")["payload"]["episode"] == direct.export_episode()
    run_experiment(spec, root / "full")
    run_experiment(spec, root / "paused", max_decisions=1)
    checkpoint = json.loads((root / "paused/checkpoint.json").read_text())
    run_experiment(spec, root / "resumed", checkpoint=checkpoint)
    full = json.loads((root / "full/checkpoint.json").read_text())
    resumed = json.loads((root / "resumed/checkpoint.json").read_text())
    assert full == resumed and full["payload"]["episode"] == direct.export_episode()
    print(
        json.dumps(
            dict(
                model_id=direct.model_id,
                python_agent_runner_exact=True,
                pause_resume_exact=True,
                completed_time_s=view["time_s"],
                output=str(root),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
