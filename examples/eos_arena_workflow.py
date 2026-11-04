"""Small open train/dev/test benchmark; synthetic repeats are not independent trials."""

import argparse
import copy
import sys
from pathlib import Path

from atomic_model_arena import protocol, runner
from atomic_model_workspace.common import atomic_json, read
from examples.eos_external_agent_workflow import descriptor, fixtures


def example_protocol():
    scenario, hold, feedback = fixtures()
    agents = {"hold": descriptor(dict(hold, heat_power_w={"wall.w0": 0.0}))}
    for gain in [0, 500, 1000]:
        agents["p-" + str(gain)] = descriptor(dict(feedback, gain_w_k=gain))
    cases = {}
    for role, preset in [("train", "uniform"), ("dev", "mixed_ports"), ("test", "gravity")]:
        value = copy.deepcopy(scenario)
        value["split"] = role
        value["config"]["preset"] = preset
        cases[role + "-case"] = value
    return dict(
        schema_version=1,
        benchmark_id="eos-open-agent-baselines-v1",
        model_id="eos-thermal-assembly-v1",
        cases=cases,
        agents=agents,
        baseline_id="hold",
        repeats=2,
        budgets=dict(
            job_wall_s=10, total_wall_s=300, worker_rss_bytes=512 * 1024**2, max_decisions=8
        ),
        constraints=dict(
            energy_residual_j=1e-6,
            entropy_residual_j_k=1e-6,
            mass_residual_kg=1e-8,
            momentum_residual_kg_m_s=1e-8,
            circuit_energy_residual_j=1e-6,
            absolute_power_w=10000,
        ),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=False)
    spec = example_protocol()
    atomic_json(a.output / "input.json", spec)
    commands = {id: [sys.executable, "-m", "atomic_model_agents.baseline"] for id in spec["agents"]}
    receipt = runner.run(spec, commands, a.output / "benchmark")
    report = runner.verify(a.output / "benchmark")
    assert receipt["status"] == "completed" and len(report["episodes"]) == 16, receipt
    assert all(row["metrics"]["eligible"] for row in report["episodes"])
    assert all(
        a["native_repeat_exact"]
        for phase in report["phases"].values()
        for a in phase["agents"].values()
    )
    selection = read(a.output / "benchmark/selection.json")
    assert all(
        task["phase"] == "train"
        for task in report["episodes"]
        if task["task_id"] in selection["training_task_ids"]
    )
    result = dict(
        schema_version=1,
        episodes=16,
        selected_agent=selection["selected_agent"],
        train_only_selection=True,
        native_repeats_exact=True,
        transitions=sum(1 for _ in (a.output / "benchmark/transitions.jsonl").open()),
        hardware_sha256=report["hardware_sha256"],
        source=protocol.source_identity(),
        token_usage=None,
        token_cost=None,
        validated=False,
        plant_referenced=False,
    )
    atomic_json(a.output / "proof.json", result)
    print(result)


if __name__ == "__main__":
    main()
