"""Physical split separation, held-out selection, native replay and honest cost metadata."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

from atomic_model_arena import metrics, runner
from atomic_model_workspace.common import atomic_json, read
from examples.eos_arena_workflow import example_protocol


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    folder = tmp_path_factory.mktemp("arena") / "benchmark"
    spec = example_protocol()
    runner.run(
        spec,
        {id: [sys.executable, "-m", "atomic_model_agents.baseline"] for id in spec["agents"]},
        folder,
    )
    return folder


@pytest.fixture
def retained(benchmark, tmp_path):
    folder = tmp_path / "benchmark"
    shutil.copytree(benchmark, folder)
    return folder


def rehash(folder, name):
    manifest = read(folder / "manifest.json")
    manifest["artifacts"][name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    atomic_json(folder / "manifest.json", manifest)


def test_complete_pipeline_split_selection_repeats_and_transitions(benchmark):
    report = runner.verify(benchmark)
    assert len(report["episodes"]) == 16
    assert report["token_usage"] is None and report["token_cost"] is None
    selection = read(benchmark / "selection.json")
    assert selection["selected_agent"] == "p-1000"
    assert all(
        row["phase"] == "train"
        for row in report["episodes"]
        if row["task_id"] in selection["training_task_ids"]
    )
    for name, phase in report["phases"].items():
        assert all(a["native_repeat_exact"] and a["eligible"] for a in phase["agents"].values())
        if name != "train":
            assert set(phase["agents"]) == {"hold", "p-1000"}
    rows = [json.loads(line) for line in (benchmark / "transitions.jsonl").read_text().splitlines()]
    assert len(rows) == 48 and {r["split"] for r in rows} == {"train", "dev", "test"}
    assert all(r["reward"] is None for r in rows)
    assert not any(r["terminated"] for r in rows)
    assert sum(r["truncated"] for r in rows) == 16
    assert sum(r["episode_completed"] for r in rows) == 16
    assert {r["completion_reason"] for r in rows if r["truncated"]} == {"time_limit"}
    assert all({c["name"] for c in row["observation"]["channels"]} == {"wall.w0"} for row in rows)
    assert all(set(row["action_w"]) == {"wall.w0"} for row in rows)
    machine = read(benchmark / "hardware.json")
    assert len(machine["cpu_affinity"]) == 1 and all(
        v == "1" for v in machine["execution_environment"]["threads"].values()
    )


@pytest.mark.parametrize(
    "defect",
    [
        "same_preset_seed",
        "same_preset_horizon",
        "same_preset_objective",
        "permission",
        "timeout",
        "rss",
        "missing_role",
        "repeats",
        "schema",
        "constraint",
    ],
)
def test_protocol_refusals_are_pre_execution(defect, tmp_path):
    spec = example_protocol()
    if defect.startswith("same_preset"):
        spec["cases"]["dev-case"]["config"]["preset"] = "uniform"
        if defect.endswith("seed"):
            spec["cases"]["dev-case"]["seed"] += 1
        elif defect.endswith("horizon"):
            spec["cases"]["dev-case"]["config"]["episode_horizon_s"] = 0.04
        else:
            spec["cases"]["dev-case"]["objective"]["target_k"] += 1
    elif defect == "permission":
        spec["agents"]["hold"]["observations"].append("primary.channel.c0")
    elif defect == "timeout":
        spec["agents"]["hold"]["decision_timeout_s"] = 1
    elif defect == "rss":
        spec["agents"]["hold"]["rss_limit_bytes"] *= 2
    elif defect == "missing_role":
        spec["cases"]["dev-case"]["split"] = "train"
    elif defect == "repeats":
        spec["repeats"] = True
    elif defect == "schema":
        spec["schema_version"] = True
    else:
        spec["constraints"]["energy_residual_j"] = float("inf")
    with pytest.raises(ValueError):
        runner.run(
            spec,
            {id: [sys.executable, "-m", "atomic_model_agents.baseline"] for id in spec["agents"]},
            tmp_path / "bad",
        )
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "defect",
    [
        "selection",
        "tasks",
        "report",
        "transitions",
        "hardware_threads",
        "hardware_cpu",
        "extra",
        "symlink",
        "source",
    ],
)
def test_tampered_artifacts_fail_even_after_outer_hash_update(retained, defect):
    if defect == "selection":
        name = "selection.json"
        value = read(retained / name)
        value["selected_agent"] = "p-0"
        atomic_json(retained / name, value)
        rehash(retained, name)
    elif defect == "tasks":
        name = "tasks.json"
        value = read(retained / name)
        value.reverse()
        atomic_json(retained / name, value)
        rehash(retained, name)
    elif defect == "report":
        name = "report.json"
        value = read(retained / name)
        value["token_usage"] = 123
        atomic_json(retained / name, value)
        rehash(retained, name)
    elif defect == "transitions":
        name = "transitions.jsonl"
        raw = (retained / name).read_text().replace('"reward":null', '"reward":1', 1)
        (retained / name).write_text(raw)
        rehash(retained, name)
    elif defect.startswith("hardware"):
        name = "hardware.json"
        value = read(retained / name)
        if defect == "hardware_threads":
            value["execution_environment"]["threads"] = {}
        else:
            value["cpu_affinity"] = [True]
        atomic_json(retained / name, value)
        rehash(retained, name)
    elif defect == "extra":
        (retained / "unlisted.py").write_text('raise AssertionError("never execute")')
    elif defect == "source":
        value = read(retained / "manifest.json")
        value["source"]["arena_sha256"] = "0" * 64
        atomic_json(retained / "manifest.json", value)
    else:
        (retained / "selection.json").rename(retained / "linked.json")
        (retained / "selection.json").symlink_to(retained / "linked.json")
    with pytest.raises((ValueError, OSError)):
        runner.verify(retained)


def test_train_selection_ignores_changed_dev_test_scores(benchmark):
    records = read(benchmark / "tasks.json")
    pack = read(benchmark / "protocol.json")
    original = metrics.select(pack, records)
    changed = copy.deepcopy(records)
    for row in changed:
        if row["phase"] != "train":
            row["metrics"]["quality"]["mae"] = 1e99
            row["metrics"]["eligible"] = False
    assert metrics.select(pack, changed) == original


def test_no_survivorship_selection_or_fake_independent_repeats(benchmark):
    pack = read(benchmark / "protocol.json")
    records = read(benchmark / "tasks.json")
    for row in records:
        if row["phase"] == "train" and row["agent_id"] != "hold":
            row["metrics"]["eligible"] = False
    result = metrics.select(pack, records)
    assert result["selected_agent"] is None
    assert all(not score["eligible"] for name, score in result["scores"].items() if name != "hold")
    report = read(benchmark / "report.json")
    rows = copy.deepcopy(report["episodes"])
    target = next(r for r in rows if r["phase"] == "dev" and r["agent_id"] == "hold")
    target["metrics"]["episode_sha256"] = "changed"
    rebuilt = metrics.report(
        pack, rows, read(benchmark / "selection.json"), read(benchmark / "hardware.json")
    )
    assert rebuilt["phases"]["dev"]["agents"]["hold"]["native_repeat_exact"] is False


def test_cli_verify_and_worker_receipt(benchmark):
    result = subprocess.run(
        [sys.executable, "-m", "atomic_model_arena", "verify", str(benchmark)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["episodes"]) == 16


def test_wrong_host_command_or_cpu_never_creates_output(tmp_path):
    spec = example_protocol()
    with pytest.raises(ValueError):
        runner.run(spec, {}, tmp_path / "bad")
    with pytest.raises(ValueError):
        runner.run(
            spec,
            {id: [sys.executable, "-m", "atomic_model_agents.baseline"] for id in spec["agents"]},
            tmp_path / "bad",
            cpu=max(os.sched_getaffinity(0)) + 1000,
        )
    assert not (tmp_path / "bad").exists()


def test_analytic_objective_integrates_zero_crossing_exactly():
    def row(t, value):
        return dict(
            model=dict(
                decision=int(t > 0),
                time_s=t,
                observation=dict(channels=[dict(name="s", value=value)]),
                heat_power_w={"p": 0} if t else None,
                ledger=dict(energy_residual_j=0, entropy_residual_j_k=0),
                circuits=dict(
                    primary=dict(
                        mass_residual_kg=0, momentum_residual_kg_m_s=0, energy_residual_j=0
                    )
                ),
            )
        )

    job = dict(
        rows=[row(0, 299), row(2, 301)],
        scenario=dict(objective=dict(sensor="s", target_k=300, tolerance_k=1)),
        status="completed",
        reason="test",
        episode=dict(fixture=True),
        attempts=[],
    )
    result = metrics.evaluate(job, example_protocol())
    assert result["quality"]["mae"] == 0.5
    assert result["quality"]["integral_absolute_error"] == 1
    assert result["quality"]["rmse"] == pytest.approx((1 / 3) ** 0.5)
    assert result["quality_metric_units"]["integral_absolute_error"] == "K s"
    assert result["quality_metric_units"]["model_peak_time_s"] == "s"
    assert result["quality_integral_unit"] == "K s"


def test_whole_worker_timeout_does_not_publish_a_valid_benchmark(tmp_path):
    from atomic_model_agents.transport import AgentFailure

    spec = example_protocol()
    spec["budgets"]["total_wall_s"] = 1
    target = tmp_path / "timeout"
    with pytest.raises(AgentFailure, match="agent_timeout"):
        runner.run(
            spec,
            {id: [sys.executable, "-m", "atomic_model_agents.baseline"] for id in spec["agents"]},
            target,
        )
    assert read(target / "controller_failure.json")["reason"] == "agent_timeout"
    with pytest.raises((ValueError, OSError)):
        runner.verify(target)
