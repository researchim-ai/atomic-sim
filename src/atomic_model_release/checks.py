"""Executable golden/model semantics, data-only migration and use-case receipts."""

import hashlib
import math
import shutil
import sys
from pathlib import Path

from atomic_model_agents import jobs
from atomic_model_arena import runner as arena
from atomic_model_runtime.episode import EOSEpisode, model_card
from atomic_model_study import store
from atomic_model_workspace.common import atomic_json, digest, read
from atomic_model_workspace.project import load_project

GOLDEN = Path(__file__).parent / "golden"
GOLDEN_MANIFEST_SHA256 = "7b7b045a0d9fff09496c775e9268d1775aacfcd3fa4bd65e29c8e3ab8f154e20"


def _near(actual, expected, tolerance):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("Golden field set changed")
        for key, value in expected.items():
            _near(actual[key], value, tolerance)
    elif (
        type(actual) not in (int, float)
        or not math.isfinite(actual)
        or abs(actual - expected) > tolerance
    ):
        raise ValueError("Golden numerical semantics changed")


def golden():
    manifest = read(GOLDEN / "manifest.json")
    if digest(manifest) != GOLDEN_MANIFEST_SHA256:
        raise ValueError("Golden manifest changed without explicit review")
    from atomic_model_workspace.common import local_file

    for name, sha in manifest["files"].items():
        if hashlib.sha256(local_file(GOLDEN, name).read_bytes()).hexdigest() != sha:
            raise ValueError("Golden resource integrity mismatch")
    project = load_project(GOLDEN / "project/project.json")
    scenario = project["scenarios"]["uniform-train"]
    from atomic_model_workspace.jobs import _row

    for name in ("oracle.json", "actuated_oracle.json"):
        oracle = read(GOLDEN / name)
        env = EOSEpisode(scenario["config"])
        action = oracle.get("heat_power_w", project["policies"]["hold"]["heat_power_w"])
        if set(env.action_units) != set(action) or any(
            unit != "W" for unit in env.action_units.values()
        ):
            raise ValueError("Golden action unit/port semantics changed")
        while not env._done:
            env.step([action[n] for n in env.config.action_channels])
        row = _row(env)
        if row["time_s"] != oracle["time_s"]:
            raise ValueError("Golden horizon changed")
        if any(
            c["unit"] != "K" or not c["valid"] or c["sample_time_s"] != row["time_s"]
            for c in row["observation"]["channels"]
        ):
            raise ValueError("Golden observation unit/freshness semantics changed")
        _near(
            {c["name"]: c["value"] for c in row["observation"]["channels"]},
            oracle["final_observation"],
            oracle["temperature_abs_tol_k"],
        )
        _near(row["ledger"], oracle["global_ledger"], oracle["ledger_abs_tol"])
        _near(row["circuits"], oracle["local_ledgers"], oracle["ledger_abs_tol"])
    return dict(
        schema_version=1,
        golden_passed=True,
        golden_cases=2,
        origin_delivery_sha256=read(GOLDEN / "oracle.json")["origin_delivery_sha256"],
        project_sha256=digest(project),
        model=model_card(),
        validated=False,
        plant_referenced=False,
    )


def migrate_project(project_path, output):
    """Explicit workspace.v1 -> study.v1 data migration; no checkpoint rewriting."""
    project = load_project(project_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    try:
        store.create(project_path, output / "study")
        result = store.export(output / "study", output / "snapshot")
        store.import_snapshot(result["study"], output / "reimported")
        current = store.inspect(output / "reimported")
        if current["project"] != project:
            raise ValueError("Migration changed model/scenario/policy meaning")
        receipt = dict(
            schema_version=1,
            migration="workspace-v1-to-study-v1",
            project_sha256=digest(project),
            study_snapshot="snapshot/study.json",
            source_checkpoints_migrated=False,
            semantic_project_equal=True,
            validated=False,
            plant_referenced=False,
        )
        atomic_json(output / "migration.json", receipt)
    except BaseException:
        shutil.rmtree(output)
        raise
    return receipt


def control_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        project = load_project(GOLDEN / "project/project.json")
        scenario = project["scenarios"]["uniform-train"]
        descriptor = dict(
            schema_version=1,
            agent_id="release-hold",
            agent_version="v1",
            observations=["wall.w0"],
            actions=["wall.w0"],
            initial_state=dict(kind="hold", heat_power_w={"wall.w0": 0}),
            decision_timeout_s=2,
            rss_limit_bytes=128 * 1024**2,
        )
        command = [sys.executable, "-m", "atomic_model_agents.baseline"]
        jobs.create(scenario, descriptor, output / "job", command)
        value = jobs.run(output / "job", command)
        if value["payload"]["status"] != "completed":
            raise ValueError("Control runbook did not complete")
        jobs.export(output / "job", output / "export")
        replay = jobs.replay(output / "export")
        migration = migrate_project(GOLDEN / "project/project.json", output / "migration")
        report = dict(
            schema_version=1,
            control_completed=True,
            agent_free_replay=replay["replay_exact"],
            migration_semantics_equal=migration["semantic_project_equal"],
            accepted_decisions=len(value["payload"]["rows"]) - 1,
            golden=golden(),
            model=model_card(),
        )
        atomic_json(output / "runbook.json", report)
    except BaseException:
        shutil.rmtree(output)
        raise
    return report


def diagnose(folder):
    value = jobs.load(folder)
    p = value["payload"]
    reason = p["reason"]
    category = (
        "numerical_refusal"
        if reason.startswith("numerical:")
        else "agent_io_or_contract"
        if reason.startswith("agent_")
        else "lifecycle"
    )
    return dict(
        schema_version=1,
        job_id=p["job_id"],
        status=p["status"],
        observed_reason=reason,
        category=category,
        accepted_decisions=len(p["rows"]) - 1,
        accepted_time_s=p["rows"][-1]["model"]["time_s"],
        checkpoint_sha256=value["sha256"],
        last_attempt=p["attempts"][-1] if p["attempts"] else None,
        diagnosis_scope="Observed lifecycle/contract evidence, not expert causal root-cause annotation",
        agent_executed=False,
        validated=False,
        plant_referenced=False,
    )


def dataset_summary(folder):
    report = arena.verify(folder)
    rows = []
    for line in (Path(folder) / "transitions.jsonl").read_bytes().splitlines():
        from atomic_model_workspace.common import loads

        rows.append(loads(line))
    roles = {role: sum(r["split"] == role for r in rows) for role in ("train", "dev", "test")}
    return dict(
        schema_version=1,
        benchmark_id=report["benchmark_id"],
        protocol_sha256=report["protocol_sha256"],
        transitions=len(rows),
        splits=roles,
        terminated=sum(r["terminated"] for r in rows),
        truncated=sum(r["truncated"] for r in rows),
        reward_defined=any(r["reward"] is not None for r in rows),
        label_origin="synthetic local policy response; not expert or causal annotation",
        agent_executed=False,
        validated=False,
        plant_referenced=False,
    )


def doctor(folder=None):
    from atomic_model_release.environment import doctor as inspect_environment

    return inspect_environment(folder)


def diagnostic_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        project = load_project(GOLDEN / "project/project.json")
        scenario = project["scenarios"]["uniform-train"]
        descriptor = dict(
            schema_version=1,
            agent_id="intentional-failure",
            agent_version="v1",
            observations=["wall.w0"],
            actions=["wall.w0"],
            initial_state=dict(kind="intentional-invalid-demo"),
            decision_timeout_s=2,
            rss_limit_bytes=128 * 1024**2,
        )
        command = [sys.executable, "-m", "atomic_model_agents.baseline"]
        before = jobs.create(scenario, descriptor, output / "job", command)["payload"]
        after = jobs.run(output / "job", command)["payload"]
        if (
            after["status"] != "failed"
            or after["reason"] != "agent_exit"
            or after["episode"] != before["episode"]
            or after["agent_state"] != before["agent_state"]
        ):
            raise ValueError("Diagnostic negative control did not preserve simulator")
        result = dict(
            schema_version=1,
            demo_passed=True,
            expected_failure=True,
            simulator_unchanged=True,
            diagnosis=diagnose(output / "job"),
            expert_cause_claimed=False,
        )
        jobs.export(output / "job", output / "export")
        atomic_json(output / "diagnostic.json", result)
    except BaseException:
        shutil.rmtree(output)
        raise
    return result


def data_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        pack = read(GOLDEN / "benchmark.json")
        commands = {
            id: [sys.executable, "-m", "atomic_model_agents.baseline"] for id in pack["agents"]
        }
        arena.run(pack, commands, output / "benchmark")
        result = dataset_summary(output / "benchmark")
        result["agent_executed"] = True
        result["audit_agent_executed"] = False
        atomic_json(output / "dataset.json", result)
    except BaseException:
        shutil.rmtree(output)
        raise
    return result
