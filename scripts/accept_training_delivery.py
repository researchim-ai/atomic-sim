"""Admit exactly ten scoped tasks only after real-core and installed checks."""

import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from atomic_model_hem_identification.workflow import identity as prior_identity
from atomic_model_training.episode import identity
from atomic_model_training.runner import envelope, unpack
from atomic_model_training.spec import atomic_json, read_json
from atomic_model_workspace.common import digest
from sdk.experiment import _source_fingerprint

TASKS = [
    "AI-09.01",
    "AI-09.03",
    "AI-09.07",
    "AI-10.01",
    "AI-10.03",
    "AI-10.04",
    "AI-10.06",
    "AI-10.07",
    "AI-10.08",
    "AI-12.07",
]
SCOPE = (
    "Linux bounded declarative short-episode training on the unchanged controlled reduced PWR; "
    "no arbitrary Python policy, surrogate, long-transient speedup, experimental or plant-reference claim"
)


def main():
    root = Path.cwd()
    protocol = read_json("validation/ai091012_training_protocol_v1.json")
    admission = read_json("validation/ai091012_training_admission_v1.json")
    benchmark = unpack("artifacts/ai091012_training_benchmark_v2.json")
    installed = unpack("/tmp/atomic-training-cold-v1/installed.json")
    witness = read_json("validation/ai091012_training_build_witness_v1.json")
    source = identity()
    if (
        protocol["tasks"] != TASKS
        or benchmark["source"] != source
        or installed["source"] != source
        or witness["source"] != source
        or benchmark["admission_sha256"] != digest(admission)
        or admission["source_sha256"] != digest(source)
        or benchmark["protocol_sha256"] != digest(protocol)
        or not benchmark["accepted"]
        or not all(benchmark["exact_checkpoint_parity"])
        or benchmark["median_speedup"] < 2.0
        or not all(row["accepted"] for row in benchmark["tiers"])
        or not installed["campaign_completed"]
        or not installed["completed_resume_equal"]
        or not installed["core_control_replay"]["accepted"]
        or not witness["all_wheel_sdist_bytes_equal"]
    ):
        raise ValueError("Source-bound runtime qualification incomplete")
    for item in witness["files"] + [witness["wheel"], witness["sdist"]]:
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Build/runtime witness changed")
    logs = []
    for name, src, expected in [
        ("regression", "/tmp/training-regression-final.log", 202),
        ("final_runtime", "/tmp/training-tests-final-v3.log", 33),
        ("installed_runtime", "/tmp/training-installed-tests.log", 33),
    ]:
        raw = Path(src).read_bytes()
        text = raw.decode()
        matches = re.findall(r"\b(\d+) passed\b", text)
        if (
            not matches
            or int(matches[-1]) != expected
            or re.search(r"\d+ (failed|error|skipped)", text)
        ):
            raise ValueError("Missing or failed executed tests: " + name)
        target = Path("artifacts") / ("ai091012_" + name + ".log")
        shutil.copyfile(src, target)
        logs.append(
            dict(
                name=name, path=str(target), passed=expected, sha256=hashlib.sha256(raw).hexdigest()
            )
        )
    foundation = _source_fingerprint()
    prior = (
        read_json("artifacts/ai09_identification_assessment_v1.json")["payload"]["source"]
        == prior_identity()
    )
    if (
        foundation != "71fe3f4ce5c9f3e4dc86a36c17c1cdb32470ca19e0214ebc9c91af1c1a997d67"
        or not prior
    ):
        raise ValueError("Frozen scientific foundation/evidence changed")
    campaign = unpack("artifacts/ai091012_training_campaign_v2/report.json")
    if not campaign["completed"] or any(
        r["receipt"]["source_sha256"] != digest(source) for r in campaign["results"]
    ):
        raise ValueError("Current CLI campaign incomplete")
    checks = dict(
        source=source,
        tests=logs,
        installed_root=installed["installed_root"],
        foundation_sha256=foundation,
        prior_chain_preserved=prior,
        source_files={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *Path("atomic_model_training").glob("*.py"),
                Path("tests/test_training_campaign.py"),
                Path("scripts/qualify_training_runtime.py"),
                Path("scripts/verify_training_install.py"),
            ]
        },
        note="202-test regression preceded final output-unit schema; all 33 runtime tests rerun on final sources and installed wheel",
        all_checks_passed=True,
    )
    atomic_json("validation/ai091012_training_checks_v1.json", envelope(checks))
    shutil.copyfile(
        "/tmp/atomic-training-cold-v1/installed.json",
        "validation/ai091012_training_installed_v1.json",
    )
    basis = [
        "docs/AI_09_10_12_TRAINING_RUNTIME_RU.md",
        "validation/ai091012_training_protocol_v1.json",
        "validation/ai091012_training_admission_v1.json",
        "validation/ai091012_training_checks_v1.json",
        "validation/ai091012_training_installed_v1.json",
        "validation/ai091012_training_build_witness_v1.json",
        "artifacts/ai091012_training_benchmark_v2.json",
    ]
    delivery = dict(
        schema_version=1,
        accepted_tasks=TASKS,
        scope=SCOPE,
        source=source,
        median_speedup=benchmark["median_speedup"],
        exact_checkpoint_parity=benchmark["exact_checkpoint_parity"],
        evidence=basis,
        all_checks_passed=True,
        validated=False,
        plant_referenced=False,
    )
    atomic_json("validation/ai091012_training_delivery_v1.json", envelope(delivery))
    plan_path = Path("validation/ai_industrial_workplan_v1.json")
    plan = read_json(plan_path)
    if plan["version"] != "1.76.0":
        raise ValueError("Unexpected plan version; refusing automatic plan rewrite")
    changed = []
    for package in plan["packages"]:
        for task in package["tasks"]:
            if task["id"] in TASKS:
                if task["status"] == "accepted":
                    raise ValueError("Task already accepted")
                task.update(status="accepted", training_runtime_scope=SCOPE)
                task["evidence"] = list(
                    dict.fromkeys(
                        task.get("evidence", [])
                        + basis
                        + ["validation/ai091012_training_delivery_v1.json"]
                    )
                )
                plan["acceptance_decisions"].append(
                    dict(
                        date="2026-11-06",
                        task=task["id"],
                        decision="accepted",
                        scope=SCOPE,
                        basis=task["evidence"][-8:],
                        validated=False,
                        plant_referenced=False,
                    )
                )
                changed.append(task["id"])
        if package["id"] == "AI-10":
            if not all(t["status"] == "accepted" for t in package["tasks"]):
                raise ValueError("AI-10 still has unaccepted tasks")
            package.update(status="accepted", training_runtime_scope=SCOPE)
            package["evidence"] = basis + ["validation/ai091012_training_delivery_v1.json"]
    if set(changed) != set(TASKS):
        raise ValueError("Expected exactly ten task transitions")
    plan.update(
        version="1.77.0",
        date="2026-11-06",
        current_training_evidence="validation/ai091012_training_delivery_v1.json",
    )
    plan["execution_history"].append(
        dict(
            date="2026-11-06",
            version="1.77.0",
            packages=["AI-09", "AI-10", "AI-12"],
            accepted_tasks=TASKS,
            scope=SCOPE,
            evidence=basis,
            note="Ten software/numerical tasks accepted; >=2x goal met only on predeclared short reset-heavy actual-core workload",
        )
    )
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    counts = Counter(t["status"] for p in plan["packages"] for t in p["tasks"])
    print(
        json.dumps(dict(changed=changed, counts=counts, source_root=str(root)), ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
