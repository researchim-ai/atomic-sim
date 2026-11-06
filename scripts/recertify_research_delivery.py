"""Refresh source-bound evidence after final guard fixes without promoting external gates."""

import json
import re
import shutil
from pathlib import Path

from atomic_model_multiphysics.provenance import identity as model_identity
from atomic_model_multiphysics.provenance import seal, sha, unseal
from atomic_model_research.release import identity as release_identity
from atomic_model_training.spec import atomic_json, read_json
from atomic_model_uncertainty.coverage import identity as uq_identity
from scripts.finalize_research_delivery import ACCEPTED, EXTERNAL, SCOPES
from sdk.experiment import _source_fingerprint


def main():
    q = unseal(read_json("artifacts/final17_multiphysics_qualification_v3.json"))
    r = unseal(read_json("artifacts/final17_release_manifest_v3.json"))
    u = unseal(read_json("artifacts/final17_uq_qualification_v2.json"))
    installed = unseal(read_json("/tmp/atomic-final17-cold-v2/installed-proof.json"))
    installed_uq = unseal(read_json("/tmp/atomic-final17-cold-v2/uq-proof.json"))
    legacy = read_json("artifacts/final17_legacy_installed_v2/proof.json")
    witness = read_json("validation/final17_build_witness_v2.json")
    if (
        q["source"] != model_identity()
        or not q["numerical_accepted"]
        or not q["openmc_pipeline_executed"]
        or r["source"] != release_identity()
        or installed["source"] != release_identity()
        or witness["source"] != release_identity()
        or witness["uncertainty_source"] != uq_identity()
        or u["source"] != uq_identity()
        or installed_uq != u
        or u["validated"]
        or installed["another_machine_proven"]
        or not installed["spatial_replay_equal"]
        or not installed["project_run"]["payload"]["report"]["completed"]
        or installed["resource_count"] != 760
        or not all(x["passed"] for x in legacy["phases"])
        or legacy["wheel_sha256"] != witness["wheel"]["sha256"]
    ):
        raise ValueError("Current-source qualification incomplete")
    if len({x["seed"] for x in q["transport"]}) != 3:
        raise ValueError("Transport seed inventory")
    for p in Path("artifacts/final17_openmc_v4").glob("replica-*/execution.json"):
        receipt = unseal(read_json(p))
        if receipt["source"] != model_identity() or receipt["status"] != "completed":
            raise ValueError("Final transport source differs")
    for item in witness["runtime_files"] + [witness["wheel"], witness["sdist"]]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Final build bytes differ")
    if _source_fingerprint() != "71fe3f4ce5c9f3e4dc86a36c17c1cdb32470ca19e0214ebc9c91af1c1a997d67":
        raise ValueError("Frozen foundation changed")
    checks = []
    for name, path in [
        ("current_source", "/tmp/final17-new-current-tests.log"),
        ("cold_installed", "/tmp/final17-final-installed-tests.log"),
    ]:
        text = Path(path).read_text()
        if not re.search(r"\b73 passed\b", text) or re.search(r"\d+ (failed|errors|skipped)", text):
            raise ValueError("Final executed tests incomplete: " + name)
        target = Path("artifacts") / ("final17_" + name + "_v2.log")
        shutil.copyfile(path, target)
        checks.append(dict(path=str(target), sha256=sha(target), passed=73))
    shutil.copyfile(
        "/tmp/atomic-final17-cold-v2/installed-proof.json", "validation/final17_installed_v2.json"
    )
    old = unseal(read_json("validation/final17_delivery_v1.json"))
    if old["accepted_tasks"] != ACCEPTED:
        raise ValueError("Unexpected previous task transitions")
    delivery = dict(
        source=release_identity(),
        uncertainty_source=uq_identity(),
        accepted_tasks=ACCEPTED,
        scopes=SCOPES,
        awaiting_external_evidence=EXTERNAL,
        tests=checks,
        source_recertified=True,
        previous_delivery_sha256=sha("validation/final17_delivery_v1.json"),
        installed_resources=760,
        openmc_seeds=[x["seed"] for x in q["transport"]],
        evidence=[
            "artifacts/final17_multiphysics_qualification_v3.json",
            "artifacts/final17_release_manifest_v3.json",
            "artifacts/final17_uq_qualification_v2.json",
            "validation/final17_build_witness_v2.json",
            "validation/final17_installed_v2.json",
            "docs/FINAL_17_IMPLEMENTATION_RU.md",
            "docs/EXTERNAL_ACCEPTANCE_HANDOFF_RU.md",
        ],
        validated=False,
        plant_referenced=False,
    )
    atomic_json("validation/final17_delivery_v2.json", seal(delivery))
    validation = unseal(read_json("validation/final17_validation_report_v1.json"))
    validation.update(
        source=release_identity(),
        uncertainty_source=uq_identity(),
        tests=checks,
        numerical_qualification="artifacts/final17_multiphysics_qualification_v3.json",
        transport=q["transport"],
        uncertainty=q["uncertainty"],
        covariance_qualification="artifacts/final17_uq_qualification_v2.json",
        installed_resources=760,
    )
    atomic_json("validation/final17_validation_report_v2.json", seal(validation))
    path = Path("validation/ai_industrial_workplan_v1.json")
    plan = read_json(path)
    if plan["version"] != "1.78.0":
        raise ValueError("Unexpected workplan version")
    plan["current_research_evidence"] = "validation/final17_delivery_v2.json"
    for package in plan["packages"]:
        for task in package["tasks"]:
            if task["id"] in ACCEPTED + list(EXTERNAL):
                task["evidence"] = list(
                    dict.fromkeys(
                        task["evidence"]
                        + delivery["evidence"]
                        + [
                            "validation/final17_delivery_v2.json",
                            "validation/final17_validation_report_v2.json",
                        ]
                    )
                )
    plan["execution_history"].append(
        dict(
            version="1.78.0",
            date="2026-11-06",
            note="Final source-bound recertification after nonfinite/unit/revision guards and explicit joint-covariance implementation",
            accepted_tasks_changed=False,
            evidence=delivery["evidence"],
        )
    )
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                accepted=13,
                external=4,
                source_current=True,
                tests=[73, 73],
                installed_resources=760,
            )
        )
    )


if __name__ == "__main__":
    main()
