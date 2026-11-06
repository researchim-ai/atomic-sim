"""Close verified software/numerical tasks; retain four real external gates."""

import json
import re
import shutil
from collections import Counter
from pathlib import Path

from atomic_model_multiphysics.provenance import identity as model_identity
from atomic_model_multiphysics.provenance import seal, sha, unseal
from atomic_model_research.release import identity as release_identity
from atomic_model_training.episode import identity as training_identity
from atomic_model_training.spec import atomic_json, read_json
from sdk.experiment import _source_fingerprint

ACCEPTED = [
    "AI-08.01",
    "AI-08.02",
    "AI-08.04",
    "AI-08.05",
    "AI-08.06",
    "AI-08.07",
    "AI-08.08",
    "AI-11.01",
    "AI-12.01",
    "AI-12.02",
    "AI-12.06",
    "AI-13.05",
    "AI-13.06",
]
EXTERNAL = {
    "AI-09.02": "Applicable coupled-system measurement histories with configuration, license and uncertainty are absent",
    "AI-09.06": "Experimental interval coverage cannot be accepted without measurement uncertainty/covariance and applicable held-out system cases",
    "AI-09.08": "Independent technical reviewer and review receipt are absent; own checks are not independent review",
    "AI-11.06": "Portable/cold replay executed on this host only; another physical machine receipt is absent",
}
SCOPES = {
    "AI-08.01": "Separate synthetic and ENDF/B-VIII.0 evaluated homogeneous research-mixture families; raw isotope SHA/license/group/unit/normalization manifests. Not a target-plant data set.",
    "AI-08.02": "Actual licensed local OpenMC 0.16.1.dev0 subprocess, three seeds, retained XML/HDF5, two-group MGXS means/std, explicit units/axes and offline import; static homogeneous reflecting slab.",
    "AI-08.04": "Steady synthetic-feedback slab/radial FV rod/constant-cp channel, nonmatching conservative maps and declared residual criteria; not dynamic boiling or plant thermal coupling.",
    "AI-08.05": "Five-axis bounded table schema with strict no extrapolation and synthetic affine interpolation/coupling verification. Evaluated temperature/composition response grids are not claimed.",
    "AI-08.06": "Numerical capture/decay chains, three independent inventories, energy-augmented CRAM16 versus expm and CE/CM second-order rational reference. No evaluated fuel burnup/yield library or ANS decay-heat claim.",
    "AI-08.07": "1D exact absorber overlap, subcell sweep, three mesh levels, dense eigenvalue references and manufactured peak convergence. No real-reactor 3D peak claim.",
    "AI-08.08": "Source-bound steady operating-point spatial agent with power actions/revision/transaction/replay on the qualified synthetic-feedback slab scope; decisions are not reactor transient time steps.",
    "AI-11.01": "Versioned default controlled-PWR model/scenario/data/declarative-agent project, immutable ancestry, CAS data and bounded portable ZIP. No arbitrary topology/model editor.",
    "AI-12.01": "One exact qualified Linux x86_64 CPython3.12.9 NumPy2.2.6 SciPy1.15.2 numerical profile; minimum=selected pins, observed dependency constraints. Other runtimes and hermetic wheelhouse are excluded.",
    "AI-12.02": "Cold offline no-deps wheel outside checkout with inherited pinned scientific dependencies, installed JSON/YAML resource inspection, actual-core portable and spatial replay, CLI and prior runbook. No other-machine or hermetic claim.",
    "AI-12.06": "Source/version/dependency/license inventory, SPDX SBOM with honest NOASSERTION, package/source checksums and local evidence archive; no authenticated publisher/signature channel.",
    "AI-13.05": "Technical local open research release candidate, installation/examples/replay/limits/failures explicitly checked. Public push/tag/release publishing remains user-controlled.",
    "AI-13.06": "Only retained measured5.277x nominal short controlled-core workload claim with unchanged source, separate new-model schemas/qualifications, no universal simulator or spatial-speedup claim.",
}


def main():
    qualification = unseal(read_json("artifacts/final17_multiphysics_qualification_v2.json"))
    installed = unseal(read_json("/tmp/atomic-final17-cold-v1/installed-proof.json"))
    legacy = read_json("artifacts/final17_legacy_installed_v1/proof.json")
    witness = read_json("validation/final17_build_witness_v1.json")
    manifest = unseal(read_json("artifacts/final17_release_manifest_v2.json"))
    prior_performance = unseal(read_json("artifacts/ai091012_training_benchmark_v2.json"))
    if (
        not qualification["numerical_accepted"]
        or not qualification["openmc_pipeline_executed"]
        or qualification["source"] != model_identity()
        or installed["source"] != release_identity()
        or witness["source"] != release_identity()
        or manifest["source"] != release_identity()
        or not installed["project_run"]["payload"]["report"]["completed"]
        or not installed["spatial_replay_equal"]
        or installed["resource_count"] < 700
        or not all(x["passed"] for x in legacy["phases"])
        or not manifest["matrix"]["observed_matches_supported"]
        or manifest["dependencies"]["missing"]
        or prior_performance["source"] != training_identity()
        or prior_performance["median_speedup"] < 2
    ):
        raise ValueError("Incomplete or stale delivered checks")
    if len({r["seed"] for r in qualification["transport"]}) != 3:
        raise ValueError("Independent transport seeds missing")
    for name in ["wheel", "sdist"]:
        item = witness[name]
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Build changed")
    if legacy["wheel_sha256"] != witness["wheel"]["sha256"]:
        raise ValueError("Legacy check used different wheel")
    for item in witness["runtime_files"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Runtime source changed")
    if _source_fingerprint() != "71fe3f4ce5c9f3e4dc86a36c17c1cdb32470ca19e0214ebc9c91af1c1a997d67":
        raise ValueError("Prior physical foundation changed")
    logs = []
    for name, source, count in [
        ("regression", "/tmp/final17-regression.log", 156),
        ("installed", "/tmp/final17-installed-tests.log", 51),
    ]:
        text = Path(source).read_text()
        matches = re.findall(r"\b(\d+) passed\b", text)
        if (
            not matches
            or int(matches[-1]) != count
            or re.search(r"\d+ (failed|errors|skipped)", text)
        ):
            raise ValueError("Tests incomplete: " + name)
        target = Path("artifacts") / ("final17_" + name + ".log")
        shutil.copyfile(source, target)
        logs.append(dict(path=str(target), sha256=sha(target), passed=count))
    shutil.copyfile(
        "/tmp/atomic-final17-cold-v1/installed-proof.json", "validation/final17_installed_v1.json"
    )
    report = dict(
        source=release_identity(),
        numerical_qualification="artifacts/final17_multiphysics_qualification_v2.json",
        independent_review=dict(status="awaiting_external_evidence", reviewer=None),
        external_gates=EXTERNAL,
        tests=logs,
        installed_resources=installed["resource_count"],
        transport=qualification["transport"],
        uncertainty=qualification["uncertainty"],
        retained_negative_cases=[
            dict(
                path="artifacts/ai06_pool_b274_screen_final_v2.json",
                sha256=sha("artifacts/ai06_pool_b274_screen_final_v2.json"),
                result="No production forced-boiling admitted rows; pool data do not validate this flow closure",
            ),
            dict(
                path="artifacts/ai09_identification_assessment_v1.json",
                sha256=sha("artifacts/ai09_identification_assessment_v1.json"),
                result="Prior physical acceptance/calibration rejection preserved; not retuned",
            ),
        ],
        validated=False,
        plant_referenced=False,
    )
    atomic_json("validation/final17_validation_report_v1.json", seal(report))
    basis = [
        "docs/FINAL_17_IMPLEMENTATION_RU.md",
        "validation/ai_final17_protocol_v1.json",
        "validation/final17_build_witness_v1.json",
        "validation/final17_installed_v1.json",
        "validation/final17_validation_report_v1.json",
        "validation/final17_supported_runtime_v1.json",
        "artifacts/final17_multiphysics_qualification_v2.json",
        "artifacts/final17_release_manifest_v2.json",
    ]
    delivery = dict(
        source=release_identity(),
        accepted_tasks=ACCEPTED,
        scopes=SCOPES,
        awaiting_external_evidence=EXTERNAL,
        evidence=basis,
        tests=logs,
        no_external_gate_promoted=True,
        validated=False,
        plant_referenced=False,
    )
    atomic_json("validation/final17_delivery_v1.json", seal(delivery))
    path = Path("validation/ai_industrial_workplan_v1.json")
    plan = read_json(path)
    if plan["version"] != "1.77.0":
        raise ValueError("Unexpected plan version")
    for package in plan["packages"]:
        for task in package["tasks"]:
            if task["id"] in ACCEPTED:
                if task["status"] == "accepted":
                    raise ValueError("Unexpected accepted task")
                task.update(status="accepted", research_delivery_scope=SCOPES[task["id"]])
                task["evidence"] = list(
                    dict.fromkeys(
                        task.get("evidence", []) + basis + ["validation/final17_delivery_v1.json"]
                    )
                )
                plan["acceptance_decisions"].append(
                    dict(
                        date="2026-11-06",
                        task=task["id"],
                        decision="accepted",
                        scope=SCOPES[task["id"]],
                        basis=basis,
                        validated=False,
                        plant_referenced=False,
                    )
                )
            elif task["id"] in EXTERNAL:
                task.update(
                    status="awaiting_external_evidence",
                    external_gate=EXTERNAL[task["id"]],
                    infrastructure_implemented=True,
                )
                task["evidence"] = list(dict.fromkeys(task.get("evidence", []) + basis))
        if package["id"] in ("AI-08", "AI-12"):
            if not all(t["status"] == "accepted" for t in package["tasks"]):
                raise ValueError("Unaccepted package tasks")
            package.update(
                status="accepted",
                research_delivery_scope="Explicit numerical research scopes; not industrial physical/plant qualification",
            )
            package["evidence"] = basis
        if package["id"] == "AI-13":
            package["status"] = "partial"
            package["remaining_dependency"] = (
                "AI-11.06 another-machine replay; own research-release software tasks accepted"
            )
    plan.update(version="1.78.0", current_research_evidence="validation/final17_delivery_v1.json")
    plan["execution_history"].append(
        dict(
            date="2026-11-06",
            version="1.78.0",
            accepted_tasks=ACCEPTED,
            awaiting_external_evidence=EXTERNAL,
            evidence=basis,
            note="13 research software/numerical tasks closed; four external gates retained honestly",
        )
    )
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                accepted=len(ACCEPTED),
                external=len(EXTERNAL),
                counts=Counter(t["status"] for p in plan["packages"] for t in p["tasks"]),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
