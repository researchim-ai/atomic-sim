"""Audit historical evidence integrity without promoting physical model status."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_folders(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from artifact_folders(key)
            yield from artifact_folders(item)
    elif isinstance(value, list):
        for item in value:
            yield from artifact_folders(item)
    elif isinstance(value, str) and value.startswith("artifacts/"):
        path = Path(value)
        if path.is_dir() and (path / "manifest.json").is_file():
            yield path


def audit():
    manifest_path = Path("validation/run_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    budgets = {sha(p): str(p) for p in sorted(Path("validation").glob("*.json"))}
    modes = []
    for mode in manifest["modes"]:
        accepted = mode.get("numerical_full_horizon_accepted", False)
        row = dict(
            id=mode["id"],
            numerical_claim=accepted,
            physical_qualified=False,
            status=mode.get("status"),
            historical_evidence=mode.get("evidence"),
            note=mode.get("note", ""),
        )
        row["horizon_s"] = mode.get("horizon_s")
        row["completed_time_s"] = mode.get("completed_time_s")
        if accepted:
            path = Path(mode["evidence"])
            report = json.loads(path.read_text())
            folders = sorted(set(artifact_folders(report)))
            if not folders:
                raise ValueError("No traceable bundles for " + mode["id"])
            inputs = []
            for folder in folders:
                m = json.loads((folder / "manifest.json").read_text())
                if not m.get("source_sha256") or not m.get("passport"):
                    raise ValueError("Missing source/passport " + str(folder))
                for name, digest in m["artifacts"].items():
                    if sha(folder / name) != digest:
                        raise ValueError("Artifact checksum mismatch " + str(folder / name))
                spec = json.loads((folder / "input.json").read_text())
                r = json.loads((folder / "report.json").read_text())
                if (
                    not r["numerical_acceptance"]
                    or r["status"] != "completed"
                    or r["completed_time_s"] != spec["duration_s"]
                ):
                    raise ValueError("Unaccepted referenced bundle " + str(folder))
                inputs.append(
                    dict(
                        bundle=str(folder),
                        input_sha256=sha(folder / "input.json"),
                        source_sha256=m["source_sha256"],
                        passport=m["passport"],
                        horizon_s=spec["duration_s"],
                        budgets=spec.get("budgets"),
                    )
                )
            budget_hash = report.get("budget_sha256")
            if budget_hash is not None and budget_hash not in budgets:
                raise ValueError("Missing original budget for " + mode["id"])
            row.update(
                evidence_sha256=sha(path),
                bundles=inputs,
                comparison_budget=budgets.get(budget_hash),
                integrity_passed=True,
            )
        modes.append(row)
    recent = []
    for name in (
        "muscl_total_initialization_delivery_v1.json",
        "muscl_total_heated_temporal_v1.json",
        "muscl_total_heated_reflection_v1.json",
        "total_reflection_v1.json",
    ):
        path = Path("validation") / name
        data = json.loads(path.read_text())
        if (
            not data["passed"]
            or data.get("validated", False)
            or data.get("plant_referenced", False)
        ):
            raise ValueError("Unexpected recent evidence status " + name)
        recent.append(
            dict(path=str(path), sha256=sha(path), kind="historical_numerical_verification")
        )
    return dict(
        schema_version=1,
        passed=True,
        kind="historical_evidence_integrity",
        manifest_sha256=sha(manifest_path),
        modes=modes,
        recent_component_evidence=recent,
        validated=False,
        plant_referenced=False,
        scope="Metadata/checksums of prior accepted numerical modes. Not re-execution, physical validation, or promotion to current runtime.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(passed=True, modes=len(report["modes"]))))


if __name__ == "__main__":
    main()
