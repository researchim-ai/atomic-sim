"""Replay a trusted primary/steam evidence bundle in a fresh source workspace.

Unlike evidence_bundle verify/extract, this command executes the frozen Python
auditors. The required manifest checksum must come from a trusted channel.
Dependencies are inherited; this runner does not install or fetch packages.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

from scripts.evidence_bundle import Limits, bounded_tarinfo, extract_bundle, safe_name


def _sdist(source, output, limits):
    output.mkdir()
    names, roots, total = set(), set(), 0
    with tarfile.open(source, "r:*", tarinfo=bounded_tarinfo(limits)) as archive:
        for member in archive:
            name = safe_name(member.name.rstrip("/"))
            parts = PurePosixPath(name).parts
            roots.add(parts[0])
            total += member.size
            if (
                len(roots) != 1
                or name.casefold() in names
                or len(names) >= limits.max_files
                or member.size < 0
                or member.size > limits.max_file_bytes
                or total > limits.max_total_bytes
            ):
                raise ValueError("Invalid or oversized sdist workspace")
            names.add(name.casefold())
            target = output / name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source_file, target.open("xb") as destination:
                    shutil.copyfileobj(source_file, destination, length=1024 * 1024)
            else:
                raise ValueError("Sdist links and special files are not executable evidence")
    if not roots:
        raise ValueError("Empty source distribution")
    return output / next(iter(roots))


def _plan(manifest, kinds):
    plan = manifest["metadata"].get("primary_steam_replay")
    allowed = {
        "release",
        "steady_qualification",
        "steady_audit",
        "transient_qualification",
        "transient_audit",
        "fixed_qualification",
        "fixed_audit",
    }
    if (
        not isinstance(plan, dict)
        or "release" not in plan
        or set(plan) - allowed
        or not isinstance(manifest["runtime"], dict)
    ):
        raise ValueError("Missing primary/steam replay plan")
    for name in plan.values():
        safe_name(name)
    if not kinds or len(kinds) != len(set(kinds)) or set(kinds) - {"steady", "transient", "fixed"}:
        raise ValueError("Select distinct steady, transient and/or fixed replay profiles")
    paths = []
    for kind in kinds:
        if not {kind + "_qualification", kind + "_audit"} <= set(plan):
            raise ValueError("Replay profile missing from the bundle")
        path = plan[kind + "_qualification"]
        if not path.startswith("artifacts/"):
            raise ValueError("Qualification must retain its repository-relative artifacts path")
        paths.append(path)
    if any(
        a == b or a.startswith(b + "/") or b.startswith(a + "/")
        for i, a in enumerate(paths)
        for b in paths[i + 1 :]
    ):
        raise ValueError("Overlapping qualification paths")
    return plan


def replay(archive, output, manifest_sha256, kinds=("steady",), wall_limit_s=3600, limits=Limits()):
    if type(wall_limit_s) not in (int, float) or not 0 < wall_limit_s <= 86400:
        raise ValueError("Replay wall limit must be positive and at most one day")
    deadline = time.monotonic() + wall_limit_s
    archive, output = Path(archive).resolve(), Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = False
    try:
        verified = extract_bundle(archive, output / "payload", manifest_sha256, limits)
        payload = output / "payload"
        manifest = json.loads((payload / "MANIFEST.json").read_text())
        plan = _plan(manifest, kinds)
        runtime = _sdist(payload / manifest["runtime"]["sdist"], output / "source", limits)
        release = payload / plan["release"]
        for folder, destination in (
            ("frozen_scripts", "scripts"),
            ("frozen_budgets", "validation"),
        ):
            (runtime / destination).mkdir(exist_ok=True)
            files = sorted((release / folder).iterdir())
            if not files:
                raise ValueError("Missing frozen replay tools or budgets")
            for file in files:
                if not file.is_file():
                    raise ValueError("Unexpected frozen companion directory")
                target = runtime / destination / file.name
                if target.exists() and target.read_bytes() != file.read_bytes():
                    raise ValueError("Frozen companion conflicts with source distribution")
                shutil.copyfile(file, target)
        clean = dict(os.environ)
        clean.pop("PYTHONPATH", None)
        results = []
        for kind in kinds:
            qualification = plan[kind + "_qualification"]
            (runtime / qualification).parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(payload / qualification, runtime / qualification)
            expected = json.loads((payload / plan[kind + "_audit"]).read_text())
            signature = expected.get("signature", {})
            assembly = signature.get("assembly", signature)
            if (
                expected.get("passed") is not True
                or assembly.get("source_sha256") != verified["source_sha256"]
            ):
                raise ValueError("Expected audit does not match bundled runtime")
            target = output / (kind + "-audit.json")
            if kind in ("steady", "fixed"):
                command = [
                    sys.executable,
                    "-m",
                    "scripts.qualify_primary_steam_design"
                    if kind == "steady"
                    else "scripts.qualify_fixed_primary_steam",
                    "--audit",
                    qualification,
                    "--output",
                    str(target),
                ]
            else:
                command = [
                    sys.executable,
                    "-m",
                    "scripts.audit_primary_steam",
                    qualification,
                    "--output",
                    str(target),
                ]
            started = True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Replay wall limit elapsed during preparation")
            with (output / (kind + "-audit.log")).open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=runtime,
                    env=clean,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=remaining,
                    check=False,
                )
            if result.returncode:
                raise RuntimeError(f"Frozen {kind} auditor failed with code {result.returncode}")
            actual = json.loads(target.read_text())
            if actual != expected:
                raise ValueError("Portable audit differs from the archived accepted report")
            results.append(
                dict(
                    profile=kind,
                    passed=True,
                    archived_audit_identical=True,
                    integrations=actual["integrations"],
                    case_families=actual["case_families"],
                    negative_checks=actual.get("negative_checks", 0),
                )
            )
        report = dict(
            schema_version=1,
            passed=True,
            manifest_sha256=verified["manifest_sha256"],
            source_sha256=verified["source_sha256"],
            profiles=results,
            dependencies="Inherited environment; frozen auditor checks exact Python/numpy/scipy versions",
            numerical_acceptance_recomputed=True,
            validated=False,
            plant_referenced=False,
        )
        (output / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    except BaseException as error:
        if not started:
            shutil.rmtree(output)
        else:
            (output / "replay.json").write_text(
                json.dumps(
                    dict(
                        schema_version=1,
                        passed=False,
                        error_type=type(error).__name__,
                        error=str(error),
                        numerical_acceptance_recomputed=False,
                    ),
                    indent=2,
                )
                + "\n"
            )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument(
        "--profile", choices=("steady", "transient", "fixed"), action="append", required=True
    )
    parser.add_argument("--wall-limit-s", type=float, default=3600)
    args = parser.parse_args()
    report = replay(
        args.archive, args.output, args.manifest_sha256, args.profile, args.wall_limit_s
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
