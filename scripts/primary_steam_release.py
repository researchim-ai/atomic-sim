"""Collect a source-matched primary/steam release into a portable evidence bundle."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.evidence_bundle import create_bundle, safe_name


def selection(
    root,
    release,
    *,
    steady=None,
    steady_audit=None,
    transient=None,
    transient_audit=None,
    fixed=None,
    fixed_audit=None,
):
    root = Path(root).resolve()

    def relative(value):
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if not path.absolute().is_relative_to(root):
            raise ValueError("Release evidence must be inside the selected root")
        return safe_name(path.absolute().relative_to(root).as_posix())

    release = relative(release)
    entries, destinations = [], {}

    def add(source, destination=None):
        source = relative(source)
        destination = safe_name(destination or source)
        if destination not in destinations:
            entries.append(dict(source=source, path=destination))
            destinations[destination] = source
        elif source != destinations[destination]:
            old = root / destinations[destination]
            if (
                not old.is_file()
                or not (root / source).is_file()
                or old.read_bytes() != (root / source).read_bytes()
            ):
                raise ValueError("Conflicting evidence destination")

    plan, identities, signatures = dict(release=release), set(), []
    profiles = (
        ("steady", steady, steady_audit, 14, 7),
        ("transient", transient, transient_audit, 53, 17),
        ("fixed", fixed, fixed_audit, None, None),
    )
    for kind, qualification, audit, count, families in profiles:
        if qualification is None and audit is None:
            continue
        if qualification is None or audit is None:
            raise ValueError("Each qualification requires its accepted audit")
        qpath, apath = relative(qualification), relative(audit)
        if not qpath.startswith("artifacts/"):
            raise ValueError("Qualification paths must retain their artifacts prefix")
        producer = json.loads((root / qpath / "qualification.json").read_text())
        checked = json.loads((root / apath).read_text())
        if kind == "fixed":
            declared = json.loads((root / qpath / "budget.json").read_text())
            families = len(declared["cases"])
            count = families * len(declared["policies"])
            if checked.get("negative_checks") != len(declared["negative_cases"]):
                raise ValueError("Fixed-boundary rejection matrix is incomplete")
            if (
                producer.get("signature", {}).get("budget_sha256")
                != hashlib.sha256((root / qpath / "budget.json").read_bytes()).hexdigest()
            ):
                raise ValueError("Fixed-boundary matrix budget differs from captured source")
        if (
            producer.get("passed") is not True
            or checked.get("passed") is not True
            or producer.get("integrations") != count
            or checked.get("integrations") != count
            or producer.get("case_families") != families
            or checked.get("case_families") != families
            or producer.get("signature") != checked.get("signature")
            or checked.get("qualification_sha256")
            != hashlib.sha256((root / qpath / "qualification.json").read_bytes()).hexdigest()
        ):
            raise ValueError("Qualification and accepted audit disagree")
        signature = checked["signature"]
        assembly = signature.get("assembly", signature)
        identities.add(assembly["source_sha256"])
        signatures.append(signature)
        plan[kind + "_qualification"], plan[kind + "_audit"] = qpath, apath
        add(qpath)
        add(apath)
    if not signatures or len(identities) != 1:
        raise ValueError("Select accepted profiles with exactly one runtime source identity")
    distributions = {}
    for kind, pattern in (("wheel", "*.whl"), ("sdist", "*.tar.gz")):
        found = sorted((root / release).glob(pattern))
        if len(found) != 1:
            raise ValueError("Release directory must contain one wheel and one sdist")
        distributions[kind] = relative(found[0])
        add(found[0])
    required = {}
    for signature in signatures:
        assembly = signature.get("assembly", signature)
        for name, digest in assembly["driver_sha256"].items():
            if name in required and required[name] != digest:
                raise ValueError("Profiles use different numerical drivers")
            required[name] = digest
        if "designer_sha256" in signature:
            required["scripts/design_primary_steam.py"] = signature["designer_sha256"]
            required["scripts/qualify_primary_steam_design.py"] = signature["qualification_sha256"]
        if "initializer_driver_sha256" in signature:
            for name, digest in signature["initializer_driver_sha256"].items():
                if name in required and required[name] != digest:
                    raise ValueError("Profiles use different initialization drivers")
                required[name] = digest
            required["scripts/qualify_fixed_primary_steam.py"] = signature["qualification_sha256"]
            required["scripts/qualify_primary_steam_design.py"] = signature[
                "stationary_verifier_sha256"
            ]
    names = set(required) | {"scripts/turbine_references.py"}
    for name in sorted(names):
        frozen = root / release / "frozen_scripts" / Path(name).name
        source = frozen if frozen.exists() else root / name
        if name in required and hashlib.sha256(source.read_bytes()).hexdigest() != required[name]:
            raise ValueError("Numerical driver differs from accepted evidence: " + name)
        add(source, release + "/frozen_scripts/" + Path(name).name)
    budgets = sorted((root / release / "frozen_budgets").glob("*.json"))
    budgets += sorted((root / "validation").glob("ai05_*budget*.json"))
    for signature in signatures:
        assembly = signature.get("assembly", signature)
        hashes = {assembly["budget_sha256"]}
        if "design_budget_sha256" in signature:
            hashes.add(signature["design_budget_sha256"])
        if "initializer_driver_sha256" in signature:
            hashes.update((signature["steady_budget_sha256"], signature["budget_sha256"]))
        for digest in hashes:
            matching = [p for p in budgets if hashlib.sha256(p.read_bytes()).hexdigest() == digest]
            if not matching:
                raise ValueError("Accepted numerical budget missing from release")
            for budget in matching:
                add(budget, release + "/frozen_budgets/" + budget.name)
    for name in (
        "scripts/evidence_bundle.py",
        "scripts/replay_primary_steam_bundle.py",
        "scripts/primary_steam_release.py",
        "docs/AI_12_EVIDENCE_BUNDLE_RU.md",
        "tests/test_evidence_bundle.py",
        "tests/test_evidence_replay.py",
        "tests/test_primary_steam_release.py",
        "tests/test_fixed_primary_steam.py",
        "tests/test_fixed_primary_steam_evidence.py",
        "docs/AI_05_FIXED_BOUNDARY_STEADY_RU.md",
        ".github/workflows/ci.yml",
        "LICENSE",
    ):
        if (root / name).is_file():
            add(name)
    source = next(iter(identities))
    return dict(
        schema_version=1,
        bundle_id="primary-steam-" + source[:12],
        scope="Source-matched numerical evidence; physical validation and plant reference are not claimed",
        entries=entries,
        runtime=dict(**distributions, source_sha256=source),
        metadata=dict(primary_steam_replay=plan, validated=False, plant_referenced=False),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--release", type=Path, required=True)
    for name in ("steady", "steady-audit", "transient", "transient-audit", "fixed", "fixed-audit"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = selection(
        args.root,
        args.release,
        steady=args.steady,
        steady_audit=args.steady_audit,
        transient=args.transient,
        transient_audit=args.transient_audit,
        fixed=args.fixed,
        fixed_audit=args.fixed_audit,
    )
    result = create_bundle(selected, args.root, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
