"""Select exact source/runtime/data/evidence bytes for the numerical publication."""

import argparse
import json
from pathlib import Path

from atomic_model_controlled_plant.session import identity
from scripts.extended_plant_release import envelope
from sdk.experiment import _source_fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if not envelope("validation/ai050607_controlled_summary_v1.json")[
        "all_three_blocks_numerical_acceptance"
    ]:
        raise ValueError("Cannot publish incomplete numerical acceptance")
    entries = {}

    def add(source, target):
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        if target in entries:
            raise ValueError("Duplicate archive destination")
        entries[target] = str(source)

    def tree(source, target):
        root = Path(source)
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                add(path, (Path(target) / path.relative_to(root)).as_posix())

    for name in [
        "atomic_model_controlled_plant",
        "atomic_model_plant_extension",
        "atomic_model_phase_assessment",
    ]:
        tree(name, "review_sources/" + name)
    for name in [
        "assess_controlled_plant",
        "controlled_plant_release",
        "describe_controlled_plant",
        "run_controlled_plant_campaign",
        "replay_controlled_programs",
        "recertify_controlled_metadata",
        "qualify_controlled_events",
        "extended_natural_initialization",
        "qualify_extended_natural",
        "qualify_extended_events",
        "fetch_pool_primary",
        "verify_pool_primary",
        "verify_controlled_install",
        "build_extended_plant_evidence",
        "select_controlled_evidence",
        "accept_050607_numerical_delivery",
        "write_050607_delivery",
    ]:
        add("scripts/" + name + ".py", "review_sources/scripts/" + name + ".py")
    for name in [
        "test_controlled_plant",
        "test_plant_extension",
        "test_phase_assessment",
        "test_phase_primary_transport",
        "test_extended_natural_initialization",
    ]:
        add("tests/" + name + ".py", "review_sources/tests/" + name + ".py")
    for name in [
        "AI_05_06_07_EXTENDED_MODEL_RU.md",
        "CURRENT_STATUS_RU.md",
        "AI_INDUSTRIAL_EXECUTION_PLAN_RU.md",
        "READINESS_REVIEW_RU.md",
    ]:
        add("docs/" + name, "review_sources/docs/" + name)
    for name in [
        "README.md",
        "pyproject.toml",
        ".gitattributes",
        ".github/workflows/ci.yml",
        ".github/workflows/numerical.yml",
    ]:
        add(name, "review_sources/" + name)
    for pattern in [
        "ai050607_controlled_*.json",
        "ai07_controlled_*.json",
        "ai06_pool_b274_*.json",
        "ai050607_extended_environment_v1.json",
        "ai050607_external_dependencies_v1.json",
        "ai_industrial_workplan_v1.json",
    ]:
        for path in sorted(Path("validation").glob(pattern)):
            if "publication" not in path.name and "bundle_selection" not in path.name:
                add(path, "validation/" + path.name)
    tree("artifacts/ai050607_controlled_final_v1", "current_campaign")
    tree("artifacts/ai050607_controlled_campaign_v2", "parent_campaign")
    tree("artifacts/ai050607_controlled_dev_source_v2", "parent_controlled_source")
    tree("artifacts/ai050607_base_dev_source_v1", "parent_base_source")
    tree("artifacts/ai050607_controlled_campaign_v1/load_rejection", "failed_600s_load")
    tree("artifacts/ai050607_controlled_dev_source_v1", "failed_600s_source")
    tree("artifacts/ai050607_final_checks_v1", "checks")
    for name in [
        "ai07_controlled_events_final_v1.json",
        "ai06_four_phenomena_v2.json",
        "ai06_pool_b274_screen_final_v2.json",
        "ai06_pool_b274_primary.csv.gz",
    ]:
        add("artifacts/" + name, "evidence/" + name)
    # Retain the original legacy plant full-request failures without relabelling.
    tree("artifacts/ai050607_plant_campaign_v1", "historical_legacy_campaign")
    wheel = next(a.build.glob("*.whl"))
    sdist = next(a.build.glob("*.tar.gz"))
    add(wheel, "runtime/" + wheel.name)
    add(sdist, "runtime/" + sdist.name)
    selection = dict(
        schema_version=1,
        bundle_id="ai050607-controlled-numerical-v1",
        scope="Source-bound bounded numerical acceptance; no experimental or real-plant attestation",
        entries=[dict(source=source, path=target) for target, source in sorted(entries.items())],
        metadata=dict(
            source=identity(),
            all_three_blocks_numerical_acceptance=True,
            validated=False,
            plant_referenced=False,
        ),
        runtime=dict(
            wheel="runtime/" + wheel.name,
            sdist="runtime/" + sdist.name,
            source_sha256=_source_fingerprint(),
        ),
    )
    with a.output.open("x") as f:
        json.dump(selection, f, indent=2)
        f.write("\n")
    print("selected", len(entries), "files")


if __name__ == "__main__":
    main()
