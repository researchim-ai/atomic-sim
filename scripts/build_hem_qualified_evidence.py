"""Build and independently verify the deterministic bounded HEM evidence bundle."""

import argparse
import json
import tempfile
from pathlib import Path

from atomic_model_hem_qualified.model import signature
from atomic_model_workspace.common import digest
from scripts.evidence_bundle import create_bundle, runtime_fingerprint, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    campaign = Path("artifacts/ai06_hem_qualified_publication_v1.json")
    record = json.loads(campaign.read_text())
    if record["sha256"] != digest(record["payload"]) or record["payload"]["source"] != signature():
        raise ValueError("Stale numerical evidence")
    installed = json.loads(Path("validation/ai06_hem_qualified_installed_v1.json").read_text())
    if (
        installed["source"] != signature()
        or len(installed["steps"]) != 6
        or any(s["returncode"] for s in installed["steps"])
    ):
        raise ValueError("Missing installed qualification")
    entries = []

    def add(source, target):
        entries.append(dict(source=str(source), path=target))

    files = list(Path("atomic_model_hem_qualified").glob("*.py")) + [
        Path(p)
        for p in [
            "tests/test_hem_qualified.py",
            "scripts/verify_hem_qualified_install.py",
            "scripts/build_hem_qualified_evidence.py",
            "README.md",
            "docs/AI_06_QUALIFIED_HEM_RU.md",
            "docs/AI_INDUSTRIAL_EXECUTION_PLAN_RU.md",
            "docs/READINESS_REVIEW_RU.md",
            "examples/qualified_hem.json",
            "pyproject.toml",
            "validation/ai_industrial_workplan_v1.json",
            "validation/ai06_hem_qualified_chf_curation_v1.json",
            "validation/ai06_hem_qualified_summary_v1.json",
            "validation/ai06_hem_qualified_passport_v1.json",
            "validation/ai06_hem_qualified_installed_v1.json",
        ]
    ]
    for p in files:
        add(p, "review_sources/" + str(p))
    add(campaign, "verification_inputs/campaign.json")
    add("tests/test_hem_qualified.py", "verification_inputs/test_hem_qualified.py")
    add("scripts/verify_hem_qualified_install.py", "verification/verify.py")
    add("artifacts/ai06_hem_qualified_publication_demo_v1.json", "verification_inputs/demo.json")
    for source, target in [
        ("artifacts/ai06_hem_network_demo_v1", "old_graph"),
        ("artifacts/ai06_hem_mixture_demo_v1", "old_mixture"),
    ]:
        for p in sorted(Path(source).rglob("*")):
            if p.is_file():
                add(p, "verification_inputs/" + target + "/" + str(p.relative_to(source)))
    wheel = Path("dist/atomic_sim-2.0.0-py3-none-any.whl")
    sdist = Path("dist/atomic_sim-2.0.0.tar.gz")
    fp = runtime_fingerprint(wheel.read_bytes(), "wheel")
    if fp != runtime_fingerprint(sdist.read_bytes(), "sdist"):
        raise ValueError("Foundation source mismatch")
    for p in [wheel, sdist]:
        add(p, "runtime/" + p.name)
    selection = dict(
        schema_version=1,
        bundle_id="ai06-hem-qualified-v1",
        scope="Bounded numerical HEM qualification only; shared IF97 EOS; no experimental/plant promotion",
        entries=entries,
        runtime=dict(
            wheel="runtime/" + wheel.name, sdist="runtime/" + sdist.name, source_sha256=fp
        ),
    )
    result = create_bundle(selection, root, args.output)
    verified = verify_bundle(args.output)
    with tempfile.TemporaryDirectory(prefix="qualified-rebuild-") as folder:
        other = Path(folder) / "copy.tar.gz"
        create_bundle(selection, root, other)
        if other.read_bytes() != args.output.read_bytes():
            raise ValueError("Nondeterministic evidence archive")
    Path("validation/ai06_hem_qualified_bundle_selection_v1.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(build=result, verification=verified, deterministic_rebuild_exact=True), indent=2
        )
    )


if __name__ == "__main__":
    main()
