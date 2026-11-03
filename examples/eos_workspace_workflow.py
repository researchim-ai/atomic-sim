"""Project -> two baselines -> campaign -> compare -> archive -> exact replay."""

import argparse
import json
import tempfile
from pathlib import Path

from atomic_model_workspace.bundle import export_bundle, import_bundle
from atomic_model_workspace.campaign import create_campaign, load_campaign, run_campaign
from atomic_model_workspace.common import digest
from atomic_model_workspace.jobs import load_job
from atomic_model_workspace.project import validate_project
from atomic_model_workspace.reports import compare_campaigns, export_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).parent / "eos_workspace/project.json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="atomic-workspace-workflow-"))
    output.mkdir(parents=True, exist_ok=True)
    admitted = validate_project(args.project)
    create_campaign(args.project, output / "full")
    assert run_campaign(output / "full")["status"] == "completed"
    create_campaign(args.project, output / "split")
    assert run_campaign(output / "split", max_jobs=1)["status"] == "paused"
    archive = export_bundle(args.project, output / "campaign.zip", campaign=output / "split")
    restored = import_bundle(output / "campaign.zip", output / "restored")
    assert run_campaign(restored["campaign"], resume=True)["status"] == "completed"
    comparison = compare_campaigns(output / "full", restored["campaign"])
    assert comparison["comparable"]
    full = load_campaign(output / "full")
    repeat = load_campaign(restored["campaign"])
    episodes = {}
    for a, b in zip(full["entries"], repeat["entries"]):
        left = load_job(output / "full" / a["job"])
        right = load_job(Path(restored["campaign"]) / b["job"])
        assert left["episode"] == right["episode"]
        episodes[a["scenario_id"] + "/" + a["policy_id"]] = digest(left["episode"])
    report = export_report(output / "full", output / "report")
    result = dict(
        schema_version=1,
        project_sha256=admitted["project_sha256"],
        jobs=len(episodes),
        policies=admitted["policies"],
        portable_archive=True,
        archive=archive,
        pause_resume_episodes_exact=True,
        numerical_episodes=episodes,
        all_full_horizons=all(j["metrics"]["completed"] for j in report["jobs"]),
        validated=False,
        plant_referenced=False,
        output=str(output),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
