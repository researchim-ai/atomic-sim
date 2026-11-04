"""Versioned parameter edit -> export -> campaign -> archive -> restore config."""

import argparse
import json
import tempfile
from pathlib import Path

from atomic_model_projects import store
from atomic_model_workspace.bundle import export_bundle, import_bundle
from atomic_model_workspace.campaign import create_campaign, load_campaign, run_campaign
from atomic_model_workspace.common import digest
from atomic_model_workspace.jobs import create_job, load_job, run_job
from atomic_model_workspace.project import load_project


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--project", type=Path, default=Path(__file__).parent / "eos_workspace/project.json"
    )
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="atomic-project-edit-"))
    output.mkdir(parents=True, exist_ok=True)
    root = output / "store"
    store.create(args.project, root)
    original = store.inspect(root)
    req = dict(
        schema_version=1,
        expected_revision=original["revision"],
        expected_project_sha256=original["project_sha256"],
        operations=[
            dict(op="set_scenario_config", id="mixed_dev", value=dict(episode_horizon_s=0.04))
        ],
        note="extend supported mixed-dev horizon",
    )
    preview = store.preview(root, req)
    assert store.inspect(root)["revision"] == 0
    store.edit(root, req)
    edited = store.inspect(root)
    assert edited["project_sha256"] == preview["proposed_project_sha256"]
    try:
        store.edit(root, req)
    except ValueError:
        pass
    else:
        raise AssertionError("Stale edit accepted")
    exported = store.export(root, output / "exported")
    assert load_project(exported["project"]) == edited["project"]
    campaign = output / "campaign"
    create_campaign(exported["project"], campaign)
    assert run_campaign(campaign)["status"] == "completed"
    state = load_campaign(campaign)
    entry = next(
        e for e in state["entries"] if e["scenario_id"] == "mixed_dev" and e["policy_id"] == "hold"
    )
    job = load_job(campaign / entry["job"])
    create_job(job["spec"], output / "control-replay")
    repeated = run_job(output / "control-replay")
    assert repeated["episode"] == job["episode"]
    archive = export_bundle(exported["project"], output / "edited-project.zip", campaign=campaign)
    imported = import_bundle(output / "edited-project.zip", output / "imported")
    assert load_project(imported["project"]) == edited["project"]
    store.restore(
        root,
        expected_revision=edited["revision"],
        expected_project_sha256=edited["project_sha256"],
        target_revision=0,
    )
    restored = store.inspect(root)
    assert restored["revision"] == 2 and restored["project"] == original["project"]
    result = dict(
        schema_version=1,
        configuration_undo_appends_revision=True,
        preview_matches_commit=True,
        stale_edit_rejected=True,
        export_and_archive_exact=True,
        edited_episode_replay_exact=True,
        jobs=len(state["entries"]),
        original_project_sha256=original["project_sha256"],
        edited_project_sha256=edited["project_sha256"],
        replayed_native_episode_sha256=digest(job["episode"]),
        archive=archive,
        validated=False,
        plant_referenced=False,
        output=str(output),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
