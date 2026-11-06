"""Portable immutable project, hostile archives and explicit runtime support."""

import copy
import hashlib
import json
import os
import zipfile

import pytest

from atomic_model_multiphysics.provenance import seal
from atomic_model_research.project import Project, local
from atomic_model_research.release import matrix_entry, sbom


def specification():
    return dict(
        schema_version=1,
        model="pwr-reduced-flow-controlled",
        model_config={},
        data=[],
        agent=dict(kind="declarative-json", trusted_code=False),
        cases=[
            dict(
                id="nominal",
                family="nominal",
                split="validation",
                seed=17,
                tier="training",
                steps=[dict(duration_s=0.1, commands={})],
            )
        ],
    )


def test_project_revision_and_archive_relocation(tmp_path):
    project = Project.create(tmp_path / "original", specification())
    spec, revision = project.read()
    updated = copy.deepcopy(spec)
    updated["cases"][0]["steps"][0]["commands"] = {"throttle": 0.9}
    new = project.revise(updated, expected_revision=revision)
    assert new != revision
    with pytest.raises(ValueError):
        project.revise(spec, expected_revision=revision)
    archive = tmp_path / "project.zip"
    project.export(archive)
    imported = Project.import_archive(archive, tmp_path / "relocated")
    assert imported.read() == project.read()
    assert not any(str(tmp_path) in p.read_text() for p in (tmp_path / "original").rglob("*.json"))


def test_real_core_project_run_and_replay_after_move(tmp_path):
    project = Project.create(tmp_path / "project", specification())
    first = project.run(tmp_path / "run")
    assert first["payload"]["report"]["completed"]
    project.export(tmp_path / "export.zip")
    moved = Project.import_archive(tmp_path / "export.zip", tmp_path / "moved")
    second = moved.run(tmp_path / "run-moved")
    a = first["payload"]["report"]["results"][0]["receipt"]
    b = second["payload"]["report"]["results"][0]["receipt"]
    assert a == b and a["state_schema"]


def test_data_are_content_addressed_and_checked(tmp_path):
    (tmp_path / "raw.csv").write_text("time_s,power_w\n0,1\n")
    spec = specification()
    raw = (tmp_path / "raw.csv").read_bytes()
    spec["data"] = [
        dict(
            id="observations",
            path="raw.csv",
            sha256=hashlib.sha256(raw).hexdigest(),
            license="CC0-1.0",
        )
    ]
    project = Project.create(tmp_path / "p", spec, data_root=tmp_path)
    spec, _ = project.read()
    assert spec["data"][0]["path"].startswith("data/")
    (project.root / spec["data"][0]["path"]).write_text("corrupt")
    with pytest.raises(ValueError):
        project.read()


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/../b", "a\\b", "a//b", "C:foo", ""])
def test_portable_paths_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        local(tmp_path, path)


def test_symlink_fifo_and_reserved_archive_rejected(tmp_path):
    project = Project.create(tmp_path / "p", specification())
    (project.root / "INVENTORY.json").write_text("{}")
    with pytest.raises(ValueError):
        project.export(tmp_path / "reserved.zip")
    (project.root / "INVENTORY.json").unlink()
    (project.root / "link").symlink_to("/etc/hosts")
    with pytest.raises(ValueError):
        project.export(tmp_path / "symlink.zip")
    os.mkfifo(tmp_path / "fifo")
    with pytest.raises(ValueError):
        Project.import_archive(tmp_path / "fifo", tmp_path / "fifo-import")


@pytest.mark.parametrize("member", ["../escape", "/absolute", "a/../x"])
def test_archive_traversal_rejected_and_output_removed(tmp_path, member):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(member, "{}")
        z.writestr("INVENTORY.json", json.dumps(seal({})))
    with pytest.raises(ValueError):
        Project.import_archive(archive, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_missing_revision_ancestor_refused(tmp_path):
    project = Project.create(tmp_path / "p", specification())
    spec, revision = project.read()
    project.revise(spec, expected_revision=revision)
    (project.root / "revisions" / (revision + ".json")).unlink()
    with pytest.raises(OSError):
        project.read()


def test_matrix_does_not_promote_observed_untested_runtime(monkeypatch):
    from atomic_model_research import release

    monkeypatch.setattr(
        release,
        "environment",
        lambda: dict(system="Other", machine="new", python="9", numpy="9", scipy="9"),
    )
    result = matrix_entry()
    assert not result["observed_matches_supported"]
    assert result["supported"][0]["system"] == "Linux"


def test_dependency_spdx_has_unique_packages_and_no_legal_claim():
    result = sbom()
    assert result["spdxVersion"] == "SPDX-2.3"
    identifiers = [p["SPDXID"] for p in result["packages"]]
    assert identifiers and len(identifiers) == len(set(identifiers))
    assert all(p["licenseConcluded"] == "NOASSERTION" for p in result["packages"])


def test_revision_budget_refused_before_head_change(tmp_path, monkeypatch):
    from atomic_model_research import project as module

    monkeypatch.setattr(module, "MAX_REVISIONS", 2)
    project = Project.create(tmp_path / "p", specification())
    spec, first = project.read()
    second = project.revise(spec, expected_revision=first)
    with pytest.raises(ValueError, match="budget"):
        project.revise(spec, expected_revision=second)
    assert project.read()[1] == second
