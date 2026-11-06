"""Transactional schema edits, CAS, source guards and portable export."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from atomic_model_projects import store
from atomic_model_workspace import common
from atomic_model_workspace.common import Lock, atomic_json, digest, read
from atomic_model_workspace.project import load_project


@pytest.fixture
def project(tmp_path):
    root = Path(__file__).resolve().parents[1] / "examples/eos_workspace"
    shutil.copytree(root, tmp_path / "project")
    return tmp_path / "project/project.json"


@pytest.fixture
def editor(project, tmp_path):
    folder = tmp_path / "store"
    store.create(project, folder)
    return folder


def request(folder, operations=None):
    view = store.inspect(folder)
    return dict(
        schema_version=1,
        expected_revision=view["revision"],
        expected_project_sha256=view["project_sha256"],
        operations=operations
        or [dict(op="set_scenario_config", id="mixed_dev", value=dict(episode_horizon_s=0.04))],
        note="changed horizon",
    )


def test_import_inspect_preview_and_commit_export_roundtrip(project, editor, tmp_path):
    view = store.inspect(editor)
    assert view["revision"] == 0 and view["project"] == load_project(project)
    assert (
        view["model"]["validated"] is False
        and view["channels"]["mixed_dev"]["actions"]["wall.w0"] == "W"
    )
    before = (editor / "HEAD.json").read_bytes()
    req = request(editor)
    preview = store.preview(editor, req)
    assert not preview["committed"] and before == (editor / "HEAD.json").read_bytes()
    assert preview["changes"] == [
        dict(
            path=["scenarios", "mixed_dev", "config", "episode_horizon_s"],
            before_present=True,
            after_present=True,
            before=0.03,
            after=0.04,
        )
    ]
    result = store.edit(editor, req)
    assert result["revision"]["payload"]["project_sha256"] == preview["proposed_project_sha256"]
    assert store.inspect(editor)["revision"] == 1
    exported = store.export(editor, tmp_path / "exported")
    assert load_project(exported["project"]) == preview["project"]
    manifest = read(tmp_path / "exported/project.json")
    assert all(
        not Path(ref).is_absolute()
        for group in ["scenarios", "policies"]
        for ref in manifest[group].values()
    )
    with pytest.raises(FileExistsError):
        store.export(editor, tmp_path / "exported")


@pytest.mark.parametrize(
    "defect",
    [
        "schema",
        "unknown",
        "stale_revision",
        "stale_hash",
        "bad_kind",
        "unknown_kind",
        "unknown_id",
        "bad_config",
        "topology",
        "policy_mask",
        "empty_project",
        "duplicate_object",
        "noop",
        "bad_note",
    ],
)
def test_rejected_transaction_keeps_head_and_history(editor, defect):
    req = request(editor)
    before = (editor / "HEAD.json").read_bytes()
    files = sorted((editor / "revisions").iterdir())
    if defect == "schema":
        req["schema_version"] = True
    elif defect == "unknown":
        req["unknown"] = 1
    elif defect == "stale_revision":
        req["expected_revision"] = True
    elif defect == "stale_hash":
        req["expected_project_sha256"] = "0" * 64
    elif defect == "bad_kind":
        req["operations"][0]["op"] = []
    elif defect == "unknown_kind":
        req["operations"][0]["op"] = "run_python"
    elif defect == "unknown_id":
        req["operations"][0]["id"] = "missing"
    elif defect == "bad_config":
        req["operations"][0]["value"]["max_step_s"] = 1
    elif defect == "topology":
        req["operations"][0]["value"]["preset"] = "uniform"
    elif defect == "policy_mask":
        req["operations"] = [
            dict(
                op="replace_policy",
                id="hold",
                value=dict(schema_version=1, kind="hold-v1", heat_power_w={"wall.w0": 1}),
            )
        ]
    elif defect == "empty_project":
        req["operations"] = [
            dict(op="remove_scenario", id=name)
            for name in store.inspect(editor)["project"]["scenarios"]
        ]
    elif defect == "duplicate_object":
        req["operations"].append(
            dict(
                op="set_objective",
                id="mixed_dev",
                value=dict(sensor="wall.w0", target_k=301, tolerance_k=0.1),
            )
        )
    elif defect == "noop":
        req["operations"][0]["value"]["episode_horizon_s"] = 0.03
    elif defect == "bad_note":
        req["note"] = "X" * 513
    with pytest.raises(ValueError):
        store.edit(editor, req)
    assert (editor / "HEAD.json").read_bytes() == before and sorted(
        (editor / "revisions").iterdir()
    ) == files


def test_entire_batch_admitted_before_any_commit(editor):
    req = request(editor)
    req["operations"].append(
        dict(
            op="set_objective",
            id="gravity_test",
            value=dict(sensor="hidden", target_k=300, tolerance_k=1),
        )
    )
    with pytest.raises(ValueError):
        store.edit(editor, req)
    assert store.inspect(editor)["revision"] == 0


def test_stale_preview_rejected_after_another_writer_commits(editor):
    old = request(editor)
    store.preview(editor, old)
    store.edit(editor, old)
    with pytest.raises(ValueError, match="Stale"):
        store.edit(editor, old)
    assert store.inspect(editor)["revision"] == 1


def test_add_remove_and_restore_append_a_new_revision(editor):
    original = store.inspect(editor)
    scenario = original["project"]["scenarios"]["mixed_dev"]
    req = request(
        editor,
        [
            dict(op="add_scenario", id="new_case", value=scenario),
            dict(op="remove_policy", id="hold"),
        ],
    )
    store.edit(editor, req)
    current = store.inspect(editor)
    assert (
        "new_case" in current["project"]["scenarios"]
        and "hold" not in current["project"]["policies"]
    )
    restored = store.restore(
        editor,
        expected_revision=1,
        expected_project_sha256=current["project_sha256"],
        target_revision=0,
    )
    assert (
        restored["payload"]["revision"] == 2
        and restored["payload"]["project"] == original["project"]
    )
    assert [r["revision"] for r in store.inspect(editor)["history"]] == [2, 1, 0]


def test_history_corruption_detected_even_when_head_is_intact(editor):
    original, _ = store.load(editor)
    store.edit(editor, request(editor))
    path = editor / "revisions" / f"{original['sha256']}.json"
    value = read(path)
    value["payload"]["note"] = "altered"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="checksum"):
        store.load(editor)


def test_failed_head_publication_retains_exact_old_project_and_allows_retry(editor, monkeypatch):
    before, _ = store.load(editor)
    req = request(editor)
    original = common.os.replace

    def fail(src, dst):
        if Path(dst).name == "HEAD.json":
            raise OSError("injected publication fault")
        return original(src, dst)

    with monkeypatch.context() as m:
        m.setattr(common.os, "replace", fail)
        with pytest.raises(OSError):
            store.edit(editor, req)
    assert store.load(editor)[0] == before
    assert len(list((editor / "revisions").glob("*.json"))) == 2
    store.edit(editor, req)
    assert store.inspect(editor)["revision"] == 1


def test_writer_lock_and_source_drift_block_edits(editor, monkeypatch):
    req = request(editor)
    with Lock(editor):
        with pytest.raises(ValueError):
            store.edit(editor, req)
    original = store.source_identity

    def drift():
        value = original()
        value["editor_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(store, "source_identity", drift)
    with pytest.raises(ValueError, match="source changed"):
        store.load(editor)


def test_revision_budget_is_enforced_before_commit(editor, monkeypatch):
    store.edit(editor, request(editor))
    monkeypatch.setattr(store, "MAX_REVISIONS", 2)
    req = request(
        editor, [dict(op="set_scenario_config", id="mixed_dev", value=dict(episode_horizon_s=0.05))]
    )
    with pytest.raises(ValueError, match="budget"):
        store.edit(editor, req)
    assert store.inspect(editor)["revision"] == 1


def test_cli_project_edit_and_export(project, tmp_path):
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [
                str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1]),
            ]
        ),
    )

    def cli(*args):
        p = subprocess.run(
            [sys.executable, "-m", "atomic_model_projects", *map(str, args)],
            env=env,
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert p.returncode == 0, p.stderr
        return json.loads(p.stdout)

    folder = tmp_path / "cli-store"
    cli("create", project, "--output", folder)
    req = request(folder)
    path = tmp_path / "request.json"
    atomic_json(path, req)
    assert not cli("preview", folder, path)["committed"]
    cli("edit", folder, path)
    assert cli("inspect", folder)["revision"] == 1
    output = cli("export", folder, "--output", tmp_path / "cli-export")
    assert (
        load_project(output["project"])["scenarios"]["mixed_dev"]["config"]["episode_horizon_s"]
        == 0.04
    )


@pytest.mark.parametrize("defect", ["schema", "path", "checksum", "operation"])
def test_unknown_or_corrupt_store_contract_rejected(editor, defect):
    head = read(editor / "HEAD.json")
    if defect == "schema":
        head["schema_version"] = 2
    elif defect == "path":
        head["checkpoint"] = "../outside.json"
    elif defect == "checksum":
        head["sha256"] = "0" * 64
    else:
        value = read(editor / head["checkpoint"])
        value["payload"]["operation"] = "unknown-v2"
        value["sha256"] = digest(value["payload"])
        target = editor / "revisions" / f"{value['sha256']}.json"
        atomic_json(target, value)
        head.update(checkpoint="revisions/" + value["sha256"] + ".json", sha256=value["sha256"])
    atomic_json(editor / "HEAD.json", head)
    with pytest.raises(ValueError):
        store.load(editor)
