"""Independent comparison semantics, full native row audit and bounded portable transport."""

import copy
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from atomic_model_results import analysis, archive, bundle
from atomic_model_study import store
from atomic_model_study.manager import Manager
from atomic_model_workspace.common import atomic_json, digest, read
from examples.eos_results_workflow import create_runs


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    output = tmp_path_factory.mktemp("results-base") / "workflow"
    folder, states = create_runs(
        Path(__file__).resolve().parents[1] / "examples/eos_reference", output
    )
    result = output / "results"
    bundle.export(folder, [s["evaluation_id"] for s in states.values()], result)
    return folder, states, result


@pytest.fixture
def retained(base, tmp_path):
    target = tmp_path / "results"
    shutil.copytree(base[2], target)
    return target


def rehash(folder, name):
    manifest = read(folder / "manifest.json")
    manifest["artifacts"][name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    atomic_json(folder / "manifest.json", manifest)


def test_two_protocols_separate_units_and_pair_ranking(base):
    folder, states, results = base
    result = bundle.verify(results)
    assert len(result["groups"]) == 2 and len(result["entries"]) == 4
    for group in result["groups"].values():
        assert len(group["evaluations"]) == 2 and group["performance_comparable"] is False
        assert len(group["channel_rankings"]) == 1
    left, right = (states[k]["evaluation_id"] for k in ["thermal-hold", "thermal-heated"])
    pair = bundle.pair(results, left, right, "wall_temperature")
    assert pair["unit"] == "K" and pair["decimated"] is False
    assert all(x <= 0 for x in pair["difference"]) and any(x < 0 for x in pair["difference"])
    assert pair["metrics"]["bias"] < 0
    assert pair["metric_units"]["mae"] == "K"
    assert pair["metric_units"]["integral_error"] == "(K) s"
    assert pair["metric_units"]["model_peak_time_s"] == "s"
    ranking = next(
        g for g in result["groups"].values() if "wall_temperature" in g["channel_rankings"]
    )["channel_rankings"]["wall_temperature"]
    assert [r["evaluation_id"] for r in ranking] == [left, right]
    assert all(row["eligible"] for row in result["entries"].values())
    assert sum(row["comparison_passed"] for row in result["entries"].values()) == 2
    thermal = [row for row in result["entries"].values() if "wall_temperature" in row["channels"]]
    assert all(row["events"][0]["status"] == "reference_event_missing" for row in thermal)
    with pytest.raises(ValueError, match="protocol"):
        bundle.pair(results, left, states["energy-hold"]["evaluation_id"], "wall_temperature")
    with pytest.raises(ValueError):
        bundle.pair(results, left, left, "wall_temperature")
    with pytest.raises(ValueError):
        bundle.pair(results, left, right, "missing")


def test_units_csv_initial_actions_and_balances(base):
    id = next(iter(base[1].values()))["evaluation_id"]
    raw = (base[2] / "evaluations" / id / "trajectory.csv").read_text()
    import csv
    import io

    rows = list(csv.DictReader(io.StringIO(raw)))
    assert {r["unit"] for r in rows} == {"K", "W", "J", "J/K", "kg", "kg m/s"}
    assert not any(r["kind"] == "action" and float(r["time_s"]) == 0 for r in rows)
    assert {"observation", "action", "global_ledger", "local_ledger"} == {r["kind"] for r in rows}


def test_deterministic_archive_roundtrip_and_replay(base, tmp_path):
    archive.pack(base[2], tmp_path / "one.zip")
    archive.pack(base[2], tmp_path / "two.zip")
    assert (tmp_path / "one.zip").read_bytes() == (tmp_path / "two.zip").read_bytes()
    result = archive.unpack(tmp_path / "one.zip", tmp_path / "imported")
    assert result["replay_exact"]
    assert bundle.verify(tmp_path / "imported") == bundle.verify(base[2])
    assert bundle.replay(tmp_path / "imported", tmp_path / "replayed")["replay_exact"]


@pytest.mark.parametrize("ids", [[], ["bad"], ["0" * 32] * 2, ["0" * 32] * 33, "not-a-list"])
def test_selection_refused_without_output(base, tmp_path, ids):
    with pytest.raises(ValueError):
        bundle.export(base[0], ids, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "kind",
    ["summary", "trajectory", "report", "native_row", "source", "missing", "extra", "symlink"],
)
def test_tampered_bundle_refused_even_after_manifest_rehash(retained, kind):
    spec = read(retained / "bundle.json")
    id = spec["evaluations"][0]
    prefix = "evaluations/" + id + "/"
    if kind == "summary":
        name = "summary.json"
        value = read(retained / name)
        value["entries"][id]["comparison_passed"] = "fake"
        atomic_json(retained / name, value)
        rehash(retained, name)
    elif kind == "trajectory":
        name = prefix + "trajectory.csv"
        (retained / name).write_text(
            "time_s,kind,channel,value,unit\n0,observation,wall.w0,999,K\n"
        )
        rehash(retained, name)
    elif kind == "report":
        state = read(retained / (prefix + "status.json"))
        name = prefix + state["report"] + "/report.json"
        value = read(retained / name)
        value["passed"] = not value["passed"]
        atomic_json(retained / name, value)
        rehash(retained, name)
    elif kind == "native_row":
        name = prefix + "job/revisions/snapshot.json"
        value = read(retained / name)
        job = value["payload"]
        job["rows"][1]["observation"]["channels"][0]["value"] -= 1
        job["rows"][2]["policy_input"] = copy.deepcopy(job["rows"][1]["observation"])
        value["sha256"] = digest(job)
        atomic_json(retained / name, value)
        rehash(retained, name)
        head_name = prefix + "job/HEAD.json"
        head = read(retained / head_name)
        head["sha256"] = value["sha256"]
        atomic_json(retained / head_name, head)
        rehash(retained, head_name)
    elif kind == "source":
        name = "manifest.json"
        value = read(retained / name)
        value["source"]["results_sha256"] = "0" * 64
        atomic_json(retained / name, value)
    elif kind == "missing":
        (retained / (prefix + "input.json")).unlink()
    elif kind == "extra":
        (retained / "unexpected.py").write_text('raise AssertionError("must not execute")')
    else:
        original = retained / (prefix + "trajectory.csv")
        original.rename(retained / "linked.csv")
        original.symlink_to(retained / "linked.csv")
    with pytest.raises((ValueError, OSError)):
        bundle.verify(retained)


def test_cancelled_prefix_not_ranked_and_stale_snapshot(base, tmp_path):
    folder = tmp_path / "study"
    shutil.copytree(base[0], folder)
    with Manager(folder) as manager:
        v = store.inspect(folder)
        state = manager.start(
            "energy-hold",
            expected_revision=v["revision"],
            expected_checkpoint_sha256=v["checkpoint_sha256"],
        )
        manager.cancel(state["evaluation_id"])
    v = store.inspect(folder)
    scenario = copy.deepcopy(v["project"]["scenarios"]["uniform_dev"])
    scenario["objective"]["target_k"] = 301
    store.edit(
        folder,
        dict(
            schema_version=1,
            expected_revision=v["revision"],
            expected_project_sha256=v["project_sha256"],
            operations=[dict(op="replace_scenario", id="uniform_dev", value=scenario)],
            note="new objective",
        ),
    )
    ids = [state["evaluation_id"], next(iter(base[1].values()))["evaluation_id"]]
    bundle.export(folder, ids, tmp_path / "retained")
    summary = bundle.verify(tmp_path / "retained")
    row = summary["entries"][state["evaluation_id"]]
    assert row["eligible"] is False and row["comparison_passed"] is None
    assert all(e["stale"] for e in summary["entries"].values())
    assert sum(len(g["evaluations"]) for g in summary["groups"].values()) == 1


def test_active_worker_guard_and_existing_output(base, tmp_path):
    folder = tmp_path / "study"
    shutil.copytree(base[0], folder)
    (folder / "EVALUATION.lock").write_text("ownership-marker")
    ids = [next(iter(base[1].values()))["evaluation_id"]]
    with pytest.raises(ValueError, match="worker"):
        bundle.export(folder, ids, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    with pytest.raises(FileExistsError):
        bundle.export(base[0], ids, base[2])


@pytest.mark.parametrize(
    "kind",
    [
        "traversal",
        "absolute",
        "duplicate",
        "symlink",
        "directory",
        "missing_manifest",
        "oversize",
        "count",
    ],
)
def test_zip_admission_before_creating_output(tmp_path, kind, monkeypatch):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        if kind == "traversal":
            z.writestr("../outside", "bad")
        elif kind == "absolute":
            z.writestr("/tmp/outside", "bad")
        elif kind == "duplicate":
            z.writestr("same", "1")
            with pytest.warns(UserWarning, match="Duplicate name"):
                z.writestr("same", "2")
        elif kind == "symlink":
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, "/etc/passwd")
        elif kind == "directory":
            z.writestr("directory/", "")
        elif kind == "missing_manifest":
            z.writestr("file", "data")
        elif kind == "oversize":
            z.writestr("manifest.json", "xx")
            monkeypatch.setattr(archive, "MAX_FILE", 1)
        else:
            z.writestr("manifest.json", "x")
            z.writestr("one", "x")
            z.writestr("two", "x")
            monkeypatch.setattr(archive, "MAX_FILES", 1)
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        archive.unpack(path, tmp_path / "bad")
    assert not (tmp_path / "bad").exists() and not (tmp_path.parent / "outside").exists()


def test_protocol_separates_seed_objective_limits_and_uncertainty(base):
    spec = read(base[2] / "bundle.json")
    runs = bundle._runs(base[2], spec)
    run = next(iter(runs.values()))
    original = analysis.protocol(run[0], run[4])
    for defect in ["seed", "objective", "budget", "uncertainty"]:
        p, ref = copy.deepcopy(run[0]), copy.deepcopy(run[4])
        b = p["study"]["references"][p["binding_id"]]
        s = p["study"]["project"]["scenarios"][b["scenario_id"]]
        if defect == "seed":
            s["seed"] += 1
        elif defect == "objective":
            s["objective"]["target_k"] += 1
        elif defect == "budget":
            p["rss_limit_bytes"] += 1024
        else:
            ref["uncertainty"][next(iter(ref["uncertainty"]))] = [0.1] * len(ref["times_s"])
        assert analysis.protocol(p, ref) != original


def test_cli_verify_and_pair(base):
    result = subprocess.run(
        [sys.executable, "-m", "atomic_model_results", "verify", str(base[2])],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["groups"]) == 2


@pytest.mark.parametrize("defect", ["project_identity", "revision_type"])
def test_current_study_metadata_is_admitted_not_just_self_hashed(retained, defect):
    spec = read(retained / "bundle.json")
    if defect == "project_identity":
        spec["study"]["project_sha256"] = "0" * 64
    else:
        spec["study"]["revision"] = True
    spec["study_checkpoint_sha256"] = digest(spec["study"])
    atomic_json(retained / "bundle.json", spec)
    rehash(retained, "bundle.json")
    with pytest.raises(ValueError, match="Study schema"):
        bundle.verify(retained)
