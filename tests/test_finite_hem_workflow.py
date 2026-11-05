"""Self-contained finite-system runs, replay, installed-style CLI and refusal paths."""

import shutil
from pathlib import Path

import pytest

from atomic_model_hem_system import workflow
from atomic_model_hem_system.__main__ import main
from atomic_model_hem_system.cases import baseline
from atomic_model_hem_system.model import FiniteHEMSystem
from atomic_model_workspace.common import atomic_json, read


@pytest.fixture(scope="module")
def demonstration(tmp_path_factory):
    folder = tmp_path_factory.mktemp("finite-system") / "demo"
    workflow.demo(folder)
    return folder


@pytest.fixture
def retained(demonstration, tmp_path):
    folder = tmp_path / "demo"
    shutil.copytree(demonstration, folder)
    return folder


def test_saved_coupled_system_full_replay_and_continuation(demonstration):
    result = workflow.replay(demonstration)
    assert any(c["phase"] == "vapor" for c in result["observations"][-1]["cells"])
    spec = read(demonstration / "input.json")
    model = FiniteHEMSystem(spec["model"])
    restored = model.restore(result["checkpoint"])
    assert restored["time_s"] == pytest.approx(0.018)
    assert restored["global_ledger"] == result["observations"][-1]["global_ledger"]
    assert model.advance(0.001)["time_s"] == pytest.approx(0.019)


def test_modified_physics_report_refused_by_real_recomputation(retained):
    result = read(retained / "result.json")
    result["observations"][0]["cells"][0]["pressure_pa"] += 1
    atomic_json(retained / "result.json", result)
    with pytest.raises(ValueError, match="replay differs"):
        workflow.replay(retained)


@pytest.mark.parametrize("bad", ["source", "extra", "symlink"])
def test_artifact_source_inventory_or_links_refused(retained, tmp_path, bad):
    if bad == "source":
        result = read(retained / "result.json")
        result["source"] = {"bad": True}
        atomic_json(retained / "result.json", result)
    elif bad == "extra":
        (retained / "extra").write_text("x")
    else:
        target = tmp_path / "input.json"
        (retained / "input.json").rename(target)
        (retained / "input.json").symlink_to(target)
    with pytest.raises(ValueError):
        workflow.replay(retained)


def test_failed_late_command_publishes_no_output(tmp_path):
    spec = dict(
        schema_version=1,
        model=baseline(),
        program=[
            dict(duration_s=0.001, command=None),
            dict(
                duration_s=0.001,
                command=dict(controls={}, expected_revision=0, controller_state={}),
            ),
        ],
    )
    with pytest.raises(ValueError):
        workflow.run(spec, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


def test_cli_run_and_replay_from_any_working_directory(tmp_path, monkeypatch):
    spec = dict(schema_version=1, model=baseline(), program=[dict(duration_s=0.001, command=None)])
    atomic_json(tmp_path / "input.json", spec)
    monkeypatch.chdir(tmp_path)
    assert main(["run", "input.json", "--output", "result"]) == 0
    assert main(["replay", "result"]) == 0
    assert Path.cwd() == tmp_path


def test_existing_output_preserved(demonstration):
    before = (demonstration / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        workflow.demo(demonstration)
    assert (demonstration / "result.json").read_bytes() == before


def test_audit_refuses_wrong_source_before_recomputing(tmp_path):
    atomic_json(tmp_path / "campaign.json", dict(source={}))
    with pytest.raises(ValueError, match="source mismatch"):
        workflow.audit(tmp_path / "campaign.json")
