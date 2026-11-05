"""Saved hybrid workflows: real integration, continuation and recomputation."""

from copy import deepcopy
from pathlib import Path

import pytest

from atomic_model_workspace.common import atomic_json, read
from core.network import NetworkError
from scripts.hybrid_graph_cli import main, replay, run
from scripts.hybrid_graph_references import cases
from scripts.solve_hydraulic_graph import sha


@pytest.fixture
def spec():
    _, model, program = next(c for c in cases() if c[0] == "explicit_online_state")
    return dict(
        schema_version=1,
        model=model,
        max_macro_step_s=1.0,
        program=program[:3],
        initial_checkpoint=None,
    )


@pytest.fixture
def output(spec, tmp_path):
    folder = tmp_path / "partial"
    run(spec, folder)
    return folder


def test_run_replay_continue_same_checkpoint_as_unsplit(spec, output, tmp_path):
    partial = replay(output)
    assert partial["observations"][-1]["controller_state"] == {"integral": 0.5, "tick": 1}
    _, _, program = next(c for c in cases() if c[0] == "explicit_online_state")
    continued = deepcopy(spec)
    continued["initial_checkpoint"] = partial["final_checkpoint"]
    continued["program"] = program[3:]
    result = run(continued, tmp_path / "continued")
    assert replay(tmp_path / "continued") == result
    whole = deepcopy(spec)
    whole["program"] = program
    assert run(whole, tmp_path / "control")["final_checkpoint"] == result["final_checkpoint"]


def test_rehashed_saved_numeric_report_fails(output):
    result = read(output / "result.json")
    result["observations"][-1]["engine"]["state"]["temperatures_k"]["a"] += 1
    atomic_json(output / "result.json", result)
    manifest = read(output / "manifest.json")
    manifest["artifacts"]["result.json"] = sha(output / "result.json")
    atomic_json(output / "manifest.json", manifest)
    with pytest.raises(NetworkError, match="full replay"):
        replay(output)


@pytest.mark.parametrize("kind", ["extra", "symlink", "source", "hash"])
def test_invalid_artifact_refused(output, tmp_path, kind):
    if kind == "extra":
        (output / "extra").write_text("x")
    elif kind == "symlink":
        original = tmp_path / "original"
        (output / "result.json").rename(original)
        (output / "result.json").symlink_to(original)
    else:
        m = read(output / "manifest.json")
        if kind == "source":
            m["source"]["cli_sha256"] = "0" * 64
        else:
            m["artifacts"]["result.json"] = "0" * 64
        atomic_json(output / "manifest.json", m)
    with pytest.raises((NetworkError, ValueError)):
        replay(output)


def test_failed_program_never_publishes(spec, tmp_path):
    spec["program"][1]["command"]["expected_revision"] = 99
    with pytest.raises(NetworkError):
        run(spec, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


def test_cli_continue_and_replay(output, tmp_path):
    _, _, program = next(c for c in cases() if c[0] == "explicit_online_state")
    atomic_json(tmp_path / "program.json", program[3:])
    assert (
        main(
            [
                "continue",
                str(output),
                "--program",
                str(tmp_path / "program.json"),
                "--output",
                str(tmp_path / "next"),
            ]
        )
        == 0
    )
    main(["replay", str(tmp_path / "next")])
    assert replay(tmp_path / "next")["observations"][-1]["time_s"] == 40.0


def test_output_not_overwritten(spec, output):
    before = (output / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(spec, output)
    assert (output / "result.json").read_bytes() == before


def test_cli_outside_resource_root_uses_child_without_changing_parent_cwd(
    spec, tmp_path, monkeypatch
):
    atomic_json(tmp_path / "input.json", spec)
    monkeypatch.chdir(tmp_path)
    assert main(["run", "input.json", "--output", "result"]) == 0
    assert main(["replay", "result"]) == 0
    assert Path.cwd() == tmp_path
    assert read(tmp_path / "result/result.json")["observations"][-1]["time_s"] == 17.0
