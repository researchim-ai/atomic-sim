"""Trusted replay orchestration, failure evidence and isolation from checkout."""

import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import evidence_bundle as bundles
from scripts import replay_primary_steam_bundle as replay


@pytest.fixture
def archive(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    data = {p + "/__init__.py": b"# synthetic runtime\n" for p in bundles.PACKAGES}
    with zipfile.ZipFile(source / "runtime.whl", "w") as wheel:
        for name, content in data.items():
            wheel.writestr(name, content)
        wheel.writestr("atomic_sim-2.dist-info/METADATA", "Name: atomic-sim\nVersion: 2\n")
    with tarfile.open(source / "runtime.tar.gz", "w:gz") as sdist:
        for name, content in data.items():
            member = tarfile.TarInfo("atomic_sim-2/" + name)
            member.size = len(content)
            sdist.addfile(member, io.BytesIO(content))
    fingerprint = bundles.runtime_fingerprint((source / "runtime.whl").read_bytes(), "wheel")
    for name in ("frozen_scripts", "frozen_budgets"):
        (source / name).mkdir()
        (source / name / ("helper.py" if name == "frozen_scripts" else "budget.json")).write_text(
            "{}"
        )
    (source / "steady").mkdir()
    (source / "steady/qualification.json").write_text('{"passed":true}')
    expected = dict(
        passed=True,
        signature=dict(assembly=dict(source_sha256=fingerprint)),
        integrations=14,
        case_families=7,
    )
    (source / "audit.json").write_text(json.dumps(expected))
    selection = dict(
        schema_version=1,
        bundle_id="replay-fixture",
        scope="Orchestration tests only",
        entries=[
            dict(source=name, path="release/" + name)
            for name in ("runtime.whl", "runtime.tar.gz", "frozen_scripts", "frozen_budgets")
        ]
        + [
            dict(source="steady", path="artifacts/steady"),
            dict(source="audit.json", path="audit.json"),
        ],
        runtime=dict(
            wheel="release/runtime.whl", sdist="release/runtime.tar.gz", source_sha256=fingerprint
        ),
        metadata=dict(
            primary_steam_replay=dict(
                release="release",
                steady_qualification="artifacts/steady",
                steady_audit="audit.json",
            )
        ),
    )
    path = tmp_path / "bundle.tar.gz"
    result = bundles.create_bundle(selection, source, path)
    return path, result["manifest_sha256"], expected


def test_matching_auditor_runs_outside_checkout_without_pythonpath(archive, tmp_path, monkeypatch):
    path, pin, expected = archive
    monkeypatch.setenv("PYTHONPATH", "/unrelated/checkout")
    output = tmp_path / "replay"

    def run(command, **kwargs):
        assert "PYTHONPATH" not in kwargs["env"]
        assert kwargs["cwd"].is_relative_to(output / "source")
        assert (kwargs["cwd"] / "artifacts/steady/qualification.json").exists()
        assert command[:3] == [replay.sys.executable, "-m", "scripts.qualify_primary_steam_design"]
        assert kwargs["timeout"] > 0
        Path(command[-1]).write_text(json.dumps(expected))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(replay.subprocess, "run", run)
    result = replay.replay(path, output, pin)
    assert result["passed"] and result["numerical_acceptance_recomputed"]
    assert result["profiles"][0]["archived_audit_identical"]
    assert json.loads((output / "replay.json").read_text()) == result


@pytest.mark.parametrize("failure", ["exit_code", "mismatch", "timeout"])
def test_failed_replay_keeps_diagnostics_without_acceptance(
    archive, tmp_path, monkeypatch, failure
):
    path, pin, expected = archive
    output = tmp_path / "replay"

    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "mismatch":
            Path(command[-1]).write_text(json.dumps(dict(expected, integrations=13)))
        return SimpleNamespace(returncode=1 if failure == "exit_code" else 0)

    monkeypatch.setattr(replay.subprocess, "run", run)
    with pytest.raises((RuntimeError, ValueError, subprocess.TimeoutExpired)):
        replay.replay(path, output, pin)
    result = json.loads((output / "replay.json").read_text())
    assert result["passed"] is False and result["numerical_acceptance_recomputed"] is False
    assert (output / "steady-audit.log").exists()


def test_incorrect_pin_cleans_preparation_without_running_code(archive, tmp_path, monkeypatch):
    path, _, _ = archive
    monkeypatch.setattr(
        replay.subprocess, "run", lambda *a, **k: pytest.fail("Code must not execute")
    )
    output = tmp_path / "replay"
    with pytest.raises(ValueError, match="Trusted manifest"):
        replay.replay(path, output, "0" * 64)
    assert not output.exists()


def test_missing_profile_cleans_preparation(archive, tmp_path):
    path, pin, _ = archive
    output = tmp_path / "replay"
    with pytest.raises(ValueError, match="profile missing"):
        replay.replay(path, output, pin, kinds=("transient",))
    assert not output.exists()


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), 86401])
def test_invalid_wall_limits_are_rejected_before_output(archive, tmp_path, value):
    path, pin, _ = archive
    output = tmp_path / "replay"
    with pytest.raises(ValueError, match="wall limit"):
        replay.replay(path, output, pin, wall_limit_s=value)
    assert not output.exists()


def test_existing_destination_is_not_modified(archive, tmp_path):
    path, pin, _ = archive
    output = tmp_path / "replay"
    output.mkdir()
    (output / "user-file").write_text("keep")
    with pytest.raises(FileExistsError):
        replay.replay(path, output, pin)
    assert (output / "user-file").read_text() == "keep"


def test_shared_deadline_prevents_execution_after_expensive_preparation(
    archive, tmp_path, monkeypatch
):
    path, pin, _ = archive
    times = iter((0, 2))
    monkeypatch.setattr(replay.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(replay.subprocess, "run", lambda *a, **k: pytest.fail("Deadline elapsed"))
    output = tmp_path / "replay"
    with pytest.raises(TimeoutError, match="wall limit"):
        replay.replay(path, output, pin, wall_limit_s=1)
    assert json.loads((output / "replay.json").read_text())["passed"] is False


@pytest.mark.parametrize("profile", [(), ("unknown",), ("steady", "steady")])
def test_unknown_empty_and_duplicated_profile_is_not_run(archive, tmp_path, profile):
    path, pin, _ = archive
    with pytest.raises(ValueError, match="profiles"):
        replay.replay(path, tmp_path / "replay", pin, kinds=profile)


def test_replay_paths_cannot_escape_or_replace_runtime_packages():
    base = dict(
        metadata=dict(
            primary_steam_replay=dict(
                release="release", steady_qualification="sdk/hijack", steady_audit="audit.json"
            )
        ),
        runtime={},
    )
    with pytest.raises(ValueError, match="repository-relative artifacts"):
        replay._plan(base, ("steady",))
    base["metadata"]["primary_steam_replay"]["steady_qualification"] = "../outside"
    with pytest.raises(ValueError):
        replay._plan(base, ("steady",))


def test_sdist_symlink_is_not_followed(tmp_path):
    path = tmp_path / "source.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("root/link")
        info.type, info.linkname = tarfile.SYMTYPE, "../../outside"
        archive.addfile(info)
    with pytest.raises(ValueError, match="links"):
        replay._sdist(path, tmp_path / "out", bundles.Limits())


def test_fixed_command_profile_invokes_its_frozen_auditor(archive, tmp_path, monkeypatch):
    _, _, original = archive
    source = tmp_path / "inputs"
    expected = dict(original, integrations=20, case_families=10, negative_checks=9)
    (source / "fixed-audit.json").write_text(json.dumps(expected))
    selected = dict(
        schema_version=1,
        bundle_id="fixed-replay-test",
        scope="Synthetic orchestration only",
        entries=[
            dict(source=n, path="release/" + n)
            for n in ("runtime.whl", "runtime.tar.gz", "frozen_scripts", "frozen_budgets")
        ]
        + [
            dict(source="steady", path="artifacts/fixed"),
            dict(source="fixed-audit.json", path="fixed-audit.json"),
        ],
        runtime=dict(
            wheel="release/runtime.whl",
            sdist="release/runtime.tar.gz",
            source_sha256=original["signature"]["assembly"]["source_sha256"],
        ),
        metadata=dict(
            primary_steam_replay=dict(
                release="release",
                fixed_qualification="artifacts/fixed",
                fixed_audit="fixed-audit.json",
            )
        ),
    )
    path = tmp_path / "fixed.tar.gz"
    created = bundles.create_bundle(selected, source, path)

    def run(command, **kwargs):
        assert command[2] == "scripts.qualify_fixed_primary_steam"
        assert command[3:5] == ["--audit", "artifacts/fixed"]
        Path(command[-1]).write_text(json.dumps(expected))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(replay.subprocess, "run", run)
    result = replay.replay(
        path, tmp_path / "fixed-replay", created["manifest_sha256"], kinds=("fixed",)
    )
    assert result["passed"] and result["profiles"][0]["integrations"] == 20
