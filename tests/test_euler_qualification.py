"""Admission must reject unverified, modified and out-of-scope requests."""

import hashlib
import json

import pytest

from core.status import DomainError
from sdk.euler_qualification import EulerQualification
from sdk.experiment import _source_fingerprint


def evidence(folder, *, accepted=True, source=None):
    run = folder / "smooth_positive_64"
    run.mkdir()
    spec = dict(config={"cells": 64}, duration_s=0.3, solver={"rtol": 1e-9})
    (run / "input.json").write_text(json.dumps(spec))
    matrix = dict(
        source_sha256=source or _source_fingerprint(),
        entries=[
            dict(id="smooth_positive", accepted=accepted, qualified_cells=[64], paths=[str(run)])
        ],
    )
    (folder / "qualification.json").write_text(json.dumps(matrix))
    hashes = {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob("*.json")
    }
    (folder / "manifest.json").write_text(json.dumps(dict(sha256=hashes)))
    return spec


def test_gate_admits_only_exact_fixture_and_returns_independent_copy(tmp_path):
    spec = evidence(tmp_path)
    gate = EulerQualification(tmp_path)
    assert gate.verify_request("smooth_positive", 64, **spec)
    changed = gate.approved_input("smooth_positive", 64)
    changed["config"]["cells"] = 32
    assert gate.approved_input("smooth_positive", 64) == spec
    with pytest.raises(DomainError):
        gate.verify_request("smooth_positive", 64, **changed)
    with pytest.raises(DomainError):
        gate.verify_request("smooth_positive", 64, **dict(spec, duration_s=0.4))
    with pytest.raises(DomainError):
        gate.approved_input("smooth_positive", 32)
    with pytest.raises(DomainError):
        gate.approved_input("unknown", 64)


def test_rejected_case_and_other_runtime_cannot_be_admitted(tmp_path):
    evidence(tmp_path, accepted=False)
    with pytest.raises(DomainError):
        EulerQualification(tmp_path).approved_input("smooth_positive", 64)
    other = tmp_path / "other"
    other.mkdir()
    evidence(other, source="different")
    with pytest.raises(DomainError):
        EulerQualification(other)


@pytest.mark.parametrize("when", ["before", "after"])
def test_modified_input_rejected_even_after_gate_created(tmp_path, when):
    evidence(tmp_path)
    gate = EulerQualification(tmp_path) if when == "after" else None
    (tmp_path / "smooth_positive_64/input.json").write_text("{}")
    with pytest.raises(ValueError):
        if gate is None:
            EulerQualification(tmp_path)
        else:
            gate.approved_input("smooth_positive", 64)


@pytest.mark.parametrize("key", ["qualification.json", "smooth_positive_64/input.json"])
def test_unlisted_evidence_not_trusted(tmp_path, key):
    evidence(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["sha256"][key]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        EulerQualification(tmp_path).approved_input("smooth_positive", 64)


def test_manifest_cannot_escape_evidence_directory(tmp_path):
    evidence(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["sha256"]["../outside"] = "arbitrary"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        EulerQualification(tmp_path)
