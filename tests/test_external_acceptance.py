"""Local source-bound replay and experimental-data admission."""

import copy

import pytest

from atomic_model_acceptance.candidates import admit, motel_candidate
from atomic_model_acceptance.replay import compare, make_request
from atomic_model_multiphysics.provenance import seal
from atomic_model_research.project import Project


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


@pytest.fixture
def pair(tmp_path):
    request = make_request(specification(), tmp_path / "request")
    project = Project.import_archive(tmp_path / "request/project.zip", tmp_path / "relocated")
    receipt = project.run(tmp_path / "run")
    return seal(request), receipt


def test_real_core_replay_without_external_promotion(pair):
    result = compare(*pair)
    assert result["numerical_replay_accepted"]
    assert result["verification_scope"] == "local numerical replay"
    assert not result["ci_required"] and not result["remote_execution_required"]
    assert not result["validated"]


@pytest.mark.parametrize("field", ["state_schema", "case_sha256", "source_sha256"])
def test_case_unit_or_source_mismatch_refused(pair, field):
    request, receipt = copy.deepcopy(pair)
    receipt["payload"]["report"]["results"][0]["receipt"][field] = "wrong"
    with pytest.raises(ValueError):
        compare(request, seal(receipt["payload"]))


def test_state_difference_and_balance_exceedance_not_accepted(pair):
    request, receipt = copy.deepcopy(pair)
    result = receipt["payload"]["report"]["results"][0]["receipt"]
    result["state"][0] += 1
    assert not compare(request, seal(receipt["payload"]))["numerical_replay_accepted"]
    request, receipt = copy.deepcopy(pair)
    receipt["payload"]["report"]["results"][0]["receipt"]["balances"]["energy_relative"] = 1.0
    assert not compare(request, seal(receipt["payload"]))["numerical_replay_accepted"]


def test_request_and_history_tampering_refused(pair):
    request, receipt = copy.deepcopy(pair)
    request["payload"]["protocol"]["state_scaled_error_max"] = 1.0
    with pytest.raises(ValueError):
        compare(seal(request["payload"]), receipt)
    request, receipt = copy.deepcopy(pair)
    receipt["payload"]["report"]["results"][0]["receipt"]["checkpoint"]["payload"]["revision"] = 999
    with pytest.raises(ValueError):
        compare(request, seal(receipt["payload"]))


def test_article_license_never_inferred_as_raw_data_license():
    candidate = motel_candidate()
    result = admit(candidate)
    assert candidate["publication"]["license"] == "CC-BY-4.0"
    assert not result["admitted"]
    assert "raw_measurements_unavailable" in result["reasons"]
    assert "raw_data_rights_unconfirmed" in result["reasons"]
    assert not result["validated"]


def test_candidate_calibration_leakage_and_string_flags_refused():
    candidate = motel_candidate()
    candidate.update(calibration_case="same", validation_cases=["same"])
    with pytest.raises(ValueError):
        admit(candidate)
    candidate = motel_candidate()
    candidate["raw_data"]["available"] = "true"
    with pytest.raises(ValueError):
        admit(candidate)


def test_packet_project_checksum_checked_before_replay(tmp_path):
    from atomic_model_acceptance.replay import verify_packet

    make_request(specification(), tmp_path / "packet")
    assert verify_packet(tmp_path / "packet")["project_archive_sha256"]
    path = tmp_path / "packet/project.zip"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_packet(tmp_path / "packet")


def test_failed_replay_cli_returns_failure(tmp_path, pair, monkeypatch):
    from atomic_model_acceptance.__main__ import main
    from atomic_model_training.spec import atomic_json

    request, receipt = copy.deepcopy(pair)
    receipt["payload"]["report"]["results"][0]["receipt"]["state"][0] += 1
    atomic_json(tmp_path / "request.json", request)
    atomic_json(tmp_path / "receipt.json", seal(receipt["payload"]))
    monkeypatch.setattr(
        "sys.argv",
        [
            "atomic-acceptance",
            "compare",
            str(tmp_path / "request.json"),
            str(tmp_path / "receipt.json"),
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    assert main() == 2


@pytest.mark.parametrize(
    "defect", [None, "stale", "missing", "bytes", "duplicate", "cache", "extra"]
)
def test_local_wheel_checks_every_source_package(tmp_path, defect):
    import warnings
    import zipfile

    from scripts.verify_local_library import check_wheel

    source = tmp_path / "src"
    (source / "first").mkdir(parents=True)
    (source / "second").mkdir()
    files = {
        "first/__init__.py": b"# first",
        "second/preset.json": b"{}",
        "standalone.py": b"# module",
    }
    for name, raw in files.items():
        (source / name).write_bytes(raw)
    (source / "example.egg-info").mkdir()
    (source / "example.egg-info" / "PKG-INFO").write_text("generated build metadata")
    archive = tmp_path / "source.whl"
    with zipfile.ZipFile(archive, "w") as z:
        for name, raw in files.items():
            if defect == "missing" and name == "second/preset.json":
                continue
            z.writestr(name, b"changed" if defect == "bytes" and name == "standalone.py" else raw)
        if defect == "stale":
            z.writestr("second/deleted.py", "# stale")
        if defect == "duplicate":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                z.writestr("first/__init__.py", "# duplicate")
        if defect == "cache":
            z.writestr("first/__pycache__/old.pyc", b"cache")
        if defect == "extra":
            z.writestr("removed_package/__init__.py", "# stale package")
    if defect is None:
        assert check_wheel(archive, source_root=source)["files"] == 3
    else:
        with pytest.raises(ValueError):
            check_wheel(archive, source_root=source)


def test_local_command_can_repeat_without_overwriting_previous_results(monkeypatch):
    from scripts import verify_local_library as local

    outputs = []
    monkeypatch.setattr(local, "verify", lambda directory, **kwargs: outputs.append(directory))
    assert local.main([]) == 0 and local.main([]) == 0
    assert len(set(outputs)) == 2
    assert all(path.startswith("artifacts/local-verification-") for path in outputs)


def test_local_command_preserves_explicit_output_and_suite(monkeypatch):
    from scripts import verify_local_library as local

    calls = []
    monkeypatch.setattr(
        local, "verify", lambda directory, **kwargs: calls.append((directory, kwargs))
    )
    assert local.main(["--output", "chosen-output", "--suite", "portable"]) == 0
    assert calls == [("chosen-output", dict(suite="portable", phase_timeout_s=1800))]


@pytest.mark.parametrize("defect", [None, "missing", "bytes", "stale"])
def test_wheel_root_mapped_resources_match_exactly(tmp_path, defect):
    import zipfile

    from scripts.verify_local_library import check_wheel

    source = tmp_path / "src"
    source.mkdir()
    (source / "module.py").write_bytes(b"# module")
    resources = tmp_path / "validation"
    resources.mkdir()
    (resources / "__init__.py").write_bytes(b"")
    # Frozen data bytes, including CRLF, must survive unchanged.
    (resources / "measurements.csv").write_bytes(b"x,y\r\n1,2\r\n")
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("module.py", b"# module")
        archive.writestr("validation/__init__.py", b"")
        if defect != "missing":
            archive.writestr(
                "validation/measurements.csv",
                b"x,y\n1,2\n" if defect == "bytes" else b"x,y\r\n1,2\r\n",
            )
        if defect == "stale":
            archive.writestr("validation/removed.json", "{}")
    if defect:
        with pytest.raises(ValueError):
            check_wheel(wheel, source_root=source, resource_roots={"validation": resources})
    else:
        result = check_wheel(wheel, source_root=source, resource_roots={"validation": resources})
        assert result["files"] == 3
        assert result["package_counts"] == {"src": 1, "validation": 2}
