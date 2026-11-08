"""Physics profile must reject empty, skipped and incomplete evidence."""

import copy

import pytest
from scripts.verify_physics import LANES, ROOT, receipt_passed


def receipt():
    return dict(
        counts=dict(collected=2, passed=2, failed=0, errors=0, skipped=0),
        exit_code=0,
        gate_failures=[],
    )


def test_success_requires_all_collected_tests_to_pass():
    assert receipt_passed(receipt(), 0)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(exit_code=1),
        lambda r: r.update(gate_failures=["no tests executed"]),
        lambda r: r["counts"].update(collected=0, passed=0),
        lambda r: r["counts"].update(skipped=1),
        lambda r: r["counts"].update(errors=1),
        lambda r: r["counts"].update(failed=1),
        lambda r: r["counts"].update(passed=1),
        lambda r: r["counts"].update(passed=True),
        lambda r: r["counts"].pop("collected"),
        lambda r: r.update(counts=None),
    ],
)
def test_incomplete_or_malformed_receipt_fails(mutation):
    value = copy.deepcopy(receipt())
    mutation(value)
    assert not receipt_passed(value, 0)


@pytest.mark.parametrize("code", [1, 2, 124, -9, False])
def test_process_failure_cannot_be_hidden_by_success_receipt(code):
    assert not receipt_passed(receipt(), code)


def test_profile_has_explicit_scope_and_existing_test_files():
    assert set(LANES) == {
        "eos-phase",
        "balances",
        "convergence",
        "restart",
        "neutronics",
        "controlled-pwr",
    }
    for lane in LANES.values():
        assert lane["scope"] and lane["tests"]
        for node in lane["tests"]:
            assert (ROOT / node.split("::")[0]).is_file()


@pytest.mark.parametrize("failure", ["skip", "source-drift", "timeout"])
def test_profile_preserves_failure_and_continues_remaining_groups(tmp_path, monkeypatch, failure):
    import subprocess
    from pathlib import Path
    from types import SimpleNamespace

    from scripts import verify_physics as profile

    from atomic_model_multiphysics.provenance import unseal
    from atomic_model_training.spec import atomic_json, read_json

    monkeypatch.setattr(
        profile,
        "LANES",
        {
            "first": {"scope": "first", "tests": ["test_a.py"]},
            "second": {"scope": "second", "tests": ["test_b.py"]},
        },
    )
    manifests = iter(
        [{"a": "original"}, {"a": "changed" if failure == "source-drift" else "original"}]
    )
    monkeypatch.setattr(profile, "source_manifest", lambda: next(manifests))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1 and failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        value = receipt()
        if len(calls) == 1 and failure == "skip":
            value["counts"].update(passed=1, skipped=1)
        target = Path(command[command.index("--atomic-receipt") + 1])
        atomic_json(target, value)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(profile.subprocess, "run", run)
    output = tmp_path / "result"
    report = profile.verify(output)
    assert len(calls) == 2
    assert report["completed"] and not report["passed"]
    assert report["lanes"][1]["passed"]
    if failure == "source-drift":
        assert not report["source_unchanged"]
    else:
        assert not report["lanes"][0]["passed"]
    assert unseal(read_json(output / "report.json")) == report
    assert not report["validated"] and not report["plant_referenced"]
