"""Pytest plugin: requested lane must execute tests, not silently skip them all."""

import json
from pathlib import Path

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("atomic-release")
    group.addoption("--atomic-receipt", default=None)
    group.addoption("--atomic-require-executed", action="store_true")
    group.addoption("--atomic-no-skips", action="store_true")


def pytest_configure(config):
    config._atomic_results = dict(passed=0, failed=0, skipped=0, errors=0, collected=0)


def pytest_collection_finish(session):
    session.config._atomic_results["collected"] = len(session.items)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    counts = item.config._atomic_results
    if report.skipped:
        counts["skipped"] += 1
    elif report.when == "call":
        counts["passed" if report.passed else "failed"] += 1
    elif report.failed:
        counts["errors"] += 1


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    counts = config._atomic_results
    reasons = []
    if config.getoption("--atomic-require-executed") and counts["passed"] + counts["failed"] == 0:
        reasons.append("no tests executed")
    if config.getoption("--atomic-no-skips") and counts["skipped"]:
        reasons.append("lane skipped required tests")
    if reasons and session.exitstatus == 0:
        session.exitstatus = 1
    target = config.getoption("--atomic-receipt")
    if target:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    counts=counts,
                    exit_code=int(session.exitstatus),
                    gate_failures=reasons,
                    external_or_physical_validation_claimed=False,
                ),
                indent=2,
            )
            + "\n"
        )
