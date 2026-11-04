"""Type-checked public consumer example; it never bypasses runtime validators."""

from pathlib import Path
from typing import Any

from atomic_model_agents import jobs
from atomic_model_contracts import AgentDescriptor, Scenario


def start_checked_job(
    scenario: Scenario, agent: AgentDescriptor, output: Path, command: list[str]
) -> dict[str, Any]:
    return jobs.create(scenario, agent, output, command)


def run_checked_job(
    output: Path, command: list[str], decisions: int | None = None
) -> dict[str, Any]:
    return jobs.run(output, command, max_decisions=decisions)
