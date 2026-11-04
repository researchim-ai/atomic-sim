from os import PathLike
from typing import Any

from atomic_model_contracts import AgentDescriptor, Scenario

PathArg = str | PathLike[str]

def create(
    scenario: Scenario,
    agent: AgentDescriptor,
    folder: PathArg,
    command: list[str],
    *,
    identity_files: list[PathArg] | tuple[()] = (),
) -> dict[str, Any]: ...
def load(folder: PathArg) -> dict[str, Any]: ...
def run(
    folder: PathArg,
    command: list[str],
    *,
    identity_files: list[PathArg] | tuple[()] = (),
    max_decisions: int | None = None,
    deadline_s: float = 3600,
) -> dict[str, Any]: ...
def cancel(folder: PathArg) -> None: ...
def resume(
    folder: PathArg,
    *,
    expected_sha256: str,
    command: list[str],
    identity_files: list[PathArg] | tuple[()] = (),
) -> dict[str, Any]: ...
def export(folder: PathArg, output: PathArg) -> dict[str, Any]: ...
def replay(folder: PathArg) -> dict[str, Any]: ...
def source_identity() -> dict[str, Any]: ...
