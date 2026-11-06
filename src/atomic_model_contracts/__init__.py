"""Static public input contracts; runtime admission remains authoritative."""

from typing import Any, Literal, TypedDict


class Config(TypedDict):
    preset: Literal["uniform", "mixed_ports", "gravity"]
    action_dt_s: float
    episode_horizon_s: float
    max_step_s: float


class Objective(TypedDict):
    sensor: str
    target_k: float
    tolerance_k: float


class Scenario(TypedDict):
    schema_version: Literal[1]
    config: Config
    seed: int
    split: Literal["train", "dev", "test"]
    objective: Objective


class AgentDescriptor(TypedDict):
    schema_version: Literal[1]
    agent_id: str
    agent_version: str
    observations: list[str]
    actions: list[str]
    initial_state: Any
    decision_timeout_s: float
    rss_limit_bytes: int
