"""Strict shared budgets and physical split guards for synthetic EOS cases."""

import hashlib
from pathlib import Path

from atomic_model_agents import jobs
from atomic_model_agents.protocol import descriptor
from atomic_model_runtime.episode import MODEL_ID
from atomic_model_workspace.common import digest, identifier, keys, number
from atomic_model_workspace.project import execution_environment, prepare_scenario


def source_identity():
    return dict(
        arena_sha256=digest(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            }
        ),
        agents=jobs.source_identity(),
    )


def normalize(spec):
    keys(
        spec,
        (
            "schema_version",
            "benchmark_id",
            "model_id",
            "cases",
            "agents",
            "baseline_id",
            "repeats",
            "budgets",
            "constraints",
        ),
    )
    if (
        type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or spec["model_id"] != MODEL_ID
    ):
        raise ValueError("Expected EOS arena schema1")
    identifier(spec["benchmark_id"])
    if not isinstance(spec["cases"], dict) or not 3 <= len(spec["cases"]) <= 12:
        raise ValueError("Arena needs 3..12 cases")
    if not isinstance(spec["agents"], dict) or not 2 <= len(spec["agents"]) <= 8:
        raise ValueError("Arena needs baseline and 1..7 candidates")
    if spec["baseline_id"] not in spec["agents"]:
        raise ValueError("Unknown baseline")
    if type(spec["repeats"]) is not int or not 2 <= spec["repeats"] <= 4:
        raise ValueError("Arena repeats must be 2..4")
    b = spec["budgets"]
    keys(b, ("job_wall_s", "total_wall_s", "worker_rss_bytes", "max_decisions"))
    budgets = dict(
        job_wall_s=number(b["job_wall_s"], 0.1, 60, "job_wall_s"),
        total_wall_s=number(b["total_wall_s"], 1, 3600, "total_wall_s"),
        worker_rss_bytes=b["worker_rss_bytes"],
        max_decisions=b["max_decisions"],
    )
    if (
        type(b["worker_rss_bytes"]) is not int
        or not 128 * 1024**2 <= b["worker_rss_bytes"] <= 4 * 1024**3
    ):
        raise ValueError("Arena worker RSS budget")
    if type(b["max_decisions"]) is not int or not 1 <= b["max_decisions"] <= 128:
        raise ValueError("Arena decision budget")
    constraint_names = (
        "energy_residual_j",
        "entropy_residual_j_k",
        "mass_residual_kg",
        "momentum_residual_kg_m_s",
        "circuit_energy_residual_j",
        "absolute_power_w",
    )
    keys(spec["constraints"], constraint_names)
    constraints = {k: number(spec["constraints"][k], 0, 1e12, k) for k in constraint_names}
    cases = {}
    agents = {}
    physical_splits = {}
    roles = set()
    shared = None
    for name, raw in sorted(spec["cases"].items()):
        identifier(name)
        scenario, env = prepare_scenario(raw)
        roles.add(scenario["split"])
        physical = digest(env._spec)
        if physical in physical_splits and physical_splits[physical] != scenario["split"]:
            raise ValueError(
                "Physical split leakage: seed/objective/horizon changes do not create independent physics"
            )
        physical_splits[physical] = scenario["split"]
        cases[name] = scenario
        for id, agent in sorted(spec["agents"].items()):
            identifier(id)
            value = descriptor(agent, env)
            permissions = {
                k: value[k]
                for k in ("observations", "actions", "decision_timeout_s", "rss_limit_bytes")
            }
            if shared is None:
                shared = permissions
            elif shared != permissions:
                raise ValueError("Agents/cases must share permissions and decision budgets")
            if id in agents and agents[id] != value:
                raise ValueError("Agent descriptor depends on case normalization")
            agents[id] = value
    if roles != {"train", "dev", "test"}:
        raise ValueError("All train/dev/test roles are required")
    return dict(
        schema_version=1,
        benchmark_id=spec["benchmark_id"],
        model_id=MODEL_ID,
        cases=cases,
        agents=agents,
        baseline_id=spec["baseline_id"],
        repeats=spec["repeats"],
        budgets=budgets,
        constraints=constraints,
    )


def schedule(pack, phase, selected=None):
    names = (
        sorted(pack["agents"])
        if phase == "train"
        else sorted([pack["baseline_id"], selected])
        if selected
        else []
    )
    rows = []
    for repeat in range(pack["repeats"]):
        order = names[repeat % len(names) :] + names[: repeat % len(names)] if names else []
        for case, scenario in pack["cases"].items():
            if scenario["split"] != phase:
                continue
            for agent in order:
                key = dict(
                    protocol_sha256=digest(pack), case_id=case, agent_id=agent, repeat=repeat
                )
                rows.append(
                    dict(
                        case_id=case,
                        agent_id=agent,
                        repeat=repeat,
                        task_id=digest(key),
                        phase=phase,
                    )
                )
    return rows


COST_SCOPE = "one shared CPU affinity, sequential order; host load/frequency not controlled; not a cross-machine speed claim"


def validate_hardware(value):
    keys(value, ("execution_environment", "cpu_affinity", "cost_scope"))
    info = value["execution_environment"]
    keys(info, ("python", "system", "release", "machine", "cpu_model", "logical_cpus", "threads"))
    for name in ("python", "system", "release", "machine", "cpu_model"):
        if not isinstance(info[name], str) or len(info[name]) > 1024:
            raise ValueError("Arena hardware identity schema")
    if info["logical_cpus"] is not None and (
        type(info["logical_cpus"]) is not int or not 1 <= info["logical_cpus"] <= 65536
    ):
        raise ValueError("Arena logical CPU schema")
    keys(
        info["threads"],
        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"),
    )
    if (
        any(v != "1" for v in info["threads"].values())
        or not isinstance(value["cpu_affinity"], list)
        or len(value["cpu_affinity"]) != 1
        or type(value["cpu_affinity"][0]) is not int
        or not 0 <= value["cpu_affinity"][0] < 65536
        or value["cost_scope"] != COST_SCOPE
    ):
        raise ValueError("Arena CPU/thread budget metadata")
    return value


def hardware():
    import os

    return dict(
        execution_environment=execution_environment(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        cost_scope=COST_SCOPE,
    )
