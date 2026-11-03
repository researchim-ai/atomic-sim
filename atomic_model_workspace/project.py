"""Versioned portable project admission; all scenarios/policies checked before work."""

import os
import platform
from pathlib import Path

from atomic_model_runtime.episode import MODEL_ID, EOSEpisode, model_card, runtime_provenance
from atomic_model_workspace.common import digest, identifier, keys, local_file, number, read
from atomic_model_workspace.policies import prepare_policy


def provenance():
    root = Path(__file__).parent
    return dict(
        workspace_sha256=digest({p.name: p.read_text() for p in sorted(root.glob("*.py"))}),
        runtime=runtime_provenance(),
    )


def execution_environment():
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if not cpu_model and cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return dict(
        python=platform.python_version(),
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        cpu_model=cpu_model,
        logical_cpus=os.cpu_count(),
        threads={
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    )


def prepare_scenario(spec):
    keys(spec, ("schema_version", "config", "seed", "split", "objective"))
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("Expected scenario schema1")
    if (
        type(spec["seed"]) is not int
        or not 0 <= spec["seed"] < 2**32
        or spec["split"] not in ("train", "dev", "test")
    ):
        raise ValueError("Invalid seed or dataset split")
    env = EOSEpisode(spec["config"])
    objective = spec["objective"]
    keys(objective, ("sensor", "target_k", "tolerance_k"))
    if objective["sensor"] not in [c["name"] for c in env.observation_metadata()["channels"]]:
        raise ValueError("Unknown objective sensor")
    objective = dict(
        sensor=objective["sensor"],
        target_k=number(objective["target_k"], 1, 10000, "target_k"),
        tolerance_k=number(objective["tolerance_k"], 0, 10000, "tolerance_k"),
    )
    return dict(
        schema_version=1,
        config=env.config.as_dict(),
        seed=spec["seed"],
        split=spec["split"],
        objective=objective,
    ), env


def normalize(manifest, scenarios, policies):
    keys(manifest, ("schema_version", "project_id", "model_id", "scenarios", "policies", "limits"))
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["model_id"] != MODEL_ID
    ):
        raise ValueError("Expected EOS project schema1")
    identifier(manifest["project_id"])
    for refs in (manifest["scenarios"], manifest["policies"]):
        if not isinstance(refs, dict) or not 1 <= len(refs) <= 32:
            raise ValueError("Project requires 1..32 scenarios and policies")
        for name in refs:
            identifier(name)
    if set(scenarios) != set(manifest["scenarios"]) or set(policies) != set(manifest["policies"]):
        raise ValueError("Project reference identity mismatch")
    limits = manifest["limits"]
    keys(limits, ("decision_timeout_s", "job_timeout_s"))
    limits = dict(
        decision_timeout_s=number(limits["decision_timeout_s"], 0.001, 60, "decision_timeout_s"),
        job_timeout_s=number(limits["job_timeout_s"], 0.1, 3600, "job_timeout_s"),
    )
    normalized_scenarios = {}
    normalized_policies = {}
    for sid, value in sorted(scenarios.items()):
        scenario, env = prepare_scenario(value)
        normalized_scenarios[sid] = scenario
        for pid, policy in sorted(policies.items()):
            admitted = prepare_policy(policy, env)
            if pid in normalized_policies and normalized_policies[pid] != admitted:
                raise ValueError("Policy is not compatible with all project scenarios")
            normalized_policies[pid] = admitted
    return dict(
        schema_version=1,
        project_id=manifest["project_id"],
        model_id=MODEL_ID,
        scenarios=normalized_scenarios,
        policies=normalized_policies,
        limits=limits,
    )


def load_project(path):
    path = Path(path)
    manifest = read(path)
    keys(manifest, ("schema_version", "project_id", "model_id", "scenarios", "policies", "limits"))
    # Validate containers before opening any referenced file.
    for group in ("scenarios", "policies"):
        if not isinstance(manifest[group], dict) or not 1 <= len(manifest[group]) <= 32:
            raise ValueError("Project reference count limit")
    values = {
        g: {k: read(local_file(path.parent, ref)) for k, ref in manifest[g].items()}
        for g in ("scenarios", "policies")
    }
    return normalize(manifest, values["scenarios"], values["policies"])


def validate_project(path):
    value = load_project(path)
    return dict(
        project_sha256=digest(value),
        model=model_card(),
        scenarios=len(value["scenarios"]),
        policies=len(value["policies"]),
        jobs=len(value["scenarios"]) * len(value["policies"]),
        provenance=provenance(),
        seed_semantics="Recorded identity; EOS runtime and built-in policies are deterministic and consume no RNG",
    )
