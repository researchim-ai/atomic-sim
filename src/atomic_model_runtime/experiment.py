"""Versioned EOS experiment runner with decision-indexed actions and exact resume."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from atomic_model_runtime.episode import (
    MODEL_ID,
    EOSEpisode,
    EpisodeRejected,
    model_card,
    runtime_provenance,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare(spec):
    if (
        not isinstance(spec, dict)
        or set(spec) != {"schema_version", "model_id", "config", "timeline"}
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or spec["model_id"] != MODEL_ID
    ):
        raise ValueError("Expected EOS experiment schema1")
    env = EOSEpisode(spec["config"])
    timeline = spec["timeline"]
    if not isinstance(timeline, list) or len(timeline) > 128:
        raise ValueError("EOS timeline size limit")
    previous = 0
    normalized = []
    for event in timeline:
        if not isinstance(event, dict) or set(event) != {"decision", "heat_power_w"}:
            raise ValueError("Malformed EOS timeline event")
        decision = event["decision"]
        if (
            type(decision) is not int
            or not previous < decision <= 1024
            or (decision - 1) * env.config.action_dt_s >= env.config.episode_horizon_s
        ):
            raise ValueError("EOS decisions must be unique ordered positive indices within horizon")
        normalized.append(
            dict(decision=decision, heat_power_w=env.validate_action(event["heat_power_w"]))
        )
        previous = decision
    return dict(
        schema_version=1, model_id=MODEL_ID, config=env.config.as_dict(), timeline=normalized
    ), env


def _load_checkpoint(value, spec, env):
    if (
        not isinstance(value, dict)
        or set(value) != {"payload", "sha256"}
        or digest(value["payload"]) != value["sha256"]
    ):
        raise ValueError("Experiment checkpoint checksum mismatch")
    p = value["payload"]
    if (
        not isinstance(p, dict)
        or set(p) != {"schema_version", "input_sha256", "episode"}
        or type(p["schema_version"]) is not int
        or p["schema_version"] != 1
        or p["input_sha256"] != digest(spec)
    ):
        raise ValueError("Experiment input identity mismatch")
    restored = EOSEpisode.from_episode(p["episode"])
    if restored.config.as_dict() != env.config.as_dict():
        raise ValueError("Experiment checkpoint configuration mismatch")
    # Ensure past commands match this scenario, not only the episode's local receipts.
    powers = env._model.snapshot()["active_heat_power_w"]
    events = {e["decision"]: e["heat_power_w"] for e in spec["timeline"]}
    for i, row in enumerate(restored.export_episode()["action_log"], 1):
        powers = events.get(i, powers)
        if row["commands"] != powers:
            raise ValueError("Checkpoint diverges from scenario control history")
    return restored


def run_experiment(spec, output, *, checkpoint=None, max_decisions=None):
    normalized, env = prepare(spec)
    if max_decisions is not None and (
        type(max_decisions) is not int or not 1 <= max_decisions <= 1024
    ):
        raise ValueError("Invalid per-run decision budget")
    if checkpoint is not None:
        env = _load_checkpoint(checkpoint, normalized, env)
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "input.json", normalized)
    started = env._model.snapshot()["time_s"]
    rows = []

    def record(phase):
        state = env._model.snapshot()
        rows.append(
            dict(
                time_s=state["time_s"],
                phase=phase,
                port_temperatures_k=state["port_temperatures_k"],
                energy_residual_j=state["energy_residual_j"],
                entropy_residual_j_k=state["entropy_residual_j_k"],
                mass_residuals_kg={n: s["mass_residual_kg"] for n, s in state["circuits"].items()},
                momentum_residuals_kg_m_s={
                    n: s["momentum_residual_kg_m_s"] for n, s in state["circuits"].items()
                },
            )
        )

    record("restart" if checkpoint is not None else "initial")
    powers = env._model.snapshot()["active_heat_power_w"]
    events = {e["decision"]: e["heat_power_w"] for e in normalized["timeline"]}
    status = "completed"
    error = None
    accepted = 0
    while not env._done:
        if max_decisions is not None and accepted >= max_decisions:
            status = "paused"
            break
        powers = events.get(env._steps + 1, powers)
        try:
            env.step([powers[k] for k in env.config.action_channels])
        except EpisodeRejected as exc:
            status = "rejected"
            error = dict(criterion=exc.criterion, message=str(exc))
            break
        accepted += 1
        record("accepted")
    payload = dict(schema_version=1, input_sha256=digest(normalized), episode=env.export_episode())
    write(folder / "checkpoint.json", dict(payload=payload, sha256=digest(payload)))
    with (folder / "trajectory.jsonl").open("w") as out:
        for row in rows:
            out.write(canonical(row) + "\n")
    ports = list(rows[0]["port_temperatures_k"])
    with (folder / "trajectory.csv").open("w", newline="") as out:
        csvout = csv.DictWriter(
            out, fieldnames=["time_s", "phase", *ports, "energy_residual_j", "entropy_residual_j_k"]
        )
        csvout.writeheader()
        for row in rows:
            csvout.writerow(
                dict(
                    time_s=row["time_s"],
                    phase=row["phase"],
                    **row["port_temperatures_k"],
                    energy_residual_j=row["energy_residual_j"],
                    entropy_residual_j_k=row["entropy_residual_j_k"],
                )
            )
    manifest = dict(
        schema_version=1,
        input_sha256=digest(normalized),
        status=status,
        error=error,
        model=model_card(),
        provenance=runtime_provenance(),
        start_time_s=started,
        completed_time_s=rows[-1]["time_s"],
        decisions_this_run=accepted,
        total_decisions=env._steps,
        numerical_acceptance=status == "completed",
        validated=False,
        plant_referenced=False,
        units=dict(
            time_s="s",
            port_temperatures_k="K",
            energy_residual_j="J",
            entropy_residual_j_k="J/K",
            mass_residuals_kg="kg",
            momentum_residuals_kg_m_s="kg m/s",
        ),
        artifacts={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.iterdir()
            if p.is_file()
        },
    )
    write(folder / "manifest.json", manifest)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--max-decisions", type=int)
    a = p.parse_args(argv)
    try:
        result = run_experiment(
            json.loads(a.input.read_text()),
            a.output,
            checkpoint=None if a.checkpoint is None else json.loads(a.checkpoint.read_text()),
            max_decisions=a.max_decisions,
        )
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    print(
        canonical(
            dict(
                status=result["status"],
                completed_time_s=result["completed_time_s"],
                output=str(a.output),
            )
        )
    )
    return 0 if result["status"] in ["completed", "paused"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
