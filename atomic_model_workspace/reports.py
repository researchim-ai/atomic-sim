"""Measured quality and local/global balance summaries; no implicit reward or promotion."""

import csv
import io
from pathlib import Path

from atomic_model_workspace.campaign import load_campaign
from atomic_model_workspace.common import atomic_json, digest
from atomic_model_workspace.jobs import load_job


def job_metrics(state):
    objective = state["spec"]["scenario"]["objective"]
    rows = state["rows"]

    def error(row):
        sensor = next(c for c in row["observation"]["channels"] if c["name"] == objective["sensor"])
        return sensor["value"] - objective["target_k"]

    iae = 0.0
    effort = 0.0
    for before, after in zip(rows, rows[1:]):
        dt = after["time_s"] - before["time_s"]
        iae += dt * (abs(error(before)) + abs(error(after))) / 2
        effort += dt * sum(abs(p) for p in after["heat_power_w"].values())
    completed = state["status"] == "completed"
    duration = rows[-1]["time_s"] - rows[0]["time_s"]
    return dict(
        status=state["status"],
        reason=state["reason"],
        completed=completed,
        accepted_decisions=state["episode"]["steps"],
        completed_time_s=rows[-1]["time_s"],
        integral_absolute_error_k_s=iae,
        mean_absolute_error_k=iae / duration if duration else None,
        final_error_k=error(rows[-1]),
        peak_absolute_error_k=max(abs(error(r)) for r in rows),
        final_target_met=completed and abs(error(rows[-1])) <= objective["tolerance_k"],
        absolute_heat_input_j=effort,
        max_absolute_energy_residual_j=max(abs(r["ledger"]["energy_residual_j"]) for r in rows),
        max_absolute_entropy_residual_j_k=max(
            abs(r["ledger"]["entropy_residual_j_k"]) for r in rows
        ),
        max_absolute_mass_residual_kg=max(
            abs(c["mass_residual_kg"]) for r in rows for c in r["circuits"].values()
        ),
        max_absolute_momentum_residual_kg_m_s=max(
            abs(c["momentum_residual_kg_m_s"]) for r in rows for c in r["circuits"].values()
        ),
        max_absolute_circuit_energy_residual_j=max(
            abs(c["energy_residual_j"]) for r in rows for c in r["circuits"].values()
        ),
        policy_wall_s=sum(r["policy_wall_s"] for r in rows),
        final_ledger=rows[-1]["ledger"],
    )


def campaign_report(folder):
    state = load_campaign(folder)
    jobs = []
    for e in state["entries"]:
        job = load_job(Path(folder) / e["job"])
        jobs.append(
            dict(
                scenario_id=e["scenario_id"],
                policy_id=e["policy_id"],
                scenario_sha256=digest(job["spec"]["scenario"]),
                policy_sha256=digest(job["spec"]["policy"]),
                split=job["spec"]["scenario"]["split"],
                seed=job["spec"]["scenario"]["seed"],
                metrics=job_metrics(job),
                attempts=e["attempts"],
            )
        )
    return dict(
        schema_version=1,
        project_sha256=state["project_sha256"],
        status=state["status"],
        provenance=state["provenance"],
        execution_environment=state["execution_environment"],
        validated=False,
        plant_referenced=False,
        jobs=jobs,
        metric_definitions=dict(
            integral_absolute_error_k_s="Trapezoidal absolute observed temperature error on accepted decision times only",
            final_target_met="Full horizon completed and final temperature within declared tolerance",
            absolute_heat_input_j="Integral of sum of absolute commanded port powers; not net external heat",
            balances="Accepted output samples; solver's own admission also applies internally",
            timing="Observed monotonic wall time; not a portable speed guarantee",
        ),
    )


def export_report(folder, output):
    report = campaign_report(folder)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "report.json", report)
    rows = []
    for entry in report["jobs"]:
        rows.append(
            {
                **{k: entry[k] for k in ("scenario_id", "policy_id", "split", "seed")},
                **{k: v for k, v in entry["metrics"].items() if k != "final_ledger"},
            }
        )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (output / "metrics.csv").write_text(buffer.getvalue())
    units = dict(
        integral_absolute_error_k_s="K s",
        mean_absolute_error_k="K",
        final_error_k="K",
        peak_absolute_error_k="K",
        absolute_heat_input_j="J",
        completed_time_s="s",
        policy_wall_s="s",
        max_absolute_energy_residual_j="J",
        max_absolute_entropy_residual_j_k="J/K",
        max_absolute_mass_residual_kg="kg",
        max_absolute_momentum_residual_kg_m_s="kg m/s",
        max_absolute_circuit_energy_residual_j="J",
    )
    import hashlib

    atomic_json(
        output / "manifest.json",
        dict(
            schema_version=1,
            units=units,
            artifacts={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.iterdir()
                if p.is_file()
            },
            validated=False,
            plant_referenced=False,
        ),
    )
    return report


def compare_campaigns(left, right):
    a, b = load_campaign(left), load_campaign(right)
    for name in ("model_id", "scenarios", "limits"):
        if a["project"][name] != b["project"][name]:
            raise ValueError(
                "Comparison requires identical scenarios, numerical settings and budgets"
            )

    def environments(state):
        values = {
            digest(attempt["execution_environment"])
            for entry in state["entries"]
            for attempt in entry["attempts"]
        }
        return values or {digest(state["execution_environment"])}

    if environments(a) != environments(b):
        raise ValueError("Comparison execution environment mismatch")
    if a["provenance"] != b["provenance"]:
        raise ValueError("Comparison source mismatch")
    ar, br = campaign_report(left), campaign_report(right)
    deltas = []
    for left_entry in ar["jobs"]:
        for right_entry in br["jobs"]:
            if left_entry["scenario_id"] != right_entry["scenario_id"]:
                continue
            lm, rm = left_entry["metrics"], right_entry["metrics"]
            deltas.append(
                dict(
                    scenario_id=left_entry["scenario_id"],
                    left_policy=left_entry["policy_id"],
                    right_policy=right_entry["policy_id"],
                    both_completed=lm["completed"] and rm["completed"],
                    right_minus_left_iae_k_s=(
                        rm["integral_absolute_error_k_s"] - lm["integral_absolute_error_k_s"]
                    )
                    if lm["completed"] and rm["completed"]
                    else None,
                )
            )
    return dict(
        schema_version=1,
        comparable=True,
        paired_deltas=deltas,
        validated=False,
        plant_referenced=False,
        left=ar,
        right=br,
        ranking=None,
        note="All jobs, failures and accepted prefixes retained. No automatic superiority claim or selection of successful cases.",
    )
