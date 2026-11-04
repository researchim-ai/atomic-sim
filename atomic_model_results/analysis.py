"""Per-unit quality comparisons; no timing ranking or cross-protocol winner."""

import csv
import io

from atomic_model_reference.compare import signal_value
from atomic_model_reference.metrics import compare_curve, grid, interpolate
from atomic_model_workspace.common import digest
from atomic_model_workspace.reports import job_metrics


def protocol(p, ref):
    binding = p["study"]["references"][p["binding_id"]]
    project = p["study"]["project"]
    return dict(
        model_id=project["model_id"],
        scenario=project["scenarios"][binding["scenario_id"]],
        reference={
            key: ref[key]
            for key in (
                "physical_case_id",
                "role",
                "kind",
                "provenance",
                "times_s",
                "values",
                "uncertainty",
                "criteria",
            )
        },
        signals={
            name: dict(signal=c["signal"], unit=c["canonical_unit"])
            for name, c in ref["channels"].items()
        },
        budgets=dict(
            decision_timeout_s=project["limits"]["decision_timeout_s"],
            wall_limit_s=p["wall_limit_s"],
            rss_limit_bytes=p["rss_limit_bytes"],
            cpus=p["cpus"],
        ),
        source=p["source"],
    )


def summary(runs, current_sha):
    entries = {}
    groups = {}
    for id, (p, state, job, report, ref) in sorted(runs.items()):
        binding = p["study"]["references"][p["binding_id"]]
        policy = p["study"]["project"]["policies"][binding["policy_id"]]
        eligible = (
            state["status"] == "completed" and report is not None and report["reason"] is None
        )
        group = digest(protocol(p, ref)) if eligible else None
        entries[id] = dict(
            evaluation_id=id,
            study_revision=p["study"]["revision"],
            study_checkpoint_sha256=p["study_checkpoint_sha256"],
            stale=p["study_checkpoint_sha256"] != current_sha,
            binding_id=p["binding_id"],
            scenario_id=binding["scenario_id"],
            policy_id=binding["policy_id"],
            policy_sha256=digest(policy),
            status=state["status"],
            reason=state.get("reason"),
            comparison_passed=report["passed"] if report else None,
            eligible=eligible,
            protocol_sha256=group,
            exclusion=None
            if eligible
            else "incomplete execution or reference coverage/gap refusal",
            reference_id=ref["reference_id"],
            reference_kind=ref["kind"],
            role=ref["role"],
            channels=report["channels"] if report else {},
            events=report["events"] if report else [],
            job_metrics=job_metrics(job) if job else None,
            final_local_ledgers=job["rows"][-1]["circuits"] if job else None,
            observed_cost={k: state.get(k) for k in ("wall_s", "peak_rss_bytes", "attempt")},
        )
        if eligible:
            groups.setdefault(
                group,
                dict(
                    protocol=protocol(p, ref),
                    evaluations=[],
                    channel_rankings={},
                    performance_comparable=False,
                ),
            )["evaluations"].append(id)
    for group in groups.values():
        for channel in group["protocol"]["signals"]:
            values = sorted(
                (entries[id]["channels"][channel]["metrics"]["mae"], id)
                for id in group["evaluations"]
            )
            ranked = []
            previous = None
            rank = 0
            for i, (value, id) in enumerate(values, 1):
                if previous is None or value != previous:
                    rank = i
                ranked.append(
                    dict(
                        evaluation_id=id,
                        rank=rank,
                        mae=value,
                        unit=entries[id]["channels"][channel]["unit"],
                    )
                )
                previous = value
            group["channel_rankings"][channel] = ranked
    return dict(
        schema_version=1,
        entries=entries,
        groups=groups,
        semantics=dict(
            ranking="MAE separately per channel and exact protocol; equal floating values share competition rank; no mixed-unit score",
            quality="Only complete execution with covered reference window and admitted gaps; failed criteria remain visible and eligible",
            cost="Observed per-attempt process wall/RSS and retained policy timings; hardware identity not captured, no speed ranking or advantage claim",
            repeats="Repeated runs are not independent physical or statistical evidence",
            trajectory="Saved decision samples, not internal solver substeps",
        ),
        validated=False,
        plant_referenced=False,
    )


def _unit(name):
    for suffix, unit in [("_j_k", "J/K"), ("_kg_m_s", "kg m/s"), ("_kg", "kg"), ("_j", "J")]:
        if name.endswith(suffix):
            return unit
    raise ValueError("Unknown ledger unit: " + name)


def trajectory(job):
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["time_s", "kind", "channel", "value", "unit"])

    def scalars(prefix, value):
        for key, item in sorted(value.items()):
            name = prefix + key
            if isinstance(item, dict):
                yield from scalars(name + ".", item)
            else:
                yield name, item, _unit(key)

    for row in job["rows"]:
        t = row["time_s"]
        for channel in row["observation"]["channels"]:
            writer.writerow([t, "observation", channel["name"], channel["value"], channel["unit"]])
        for key, value in sorted((row["heat_power_w"] or {}).items()):
            writer.writerow([t, "action", key, value, "W"])
        for key, value, unit in scalars("", row["ledger"]):
            writer.writerow([t, "global_ledger", key, value, unit])
        for key, value, unit in scalars("", row["circuits"]):
            writer.writerow([t, "local_ledger", key, value, unit])
    return out.getvalue().encode()


def pair(runs, left, right, channel):
    if left == right or left not in runs or right not in runs:
        raise ValueError("Select two distinct retained evaluations")
    a, b = runs[left], runs[right]
    for p, s, j, r, ref in (a, b):
        if s["status"] != "completed" or r is None or r["reason"] is not None:
            raise ValueError("Pair requires complete comparable reports")
    if protocol(a[0], a[4]) != protocol(b[0], b[4]):
        raise ValueError("Different comparison protocols")
    if channel not in a[4]["channels"]:
        raise ValueError("Unknown comparison channel")
    ref = a[4]
    signal = ref["channels"][channel]["signal"]
    lo, hi = ref["criteria"]["window_s"]
    ta, tb = ([r["time_s"] for r in x[2]["rows"]] for x in (a, b))
    va, vb = ([signal_value(r, signal) for r in x[2]["rows"]] for x in (a, b))
    points = sorted(set(grid(ta, lo, hi) + grid(tb, lo, hi)))
    av = [interpolate(ta, va, t) for t in points]
    bv = [interpolate(tb, vb, t) for t in points]
    metrics = compare_curve(ta, va, tb, vb, [lo, hi])
    unit = ref["channels"][channel]["canonical_unit"]
    metric_units = {
        key: (
            "1"
            if key in ("comparison_knots", "fraction_time_within_declared_uncertainty")
            else "s"
            if key.endswith("_s")
            else "(" + unit + ") s"
            if "integral" in key
            else unit
        )
        for key in metrics
    }
    return dict(
        schema_version=1,
        left=left,
        right=right,
        channel=channel,
        unit=ref["channels"][channel]["canonical_unit"],
        time_s=points,
        left_value=av,
        right_value=bv,
        difference=[x - y for x, y in zip(av, bv)],
        metrics=metrics,
        metric_units=metric_units,
        metric_orientation="left minus right",
        decimated=False,
        validated=False,
        plant_referenced=False,
    )
