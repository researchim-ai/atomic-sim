"""Compare a completed EOS job with declared data and immutable criteria."""

import hashlib
import shutil
from pathlib import Path

from atomic_model_reference.bundle import load_bundle, source_identity
from atomic_model_reference.data import load_input, problem
from atomic_model_reference.metrics import compare_curve, first_crossing, max_gap
from atomic_model_runtime.episode import model_card
from atomic_model_workspace.common import atomic_json, canonical, digest
from atomic_model_workspace.jobs import load_job


def signal_value(row, signal):
    kind, name = signal.split(":", 1)
    if kind == "temperature":
        return next(c["value"] for c in row["observation"]["channels"] if c["name"] == name)
    if kind == "ledger":
        return row["ledger"][name]
    circuit, field = name.split(".", 1)
    return row["circuits"][circuit][field]


def compare_job(job_folder, reference_folder, output):
    ref, manifest = load_bundle(reference_folder)
    input_spec, raw_csv, parsed = load_input(Path(reference_folder) / "input.json")
    if canonical(parsed) != canonical(ref):
        raise ValueError("Reference changed during comparison")
    job = load_job(job_folder)
    normalized, physical, _ = problem(
        dict(
            model_id=ref["problem"]["model_id"],
            config=job["spec"]["scenario"]["config"],
            policy=job["spec"]["policy"],
        )
    )
    if digest(normalized) != ref["problem_sha256"] or physical != ref["physical_conditions_sha256"]:
        raise ValueError("Reference is bound to a different declared model problem")
    times = [r["time_s"] for r in job["rows"]]
    start, end = ref["criteria"]["window_s"]
    reason = None
    if job["status"] != "completed":
        reason = "model_episode_incomplete"
    elif times[0] > start or times[-1] < end:
        reason = "model_does_not_cover_window"
    elif max_gap(times, start, end) > ref["criteria"]["max_model_gap_s"]:
        reason = "model_interpolation_gap"
    channels = {}
    events = []
    if reason is None:
        for name, c in ref["channels"].items():
            values = [signal_value(row, c["signal"]) for row in job["rows"]]
            m = compare_curve(
                times,
                values,
                ref["times_s"],
                ref["values"][name],
                [start, end],
                ref["uncertainty"][name],
            )
            limits = ref["criteria"]["channels"][name]
            checks = {key: m[key.removesuffix("_max")] <= value for key, value in limits.items()}
            channels[name] = dict(
                unit=c["canonical_unit"],
                metric_units={
                    key: (
                        "1"
                        if key in ("comparison_knots", "fraction_time_within_declared_uncertainty")
                        else "s"
                        if key.endswith("_s")
                        else "(" + c["canonical_unit"] + ") s"
                        if "integral" in key
                        else c["canonical_unit"]
                    )
                    for key in m
                },
                signal=c["signal"],
                metrics=m,
                criteria=limits,
                checks=checks,
                passed=all(checks.values()),
            )
        for e in ref["criteria"]["events"]:
            c = ref["channels"][e["channel"]]
            values = [signal_value(row, c["signal"]) for row in job["rows"]]
            model_time = first_crossing(times, values, start, end, e["threshold"], e["direction"])
            reference_time = first_crossing(
                ref["times_s"],
                ref["values"][e["channel"]],
                start,
                end,
                e["threshold"],
                e["direction"],
            )
            delta = (
                None
                if model_time is None or reference_time is None
                else model_time - reference_time
            )
            events.append(
                dict(
                    id=e["id"],
                    channel=e["channel"],
                    threshold=e["threshold"],
                    threshold_unit=c["canonical_unit"],
                    direction=e["direction"],
                    model_time_s=model_time,
                    reference_time_s=reference_time,
                    error_s=delta,
                    max_time_error_s=e["max_time_error_s"],
                    passed=delta is not None and abs(delta) <= e["max_time_error_s"],
                    status="reference_event_missing"
                    if reference_time is None
                    else "model_event_missing"
                    if model_time is None
                    else "compared",
                )
            )
    passed = (
        reason is None
        and all(c["passed"] for c in channels.values())
        and all(e["passed"] for e in events)
    )
    result = dict(
        schema_version=1,
        reference_id=ref["reference_id"],
        physical_case_id=ref["physical_case_id"],
        role=ref["role"],
        evidence_kind=ref["kind"],
        provenance=ref["provenance"],
        provenance_authenticity=ref["provenance_authenticity"],
        reference_sha256=manifest["normalized_reference_sha256"],
        csv_sha256=ref["csv_sha256"],
        problem_sha256=ref["problem_sha256"],
        physical_conditions_sha256=physical,
        job_payload_sha256=digest(job),
        model_status=job["status"],
        model=model_card(),
        source=source_identity(),
        criteria=ref["criteria"],
        passed=passed,
        reason=reason,
        channels=channels,
        events=events,
        validated=False,
        plant_referenced=False,
        promotion="none; metric pass is not experimental or plant validation",
        metric_semantics=dict(
            interpolation="piecewise linear on union of model/reference knots within explicit window; no extrapolation",
            integrals="analytic segment integrals of error, absolute error and squared error; time weighted",
            peaks="first global maximum within window, not feature-matched multiple peaks",
            events="first directional strict-side to inclusive-threshold crossing; initial equality/plateaus do not create a new crossing",
            uncertainty="fraction of time within supplied reference ±uncertainty, not a confidence/coverage certification; unknown stays null",
        ),
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        atomic_json(output / "report.json", result)
        atomic_json(output / "reference_snapshot.json", ref)
        atomic_json(output / "reference_input.json", input_spec)
        (output / "data.csv").write_bytes(raw_csv)
        atomic_json(output / "job_snapshot.json", job)
        atomic_json(
            output / "manifest.json",
            dict(
                schema_version=1,
                source=result["source"],
                artifacts={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in output.iterdir()
                    if p.is_file()
                },
                validated=False,
                plant_referenced=False,
            ),
        )
    except BaseException:
        shutil.rmtree(output)
        raise
    return result
