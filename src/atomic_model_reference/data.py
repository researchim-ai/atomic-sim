"""Strict reference CSV admission, physical case identity and declared units."""

import csv
import hashlib
import io
from pathlib import Path

from atomic_model_runtime.episode import MODEL_ID, EOSEpisode
from atomic_model_workspace.common import (
    canonical,
    digest,
    identifier,
    keys,
    local_file,
    number,
    read,
)
from atomic_model_workspace.policies import prepare_policy

MAX_CSV_BYTES = 8 * 1024 * 1024
MAX_ROWS = 20000
UNITS = {
    "K": ("temperature", 1.0, 0.0),
    "degC": ("temperature", 1.0, 273.15),
    "J": ("energy", 1.0, 0.0),
    "kJ": ("energy", 1000.0, 0.0),
    "J/K": ("entropy", 1.0, 0.0),
    "kJ/K": ("entropy", 1000.0, 0.0),
    "kg": ("mass", 1.0, 0.0),
    "kg m/s": ("momentum", 1.0, 0.0),
}
LEDGERS = {
    "total_energy_change_j": "J",
    "external_heat_j": "J",
    "exported_drag_heat_j": "J",
    "entropy_change_j_k": "J/K",
    "external_entropy_j_k": "J/K",
    "generated_entropy_j_k": "J/K",
    "energy_residual_j": "J",
    "entropy_residual_j_k": "J/K",
}


def problem(spec):
    keys(spec, ("model_id", "config", "policy"))
    if spec["model_id"] != MODEL_ID:
        raise ValueError("Reference supports only declared EOS model")
    env = EOSEpisode(spec["config"])
    policy = prepare_policy(spec["policy"], env)
    normalized = dict(model_id=MODEL_ID, config=env.config.as_dict(), policy=policy)
    physical_config = {"preset_spec_sha256": digest(env._spec)}
    if policy["kind"] != "hold-v1":
        physical_config["action_dt_s"] = env.config.action_dt_s
    # Horizon/output crop and numerical max_step must not create a new held-out physical case.
    identity = digest(dict(model_id=MODEL_ID, physical_config=physical_config, policy=policy))
    signals = {"temperature:" + c["name"]: "K" for c in env.observation_metadata()["channels"]}
    signals.update({"ledger:" + k: v for k, v in LEDGERS.items()})
    for circuit in ("primary", "secondary"):
        for name, unit in [
            ("mass_residual_kg", "kg"),
            ("momentum_residual_kg_m_s", "kg m/s"),
            ("energy_residual_j", "J"),
        ]:
            signals[f"circuit:{circuit}.{name}"] = unit
    return normalized, identity, signals


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError("Missing/bounded provenance " + name)
    return value


def prepare(spec, raw):
    keys(
        spec,
        (
            "schema_version",
            "reference_id",
            "physical_case_id",
            "role",
            "kind",
            "provenance",
            "problem",
            "csv",
            "channels",
            "criteria",
        ),
    )
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("Expected reference schema1")
    identifier(spec["reference_id"])
    identifier(spec["physical_case_id"])
    if spec["role"] not in ("calibration", "validation", "test") or spec["kind"] not in (
        "experiment",
        "intercode",
        "numerical_reference",
    ):
        raise ValueError("Unknown reference role or evidence kind")
    keys(spec["provenance"], ("title", "source", "license", "revision", "notes"))
    provenance = {k: _text(v, k) for k, v in spec["provenance"].items()}
    normalized, physical, signals = problem(spec["problem"])
    wire = spec["csv"]
    keys(wire, ("path", "sha256", "time_column", "time_unit"))
    if (
        not isinstance(raw, bytes)
        or not 0 < len(raw) <= MAX_CSV_BYTES
        or hashlib.sha256(raw).hexdigest() != wire["sha256"]
    ):
        raise ValueError("Reference CSV bytes/hash limit mismatch")
    if wire["time_unit"] not in ("s", "ms"):
        raise ValueError("Unsupported time unit")
    _text(wire["time_column"], "time column")
    channels = spec["channels"]
    if not isinstance(channels, dict) or not 1 <= len(channels) <= 32:
        raise ValueError("Reference needs 1..32 channels")
    normalized_channels = {}
    columns = {wire["time_column"]}
    for name, c in channels.items():
        identifier(name)
        keys(c, ("signal", "column", "unit", "uncertainty_column"))
        if (
            not isinstance(c["signal"], str)
            or not isinstance(c["unit"], str)
            or c["signal"] not in signals
            or c["unit"] not in UNITS
            or UNITS[c["unit"]][0] != UNITS[signals[c["signal"]]][0]
        ):
            raise ValueError("Unsupported signal or incompatible unit")
        _text(c["column"], "value column")
        for col in (c["column"], c["uncertainty_column"]):
            if col is None:
                continue
            _text(col, "column")
            if col in columns:
                raise ValueError("Column bound more than once")
            columns.add(col)
        normalized_channels[name] = dict(**c, canonical_unit=signals[c["signal"]])
    criteria = spec["criteria"]
    keys(criteria, ("window_s", "max_reference_gap_s", "max_model_gap_s", "channels", "events"))
    window = criteria["window_s"]
    if not isinstance(window, list) or len(window) != 2:
        raise ValueError("Expected explicit comparison window")
    start, end = [number(t, 0, 1e9, "window time") for t in window]
    if start >= end:
        raise ValueError("Empty reference comparison window")
    gaps = {
        key: number(criteria[key], 1e-12, 1e9, key)
        for key in ("max_reference_gap_s", "max_model_gap_s")
    }
    if not isinstance(criteria["channels"], dict) or set(criteria["channels"]) != set(channels):
        raise ValueError("Declare criteria for every reference channel")
    limits = {}
    for name, values in criteria["channels"].items():
        keys(
            values,
            ("mae_max", "rmse_max", "peak_absolute_error_max", "integral_absolute_error_max"),
        )
        limits[name] = {k: number(v, 0, 1e200, k) for k, v in values.items()}
    events = criteria["events"]
    admitted_events = []
    seen = set()
    if not isinstance(events, list) or len(events) > 64:
        raise ValueError("Event criteria count limit")
    for e in events:
        keys(e, ("id", "channel", "threshold", "direction", "max_time_error_s"))
        identifier(e["id"])
        if e["id"] in seen or e["channel"] not in channels or e["direction"] not in ("up", "down"):
            raise ValueError("Invalid/duplicate event criterion")
        seen.add(e["id"])
        admitted_events.append(
            dict(
                id=e["id"],
                channel=e["channel"],
                direction=e["direction"],
                threshold=number(e["threshold"], -1e100, 1e100, "canonical event threshold"),
                max_time_error_s=number(e["max_time_error_s"], 0, 1e9, "event time budget"),
            )
        )
    try:
        rows = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
        header = next(rows)
        if not header or len(header) != len(set(header)) or set(header) != columns:
            raise ValueError("CSV headers must match declared columns exactly")
        index = {name: i for i, name in enumerate(header)}
        times = []
        values = {k: [] for k in channels}
        sigmas = {
            k: [] if c["uncertainty_column"] is not None else None for k, c in channels.items()
        }
        for row in rows:
            if len(times) >= MAX_ROWS or len(row) != len(header):
                raise ValueError("CSV row/width limit")
            t = number(float(row[index[wire["time_column"]]]), 0, 1e12, "reference time") * (
                0.001 if wire["time_unit"] == "ms" else 1.0
            )
            if times and t <= times[-1]:
                raise ValueError("Reference times must increase strictly")
            times.append(t)
            for name, c in channels.items():
                _, scale, offset = UNITS[c["unit"]]
                y = (
                    number(float(row[index[c["column"]]]), -1e100, 1e100, "reference value") * scale
                    + offset
                )
                if normalized_channels[name]["canonical_unit"] == "K" and y <= 0:
                    raise ValueError("Temperature outside positive kelvin range")
                values[name].append(y)
                if sigmas[name] is not None:
                    sigmas[name].append(
                        number(
                            float(row[index[c["uncertainty_column"]]]),
                            0,
                            1e100,
                            "reference uncertainty",
                        )
                        * scale
                    )
    except (UnicodeError, csv.Error, StopIteration) as exc:
        raise ValueError("Malformed reference CSV") from exc
    if len(times) < 2 or times[0] > start or times[-1] < end:
        raise ValueError("Reference must cover declared window with at least two samples")
    from atomic_model_reference.metrics import max_gap

    if max_gap(times, start, end) > gaps["max_reference_gap_s"]:
        raise ValueError("Reference interpolation gap exceeds criterion")
    result = dict(
        schema_version=1,
        reference_id=spec["reference_id"],
        physical_case_id=spec["physical_case_id"],
        role=spec["role"],
        kind=spec["kind"],
        provenance=provenance,
        provenance_authenticity="user-declared, not authenticated",
        problem=normalized,
        physical_conditions_sha256=physical,
        problem_sha256=digest(normalized),
        csv_sha256=wire["sha256"],
        channels=normalized_channels,
        criteria=dict(window_s=[start, end], **gaps, channels=limits, events=admitted_events),
        times_s=times,
        values=values,
        uncertainty=sigmas,
        validated=False,
        plant_referenced=False,
    )
    if len(canonical(result).encode()) > 16 * 1024 * 1024:
        raise ValueError("Normalized reference size limit")
    return result


def load_input(path):
    path = Path(path)
    spec = read(path)
    keys(
        spec,
        (
            "schema_version",
            "reference_id",
            "physical_case_id",
            "role",
            "kind",
            "provenance",
            "problem",
            "csv",
            "channels",
            "criteria",
        ),
    )
    keys(spec["csv"], ("path", "sha256", "time_column", "time_unit"))
    csv_path = local_file(path.parent, spec["csv"]["path"])
    with csv_path.open("rb") as stream:
        raw = stream.read(MAX_CSV_BYTES + 1)
    return spec, raw, prepare(spec, raw)
