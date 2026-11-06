"""Bounded primary reader; pressure and thermal clocks are never zipped together."""

import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

RAW_PIN = "e6bf6caf68072eca8a6f696bc39e31db4249c28fe13feb87a3299bef99c67531"
THERMAL = {
    "time_s": "Time_Thermocouple (seconds)",
    "tc1_k": "Thermo-couple_1 (degC)",
    "tc2_k": "Thermo-couple_2 (degC)",
    "tc3_k": "Thermo-couple_3 (degC)",
    "tc4_k": "Thermo-couple_4 (degC)",
    "vapor_temperature_k": "Vapour Temp (degC)",
    "liquid_temperature_k": "Liquid Temp (degC)",
    "surface_temperature_k": "Surface Temperature (degC)",
    "heat_flux_w_m2": "Heat Flux (W/cm^2)",
    "htc_w_m2_k": "HTC (W/cm^2.K)",
    "wall_superheat_k": "Wall Superheat (degC)",
}
PRESSURE = {"time_s": "Time_Pressure (seconds)", "pressure_pa": "Pressure (kPa)"}
METADATA = [
    "Target Pressure_value (kPa)",
    "Mean Pressure_value (kPa)",
    "Saturation Temperature_value (degC)",
    "CHF_value",
    "CHF_Time_value",
    "HTC_value",
    "T_vapour_mean",
    "T_liquid_mean",
]
PROTOCOL = dict(
    schema_version=1,
    case="B-274",
    nominal_pressure_pa=100000,
    pressure_native_stride=2500,
    thermal_native_stride=300,
    include_first_and_last_sample=True,
    filter_on_error_or_phase=False,
    clocks_joined=False,
    heat_flux_includes_negative_values=True,
    whole_heating_and_cooling_history_retained=True,
    max_primary_rows=10_000_000,
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(1024 * 1024):
            h.update(data)
    return h.hexdigest()


def number(raw):
    if raw in ("", "nan", "NaN"):
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("Nonfinite primary number")
    return value


def convert(row, mapping):
    result = {}
    for key, column in mapping.items():
        value = number(row[column])
        if value is not None:
            if key.endswith("_k") and key not in ("wall_superheat_k", "htc_w_m2_k"):
                value += 273.15
            elif key in ("heat_flux_w_m2", "htc_w_m2_k"):
                value *= 10000.0
            elif key == "pressure_pa":
                value *= 1000.0
        result[key] = value
    return result


def decode(rows):
    samples = {"pressure": [], "thermal": []}
    counts = {"pressure": 0, "thermal": 0}
    previous = {"pressure": None, "thermal": None}
    last = {}
    metadata = {}
    stats = {}
    for row_index, row in enumerate(rows):
        if row_index >= PROTOCOL["max_primary_rows"]:
            raise ValueError("Primary row budget exceeded")
        for name in METADATA:
            value = number(row[name])
            if value is not None:
                if name in metadata and metadata[name] != value:
                    raise ValueError("Conflicting primary scalar")
                metadata[name] = value
        for kind, mapping, stride in [("pressure", PRESSURE, 2500), ("thermal", THERMAL, 300)]:
            time = number(row[mapping["time_s"]])
            if time is None:
                if any(
                    number(row[column]) is not None
                    for key, column in mapping.items()
                    if key != "time_s"
                ):
                    raise ValueError("Sensor value without its native clock")
                continue
            point = convert(row, mapping)
            if previous[kind] is not None and time <= previous[kind]:
                raise ValueError("Nonmonotone native clock")
            if any(point[key] is None for key in mapping if key != "htc_w_m2_k"):
                raise ValueError("Missing required native-clock variable")
            previous[kind] = time
            point["source_row"] = row_index
            for key, value in point.items():
                if key in ("source_row", "time_s") or value is None:
                    continue
                stat = stats.setdefault(kind + "/" + key, dict(min=value, max=value))
                stat["min"] = min(stat["min"], value)
                stat["max"] = max(stat["max"], value)
            if counts[kind] % stride == 0:
                samples[kind].append(point)
            counts[kind] += 1
            last[kind] = point
    for kind in samples:
        if counts[kind] < 2:
            raise ValueError("Missing native history")
        if samples[kind][-1]["source_row"] != last[kind]["source_row"]:
            samples[kind].append(last[kind])
    if set(metadata) != set(METADATA):
        raise ValueError("Missing primary case metadata")
    return dict(samples=samples, native_counts=counts, scalars=metadata, ranges=stats)


def write_csv(path, rows):
    with Path(path).open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)


def import_primary(path, output):
    path = Path(path)
    output = Path(output)
    if sha(path) != RAW_PIN:
        raise ValueError("Primary B-274 byte identity mismatch")
    if output.exists():
        raise FileExistsError(output)
    with path.open(encoding="utf-8", newline="") as stream:
        decoded = decode(csv.DictReader(stream))
    output.mkdir(parents=True)
    try:
        for kind, rows in decoded["samples"].items():
            write_csv(output / (kind + ".csv.gz"), rows)
        result = dict(
            schema_version=1,
            source_doi="10.5281/zenodo.22234711",
            case="B-274",
            primary_member="ned3-013_SubatmosphericBoiling/A-FlatCu/100kPa/Boiling-274_csv_data.csv",
            primary_sha256=RAW_PIN,
            primary_bytes=path.stat().st_size,
            primary_member_crc32=3284962502,
            whole_zip_checksum_verified=False,
            protocol=PROTOCOL,
            parser_sha256=sha(__file__),
            native_counts=decoded["native_counts"],
            scalars=decoded["scalars"],
            ranges=decoded["ranges"],
            retained_counts={k: len(v) for k, v in decoded["samples"].items()},
            files={name: sha(output / name) for name in ["pressure.csv.gz", "thermal.csv.gz"]},
            data_license="CC-BY-4.0",
            validated=False,
            plant_referenced=False,
        )
        (output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result
    except Exception:
        # Never leave a manifest that appears to certify partial processing.
        manifest = output / "manifest.json"
        if manifest.exists():
            manifest.unlink()
        raise
