"""Independently compare pinned SI traces to the original CSV with pandas' parser."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from atomic_model_phase_assessment.dataset import load
from atomic_model_phase_assessment.importer import RAW_PIN

# Independently specified units/columns, not the production converter.
COLUMNS = {
    "pressure": {
        "time_s": ("Time_Pressure (seconds)", 1.0, 0.0),
        "pressure_pa": ("Pressure (kPa)", 1000.0, 0.0),
    },
    "thermal": {
        "time_s": ("Time_Thermocouple (seconds)", 1.0, 0.0),
        **{f"tc{i}_k": (f"Thermo-couple_{i} (degC)", 1.0, 273.15) for i in range(1, 5)},
        "vapor_temperature_k": ("Vapour Temp (degC)", 1.0, 273.15),
        "liquid_temperature_k": ("Liquid Temp (degC)", 1.0, 273.15),
        "surface_temperature_k": ("Surface Temperature (degC)", 1.0, 273.15),
        "heat_flux_w_m2": ("Heat Flux (W/cm^2)", 10000.0, 0.0),
        "htc_w_m2_k": ("HTC (W/cm^2.K)", 10000.0, 0.0),
        "wall_superheat_k": ("Wall Superheat (degC)", 1.0, 0.0),
    },
}


def verify(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    if h.hexdigest() != RAW_PIN:
        raise ValueError("Primary identity mismatch")
    manifest, traces = load()
    expected = {k: {r["source_row"]: r for r in rows} for k, rows in traces.items()}
    counts = dict(pressure=0, thermal=0)
    seen = dict(pressure=0, thermal=0)
    previous = dict(pressure=None, thermal=None)
    errors = {}
    columns = sorted({v[0] for mapping in COLUMNS.values() for v in mapping.values()})
    offset = 0
    for frame in pd.read_csv(path, usecols=columns, chunksize=100000, float_precision="round_trip"):
        for kind, mapping in COLUMNS.items():
            clock = frame[mapping["time_s"][0]].dropna().to_numpy()
            if len(clock):
                if not np.all(np.isfinite(clock)) or np.any(np.diff(clock) <= 0):
                    raise ValueError("Independent clock check failed")
                if previous[kind] is not None and clock[0] <= previous[kind]:
                    raise ValueError("Independent clock continuity failed")
                previous[kind] = float(clock[-1])
                counts[kind] += len(clock)
            selected = [i for i in expected[kind] if offset <= i < offset + len(frame)]
            for index in selected:
                row = expected[kind][index]
                native = frame.iloc[index - offset]
                for field, (column, gain, bias) in mapping.items():
                    value = native[column]
                    if pd.isna(value):
                        if row[field] is not None:
                            raise ValueError("Null mismatch")
                        continue
                    value = float(value) * gain + bias
                    error = abs(value - row[field])
                    errors[kind + "/" + field] = max(errors.get(kind + "/" + field, 0.0), error)
                    if error > max(1e-10, abs(value) * 2e-15):
                        raise ValueError("Independent SI sample mismatch")
                seen[kind] += 1
        offset += len(frame)
    if counts != manifest["native_counts"] or seen != manifest["retained_counts"]:
        raise ValueError("Independent history count mismatch")
    return dict(
        primary_sha256=RAW_PIN,
        native_counts=counts,
        compared_counts=seen,
        max_absolute_errors=errors,
        parser="pandas C round_trip; independent SI formulas",
        validated=False,
        clocks_joined=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = verify(a.input)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
