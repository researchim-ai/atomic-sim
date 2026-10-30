"""Численные эталоны: аналитическое испарение и пространственный порядок канала."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from physics.hydraulics.isobaric import ChannelConfig, IsobaricChannel, isobaric_state
from physics.materials.iapws import saturation_properties
from sdk.experiment import _source_fingerprint


def verify():
    pressure = 7e6
    liquid, vapor = saturation_properties(pressure)
    h0 = liquid.h_j_kg + 0.1 * (vapor.h_j_kg - liquid.h_j_kg)
    b = (vapor.v_m3_kg - liquid.v_m3_kg) / (vapor.h_j_kg - liquid.h_j_kg)
    v0 = 1 / isobaric_state(pressure, h0).density
    volume, heat, duration = 0.01, 2e5, 2.0
    vt = v0 * np.exp(heat * b / volume * duration)
    ht = h0 + (vt - v0) / b
    analytic = []
    for rtol in (1e-3, 1e-5, 1e-7):
        c = IsobaricChannel(ChannelConfig(1, volume, pressure, h0, 0.0, heat))
        start = time.monotonic()
        row = c.advance(duration, rtol=rtol, max_step_s=2.0)
        analytic.append(
            {
                "rtol": rtol,
                "wall_time_s": time.monotonic() - start,
                "enthalpy_error_j_kg": float(abs(row["outlet_enthalpy_j_kg"] - ht)),
                "mass_error_kg": float(abs(row["stored_mass_kg"] - volume / vt)),
                "energy_residual_j": float(abs(row["energy_residual_j"])),
            }
        )
    spatial = []
    inlet = liquid.h_j_kg + 0.05 * (vapor.h_j_kg - liquid.h_j_kg)
    for cells in (4, 8, 16):
        c = IsobaricChannel(ChannelConfig(cells, 0.004, pressure, inlet, 1.0, 1.5e5))
        c.enable_continuation(20.0)
        start = time.monotonic()
        row = c.advance(20.0, rtol=1e-8, max_step_s=0.5)
        centers = (np.arange(cells) + 0.5) / cells
        reference = inlet + 1.5e5 * centers
        spatial.append(
            {
                "cells": cells,
                "duration_s": 20.0,
                "wall_time_s": time.monotonic() - start,
                "mean_enthalpy_error_j_kg": float(np.mean(abs(c._state[:cells] - reference))),
                "outlet_error_j_kg": float(abs(row["outlet_enthalpy_j_kg"] - inlet - 1.5e5)),
                "mass_residual_kg": float(row["mass_residual_kg"]),
                "energy_residual_j": float(row["energy_residual_j"]),
            }
        )
    errors = [item["enthalpy_error_j_kg"] for item in analytic]
    ratios = [
        spatial[i]["mean_enthalpy_error_j_kg"] / spatial[i + 1]["mean_enthalpy_error_j_kg"]
        for i in range(2)
    ]
    checks = {
        "temporal_error_decreases": errors[2] < errors[1] < errors[0],
        "finest_temporal_error_j_kg_below_1": errors[-1] < 1.0,
        "first_order_spatial": all(1.9 < ratio < 2.1 for ratio in ratios),
        "outlet_steady_balance": all(item["outlet_error_j_kg"] < 2.0 for item in spatial),
    }
    return {
        "schema_version": 1,
        "kind": "numerical_verification",
        "validated": False,
        "plant_referenced": False,
        "source_sha256": _source_fingerprint(),
        "pressure_pa": pressure,
        "analytic_case": {
            "volume_m3": volume,
            "heat_w": heat,
            "duration_s": duration,
            "initial_enthalpy_j_kg": h0,
        },
        "analytic": analytic,
        "spatial": spatial,
        "spatial_error_ratios": ratios,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify()
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": result["passed"], "checks": result["checks"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
