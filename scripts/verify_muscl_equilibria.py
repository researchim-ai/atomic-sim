"""Reproducible production-RHS audit of discrete MUSCL total-energy equilibria."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from physics.hydraulics.dynamic_channel import DynamicWallChannel
from physics.hydraulics.total_energy_initialization import equilibrated_total_config
from scripts.verify_pipe_mesh import configuration
from sdk.experiment import _source_fingerprint


def verify():
    cases = []
    for n in (3, 8, 16):
        for heat in (0.0, 3000.0):
            for sign in (-1, 1):
                cfg = replace(
                    configuration(n, heat),
                    energy_formulation="compatible_total",
                    enthalpy_transport="muscl_mc",
                    left_pressure_pa=7.02e6 if sign > 0 else 7e6,
                    right_pressure_pa=7e6 if sign > 0 else 7.02e6,
                    left_temperature_k=500.0 if sign > 0 else 490.0,
                    right_temperature_k=490.0 if sign > 0 else 500.0,
                    junction_loss_coefficients=tuple(np.linspace(1, 4, n + 1)[::sign]),
                )
                model = DynamicWallChannel(equilibrated_total_config(cfg))
                rhs = model.derivatives(0, model._state)
                measured = dict(
                    mass_rate_kg_s=float(max(abs(rhs[: 2 * n : 2]))),
                    internal_energy_rate_w=float(max(abs(rhs[1 : 2 * n : 2]))),
                    flow_rate_kg_s2=float(max(abs(rhs[2 * n : 3 * n + 1]))),
                    wall_rate_k_s=float(max(abs(rhs[3 * n + 1 : 4 * n + 1]))),
                )
                tolerances = dict(
                    mass_rate_kg_s=1e-12,
                    internal_energy_rate_w=1e-3,
                    flow_rate_kg_s2=1e-6,
                    wall_rate_k_s=1e-9,
                )
                cases.append(
                    dict(
                        cells=n,
                        heat_w=heat,
                        sign=sign,
                        flow_kg_s=model.config.initial_flow_kg_s,
                        measured=measured,
                        tolerances=tolerances,
                        passed=all(v <= tolerances[k] for k, v in measured.items()),
                    )
                )
    return dict(
        passed=all(c["passed"] for c in cases),
        cases=cases,
        source_sha256=_source_fingerprint(),
        validated=False,
        plant_referenced=False,
        scope="Discrete steady production RHS, not mesh accuracy or experimental validation",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    report = verify()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(passed=report["passed"], cases=len(report["cases"]))))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
