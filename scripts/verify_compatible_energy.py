"""Exact frozen-density inertia/dissipation reference, not a full water-channel solution."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from core.solvers import integrate
from physics.hydraulics.dynamic_channel import DynamicWallChannel
from physics.hydraulics.mechanical_energy import kinematics, mechanical_rates
from scripts.verify_pipe_mesh import configuration
from sdk.experiment import _source_fingerprint


def verify():
    cases = []
    for method in ("BDF", "Radau"):
        for sign in (-1.0, 1.0):
            c = DynamicWallChannel(
                replace(
                    configuration(3, 0.0),
                    energy_formulation="compatible_total",
                    left_pressure_pa=7e6,
                    right_pressure_pa=7e6,
                    initial_pressure_pa=7e6,
                    initial_flow_kg_s=sign * 2.0,
                )
            )
            cfg = c.config
            rho = c._state[0] / (cfg.total_volume_m3 / cfg.cells)
            beta = sum(cfg.junction_loss_coefficients) / (
                2 * rho * cfg.area_m2 * sum(cfg.junction_lengths_m)
            )
            exact = sign * 2 / (1 + beta * 2 * 2.0)
            initial = c.initial_kinetic_energy_j
            for rtol in (1e-4, 1e-6, 1e-8):

                def rhs(t, z):
                    y = c._state.copy()
                    y[6:10] = z[0]
                    rates = mechanical_rates(c, y)
                    if max(abs(rates["acceleration"] - rates["acceleration"][0])) > 1e-9:
                        raise ValueError("Uniform inertia reference lost symmetry")
                    return np.array([rates["acceleration"][0], sum(rates["dissipation"])])

                sol = integrate(
                    rhs,
                    np.array([sign * 2.0, 0.0]),
                    (0.0, 2.0),
                    method=method,
                    rtol=rtol,
                    atol=[1e-12, 1e-9],
                    max_step=0.1,
                )
                y = c._state.copy()
                y[6:10] = sol.final[0]
                kinetic = float(sum(kinematics(c, y)["kinetic"]))
                cases.append(
                    dict(
                        method=method,
                        sign=sign,
                        rtol=rtol,
                        flow_error_kg_s=float(abs(sol.final[0] - exact)),
                        energy_residual_j=float(kinetic - initial + sol.final[1]),
                        dissipated_energy_j=float(sol.final[1]),
                        kinetic_energy_j=kinetic,
                        exact_flow_kg_s=float(exact),
                    )
                )
    fine = [c for c in cases if c["rtol"] == 1e-8]
    return dict(
        schema_version=1,
        kind="mechanical_subsystem_verification",
        validated=False,
        plant_referenced=False,
        source_sha256=_source_fingerprint(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=cases,
        passed=all(
            c["flow_error_kg_s"] < 1e-6
            and abs(c["energy_residual_j"]) < 1e-5
            and c["dissipated_energy_j"] > 0
            for c in fine
        ),
        equation="F(t)=F0/(1+beta abs(F0)t); K(t)-K(0)+integral(D dt)=0",
        scope="Frozen density and thermodynamic background, exact uniform inertia/drag subsystem; no analytic solution or physical validation of the coupled heating water channel",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = verify()
    with args.output.open("x") as f:
        json.dump(r, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(passed=r["passed"], fine=[c for c in r["cases"] if c["rtol"] == 1e-8])))
    if not r["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
