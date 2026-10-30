"""Exact signed quadratic-drag coastdown, both implicit solvers."""

import argparse
import hashlib
import json
from pathlib import Path

from physics.hydraulics.dynamic_channel import DynamicConfig, DynamicWallChannel
from sdk.experiment import _source_fingerprint


def verify():
    cases = []
    for method in ("BDF", "Radau"):
        for sign in (-1.0, 1.0):
            for rtol in (1e-5, 1e-7, 1e-9):
                c = DynamicWallChannel(
                    DynamicConfig(
                        1,
                        0.002,
                        7e6,
                        7e6,
                        500.0,
                        500.0,
                        7e6,
                        500.0,
                        0.001,
                        1.0,
                        10.0,
                        8000.0,
                        1000.0,
                        0.0,
                        sign * 2.0,
                    )
                )
                rho = c.network.volumes["0"].thermo.rho_kg_m3
                k = c.config.loss_coefficient / (
                    2 * rho * c.config.area_m2 * c.config.junction_length_m
                )
                exact = sign * 2.0 / (1 + k * 2.0 * 2.0)
                c.enable_continuation(2.0)
                row = c.advance(2.0, rtol=rtol, method=method, max_step_s=0.05)
                cases.append(
                    dict(
                        method=method,
                        initial_flow_kg_s=sign * 2.0,
                        rtol=rtol,
                        exact_final_flow_kg_s=exact,
                        flow_error_kg_s=abs(row["inlet_mass_flow_kg_s"] - exact),
                        pressure_error_pa=abs(row["min_pressure_pa"] - 7e6),
                        mass_residual_kg=abs(row["mass_residual_kg"]),
                        energy_residual_j=abs(row["energy_residual_j"]),
                    )
                )
    fine = [x for x in cases if x["rtol"] == 1e-9]
    coarse = [x for x in cases if x["rtol"] == 1e-5]
    improved = all(f["flow_error_kg_s"] < c["flow_error_kg_s"] for f, c in zip(fine, coarse))
    return dict(
        schema_version=1,
        kind="numerical_verification",
        validated=False,
        plant_referenced=False,
        source_sha256=_source_fingerprint(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        equation="dF/dt=-k F abs(F); F(t)=F0/(1+k abs(F0)t), k=K/(2 rho A L)",
        horizon_s=2.0,
        scope="Equal-temperature/pressure reservoirs, uniform fluid, no heat; frictional mechanical energy is not in the internal-energy inventory",
        cases=cases,
        coarse_to_fine_improvement=improved,
        passed=improved
        and all(
            x["flow_error_kg_s"] < 2e-7
            and x["pressure_error_pa"] < 0.01
            and x["energy_residual_j"] < 0.001
            for x in fine
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = verify()
    with args.output.open("x") as f:
        json.dump(r, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(passed=r["passed"], fine=[x for x in r["cases"] if x["rtol"] == 1e-9])))
    if not r["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
