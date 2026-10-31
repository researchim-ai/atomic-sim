"""Generate explicit total-energy initial conditions and strict absolute budgets."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from core.status import DomainError
from physics.hydraulics.dynamic_channel import DynamicConfig
from physics.hydraulics.total_energy_initialization import equilibrated_total_config
from sdk.dynamic_experiment import validate_dynamic_experiment


def make_spec(case="wall_pulse", *, tight=False, sign=1, cells=3, transport="donor", heat_w=0.0):
    if case not in ("wall_pulse", "coastdown") or sign not in (-1, 1):
        raise ValueError("Unknown case/sign")
    c = DynamicConfig.from_pipe(
        cells=cells,
        total_length_m=3.0,
        total_loss_coefficient=30.0,
        area_m2=0.001,
        left_pressure_pa=7.02e6 if case == "wall_pulse" and sign > 0 else 7e6,
        right_pressure_pa=7.02e6 if case == "wall_pulse" and sign < 0 else 7e6,
        left_temperature_k=500.0,
        right_temperature_k=500.0,
        initial_pressure_pa=7.01e6 if case == "wall_pulse" else 7e6,
        initial_temperature_k=500.0,
        total_heat_capacity_j_k=8000.0,
        total_ua_w_k=1000.0,
        heat_w=heat_w,
        energy_formulation="compatible_total",
        enthalpy_transport=transport,
    )
    if case == "wall_pulse":
        c = equilibrated_total_config(c, wall_offset_k=2.0)
    else:
        c = replace(c, initial_flow_kg_s=2.0 * sign)
    return validate_dynamic_experiment(
        dict(
            schema_version=1,
            name="total_energy_"
            + case
            + ("_negative" if sign < 0 else "")
            + ("_tight" if tight else ""),
            engine="compatible-total-energy-channel",
            duration_s=8.0 if case == "wall_pulse" else 2.0,
            step_s=0.2 if case == "wall_pulse" else 0.05,
            hydraulics=c.as_dict(),
            solver=dict(
                method="Radau",
                rtol=1e-9 if tight else 1e-8,
                max_step_s=0.05 if tight else 0.1,
                jacobian_mode="analytic",
                absolute_tolerances=dict(
                    mass_kg=1e-12,
                    energy_j=1e-5,
                    flow_kg_s=1e-9,
                    wall_temperature_k=1e-9,
                    boundary_mass_kg=1e-12,
                    boundary_energy_j=1e-5,
                    mechanical_work_j=1e-8 if tight else 1e-7,
                ),
            ),
            budgets=dict(
                mass_relative=1e-6,
                energy_relative=1e-5,
                energy_absolute_j=1e-3,
                mechanical_absolute_j=1e-4,
            ),
        )
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["wall_pulse", "coastdown"], default="wall_pulse")
    p.add_argument("--tight", action="store_true")
    p.add_argument("--sign", type=int, choices=[-1, 1], default=1)
    p.add_argument("--cells", type=int, default=3)
    p.add_argument("--transport", choices=["donor", "muscl_mc"], default="donor")
    p.add_argument("--heat-w", type=float, default=0.0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        spec = make_spec(
            args.case,
            tight=args.tight,
            sign=args.sign,
            cells=args.cells,
            transport=args.transport,
            heat_w=args.heat_w,
        )
        with args.output.open("x") as f:
            json.dump(spec, f, indent=2, allow_nan=False)
            f.write("\n")
    except (ValueError, DomainError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
