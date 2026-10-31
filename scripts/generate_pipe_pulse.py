"""Build normalized full-IC input; JSON is accepted by atomic-dynamic-experiment."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from core.status import DomainError
from physics.hydraulics.initialization import equilibrated_config
from scripts.verify_pipe_mesh import configuration
from sdk.dynamic_experiment import validate_dynamic_experiment


def make_spec(
    cells, *, shape="sin2", tight=False, scheme="donor", jacobian_mode="finite_difference"
):
    if type(cells) is not int or not 2 <= cells <= 32:
        raise ValueError("Pulse generator requires 2..32 cells")
    if shape not in ("uniform", "sin2"):
        raise ValueError("Unknown wall pulse shape")
    base = equilibrated_config(configuration(cells, 0.0))
    offset = (
        np.full(cells, 5.0)
        if shape == "uniform"
        else 10 * np.sin(np.pi * (np.arange(cells) + 0.5) / cells) ** 2
    )
    c = replace(
        base,
        enthalpy_transport=scheme,
        initial_wall_temperature_profile_k=tuple(
            float(t + q) for t, q in zip(base.initial_temperature_profile_k, offset)
        ),
    )
    spec = dict(
        schema_version=1,
        name=f"pipe_{shape}_pulse_n{cells}" + ("_tight" if tight else ""),
        engine="pressure-driven-wall-channel",
        duration_s=8.0,
        step_s=0.2,
        hydraulics=c.as_dict(),
        solver=dict(
            method="Radau",
            rtol=1e-8 if tight else 1e-7,
            max_step_s=0.05 if tight else 0.1,
            jacobian_relative_step=1e-5,
        ),
    )
    if scheme == "muscl_mc":
        spec["name"] += "_muscl_mc"
        spec["solver"]["rtol"] = 1e-9 if tight else 1e-8
        spec["solver"]["absolute_tolerances"] = dict(
            mass_kg=1e-12,
            energy_j=1e-5,
            flow_kg_s=1e-9,
            wall_temperature_k=1e-9,
            boundary_mass_kg=1e-12,
            boundary_energy_j=1e-5,
        )
    if jacobian_mode == "analytic":
        spec["solver"].pop("jacobian_relative_step")
        spec["solver"]["jacobian_mode"] = "analytic"
        spec["name"] += "_jac_analytic"
    elif jacobian_mode != "finite_difference":
        raise ValueError("Unknown Jacobian mode")
    return validate_dynamic_experiment(spec)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--shape", choices=["uniform", "sin2"], default="sin2")
    parser.add_argument("--tight", action="store_true")
    parser.add_argument("--scheme", choices=["donor", "muscl_mc"], default="donor")
    parser.add_argument(
        "--jacobian-mode", choices=["finite_difference", "analytic"], default="finite_difference"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        spec = make_spec(
            args.cells,
            shape=args.shape,
            tight=args.tight,
            scheme=args.scheme,
            jacobian_mode=args.jacobian_mode,
        )
        with args.output.open("x") as f:
            json.dump(spec, f, indent=2, allow_nan=False)
            f.write("\n")
    except (ValueError, DomainError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
