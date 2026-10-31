"""Directional and nonlinear conservation-identity audit for compatible total energy."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from physics.hydraulics.dynamic_channel import DynamicWallChannel
from physics.hydraulics.mechanical_energy import kinetic_gradient_hessian
from scripts.verify_pipe_mesh import configuration
from sdk.experiment import _source_fingerprint


def verify():
    cases = []
    for scheme in ("donor", "muscl_mc"):
        for sign in (-1.0, 1.0):
            cfg = replace(
                configuration(3, 0.0),
                energy_formulation="compatible_total",
                enthalpy_transport=scheme,
                initial_flow_kg_s=sign,
                initial_temperature_profile_k=(496.0, 500.0, 505.0),
                initial_wall_temperature_profile_k=(510.0,) * 3,
            )
            m = DynamicWallChannel(cfg)
            y = m._state.copy()
            y[6:10] = sign * np.array([0.8, 1.1, 0.9, 1.2])
            jac = m.analytic_jacobian(0.0, y)
            errors = []
            for seed in (14, 44, 91):
                v = np.random.default_rng(seed).normal(size=len(y)) * np.maximum(abs(y), 1.0)
                eps = 1e-6
                numeric = (m.derivatives(0.0, y + eps * v) - m.derivatives(0.0, y - eps * v)) / (
                    2 * eps
                )
                exact = jac @ v
                errors.append(
                    float(max(abs(numeric - exact) / (1 + np.maximum(abs(numeric), abs(exact)))))
                )
            gradient, hessian = kinetic_gradient_hessian(m, y)
            for i in range(3):
                gradient[2 * i + 1] += 1.0
                gradient[10 + i] += cfg.total_heat_capacity_j_k / 3
            rhs = m.derivatives(0.0, y)
            identity = jac.T @ gradient + hessian @ rhs - jac[-1, :].toarray().ravel()
            cases.append(
                dict(
                    scheme=scheme,
                    sign=sign,
                    size=len(y),
                    nnz=jac.nnz,
                    directional_scaled_errors=errors,
                    full_energy_rate_residual_w=float(gradient @ rhs - rhs[-1]),
                    nonlinear_energy_column_identity=float(max(abs(identity))),
                    zero_integral_columns=bool(jac[:, -5:].nnz == 0),
                )
            )
    return dict(
        schema_version=1,
        kind="compatible_energy_jacobian_verification",
        source_sha256=_source_fingerprint(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        validated=False,
        plant_referenced=False,
        cases=cases,
        passed=all(
            max(c["directional_scaled_errors"]) < 1e-4
            and abs(c["full_energy_rate_residual_w"]) < 1e-7
            and c["nonlinear_energy_column_identity"] < 1e-6
            and c["zero_integral_columns"]
            for c in cases
        ),
        identity="J.T grad(E) + Hess(E) f = grad(boundary total-enthalpy power)",
        scope="Piecewise local derivatives away from sign/limiter switches; reduced discrete model, not physical validation",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    r = verify()
    with args.output.open("x") as f:
        json.dump(r, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(r))
    if not r["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
