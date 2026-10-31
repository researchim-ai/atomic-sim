"""Numerical derivative audit of the piecewise analytic liquid-channel Jacobian."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from physics.hydraulics.dynamic_channel import DynamicWallChannel
from scripts.verify_pipe_mesh import configuration
from sdk.experiment import _source_fingerprint


def verify():
    cases = []
    for scheme in ("donor", "muscl_mc"):
        for sign in (-1.0, 1.0):
            cfg = replace(
                configuration(4, 0.0),
                enthalpy_transport=scheme,
                initial_flow_kg_s=sign,
                initial_temperature_profile_k=(496.0, 498.0, 501.0, 505.0),
                initial_wall_temperature_profile_k=(510.0,) * 4,
            )
            m = DynamicWallChannel(cfg)
            y = m._state.copy()
            jac = m.analytic_jacobian(0.0, y)
            errors = []
            for seed in (11, 27, 71, 93):
                direction = np.random.default_rng(seed).normal(size=y.size) * np.maximum(
                    abs(y), 1.0
                )
                eps = 1e-6
                fd = (
                    m.derivatives(0.0, y + eps * direction)
                    - m.derivatives(0.0, y - eps * direction)
                ) / (2 * eps)
                exact = jac @ direction
                errors.append(float(max(abs(fd - exact) / (1 + np.maximum(abs(fd), abs(exact))))))
            energy = (
                sum(jac[2 * i + 1, :] for i in range(4))
                + sum(cfg.total_heat_capacity_j_k / 4 * jac[13 + i, :] for i in range(4))
                - jac[-1, :]
            )
            mass = sum(jac[2 * i, :] for i in range(4)) - jac[-2, :]
            e = float(max(abs(energy.toarray().ravel())))
            b = float(max(abs(mass.toarray().ravel())))
            cases.append(
                dict(
                    scheme=scheme,
                    flow_sign=sign,
                    size=jac.shape[0],
                    nnz=jac.nnz,
                    directional_scaled_errors=errors,
                    energy_column_identity_max=e,
                    mass_column_identity_max=b,
                    counter_columns_zero=bool(jac[:, -2:].nnz == 0),
                    passed=max(errors) < 1e-4 and e < 1e-6 and b < 1e-12 and jac[:, -2:].nnz == 0,
                )
            )
    return dict(
        schema_version=1,
        kind="jacobian_numerical_verification",
        validated=False,
        plant_referenced=False,
        source_sha256=_source_fingerprint(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=cases,
        passed=all(c["passed"] for c in cases),
        scope="Local directional differences away from limiter/flow ties and exact conservation column identities; nonsmooth points have a selected generalized derivative, not a unique classical derivative",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = verify()
    with args.output.open("x") as f:
        json.dump(r, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(r))
    if not r["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
