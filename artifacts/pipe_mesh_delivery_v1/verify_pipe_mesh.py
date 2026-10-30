"""Steady spatial convergence against an independent pressure/enthalpy BVP.

This verifies a steady operator and geometry scaling, not transient spatial
accuracy, acoustic waves or total mechanical energy.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from physics.hydraulics.dynamic_channel import DynamicConfig, DynamicWallChannel
from physics.materials.iapws import properties_ph, properties_pt
from sdk.experiment import _source_fingerprint


def configuration(cells, heat):
    return DynamicConfig.from_pipe(
        cells=cells,
        total_length_m=3.0,
        total_loss_coefficient=30.0,
        left_pressure_pa=7.02e6,
        right_pressure_pa=7e6,
        left_temperature_k=500.0,
        right_temperature_k=500.0,
        initial_pressure_pa=7.01e6,
        initial_temperature_k=500.0,
        area_m2=0.001,
        total_heat_capacity_j_k=8000.0,
        total_ua_w_k=1000.0,
        heat_w=heat,
    )


def continuous_reference(config, *, tolerance=1e-12):
    c = config
    if c.left_pressure_pa <= c.right_pressure_pa:
        raise ValueError("Steady mesh benchmark requires forward pressure drive")
    inlet = properties_pt(c.left_pressure_pa, c.left_temperature_k)
    total_k = sum(c.junction_loss_coefficients)
    if total_k <= 0:
        raise ValueError("Positive total resistance required")
    estimate = c.area_m2 * np.sqrt(
        2 * inlet.rho_kg_m3 * (c.left_pressure_pa - c.right_pressure_pa) / total_k
    )

    def integrate(flow, dense=False):
        def rhs(x, p):
            h = inlet.h_j_kg + c.heat_w * x / flow
            state = properties_ph(float(p[0]), h)
            if state.region != 1:
                raise ValueError("Reference left liquid region")
            return [-total_k * flow**2 / (2 * state.rho_kg_m3 * c.area_m2**2)]

        result = solve_ivp(
            rhs,
            (0.0, 1.0),
            [c.left_pressure_pa],
            method="DOP853",
            rtol=tolerance,
            atol=1e-7,
            dense_output=dense,
        )
        if not result.success:
            raise RuntimeError(result.message)
        return result

    flow = brentq(
        lambda f: integrate(f).y[0, -1] - c.right_pressure_pa,
        0.5 * estimate,
        2 * estimate,
        xtol=1e-12,
    )
    return float(flow), integrate(flow, True), float(inlet.h_j_kg)


def discrete_steady(config):
    """Scalar recurrence from the actual donor-pressure loss law, no time march."""
    c = config
    n = c.cells
    inlet = properties_pt(c.left_pressure_pa, c.left_temperature_k)
    estimate = c.area_m2 * np.sqrt(
        2
        * inlet.rho_kg_m3
        * (c.left_pressure_pa - c.right_pressure_pa)
        / sum(c.junction_loss_coefficients)
    )

    def profile(flow):
        p = c.left_pressure_pa
        h = inlet.h_j_kg
        states = []
        for j, loss in enumerate(c.junction_loss_coefficients):
            donor = properties_ph(p, h)
            p -= loss * flow**2 / (2 * donor.rho_kg_m3 * c.area_m2**2)
            if j < n:
                h = inlet.h_j_kg + c.heat_w * (j + 1) / (flow * n)
                state = properties_ph(p, h)
                if state.region != 1:
                    raise ValueError("Discrete steady state left liquid region")
                states.append(state)
        return p, states

    flow = brentq(
        lambda f: profile(f)[0] - c.right_pressure_pa, 0.5 * estimate, 2 * estimate, xtol=1e-12
    )
    final_pressure, states = profile(flow)
    model = DynamicWallChannel(c)
    y = model._state.copy()
    for i, state in enumerate(states):
        mass = state.rho_kg_m3 * c.total_volume_m3 / n
        y[2 * i : 2 * i + 2] = [mass, mass * state.u_j_kg]
        y[3 * n + 1 + i] = state.T_k + c.heat_w / c.total_ua_w_k
    y[2 * n : 3 * n + 1] = flow
    derivative = model.derivatives(0.0, y)
    # Evaluate the implemented operator at the constructed equilibrium. No
    # fictitious history or zeroed global inventory residual is claimed here.
    residual = dict(
        mass_rate_kg_s=float(max(abs(derivative[: 2 * n : 2]))),
        fluid_energy_rate_w=float(max(abs(derivative[1 : 2 * n : 2]))),
        acceleration_kg_s2=float(max(abs(derivative[2 * n : 3 * n + 1]))),
        wall_temperature_rate_k_s=float(max(abs(derivative[3 * n + 1 : 4 * n + 1]))),
        boundary_pressure_pa=float(abs(final_pressure - c.right_pressure_pa)),
    )
    return float(flow), states, residual


def verify():
    results = []
    for heat in (0.0, 3000.0):
        ref_config = configuration(4, heat)
        flow, solution, h0 = continuous_reference(ref_config)
        strict_flow, strict_solution, _ = continuous_reference(ref_config, tolerance=2.3e-14)
        reference_difference = dict(
            flow_kg_s=abs(flow - strict_flow),
            pressure_pa=float(
                max(
                    abs(
                        solution.sol(np.linspace(0, 1, 101))[0]
                        - strict_solution.sol(np.linspace(0, 1, 101))[0]
                    )
                )
            ),
        )
        rows = []
        for n in (4, 8, 16):
            c = configuration(n, heat)
            computed, states, residual = discrete_steady(c)
            positions = (np.arange(n) + 0.5) / n
            reference_pressures = solution.sol(positions)[0]
            reference_states = [
                properties_ph(float(p), h0 + heat * float(x) / flow)
                for p, x in zip(reference_pressures, positions)
            ]
            rows.append(
                dict(
                    cells=n,
                    total_length_m=sum(c.junction_lengths_m),
                    total_loss_coefficient=sum(c.junction_loss_coefficients),
                    total_volume_m3=c.total_volume_m3,
                    flow_error_kg_s=abs(computed - flow),
                    pressure_max_error_pa=float(
                        max(abs(s.p_pa - p) for s, p in zip(states, reference_pressures))
                    ),
                    enthalpy_max_error_j_kg=float(
                        max(abs(s.h_j_kg - r.h_j_kg) for s, r in zip(states, reference_states))
                    ),
                    fluid_temperature_max_error_k=float(
                        max(abs(s.T_k - r.T_k) for s, r in zip(states, reference_states))
                    ),
                    wall_temperature_max_error_k=float(
                        max(abs(s.T_k - r.T_k) for s, r in zip(states, reference_states))
                    ),
                    operator_residual=residual,
                )
            )
        metrics = ["flow_error_kg_s", "pressure_max_error_pa"]
        if heat:
            metrics += ["enthalpy_max_error_j_kg", "fluid_temperature_max_error_k"]
        orders = {
            key: [float(np.log2(rows[i][key] / rows[i + 1][key])) for i in range(2)]
            for key in metrics
        }
        residual_ok = all(
            r["operator_residual"]["mass_rate_kg_s"] < 1e-10
            and r["operator_residual"]["fluid_energy_rate_w"] < 0.001
            and r["operator_residual"]["acceleration_kg_s2"] < 1e-5
            and r["operator_residual"]["wall_temperature_rate_k_s"] < 1e-7
            for r in rows
        )
        converged = all(order > 0.8 for seq in orders.values() for order in seq)
        results.append(
            dict(
                heat_w=heat,
                reference_flow_kg_s=flow,
                reference_difference=reference_difference,
                rows=rows,
                observed_orders=orders,
                passed=converged and residual_ok and reference_difference["pressure_pa"] < 0.001,
            )
        )
    return dict(
        schema_version=1,
        kind="steady_spatial_verification",
        validated=False,
        plant_referenced=False,
        source_sha256=_source_fingerprint(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_equations="dh/dx=P/F; dp/dx=-K_total F²/(2 rho(p,h) A²), x in [0,1]",
        scope="Steady first-order donor model; not transient mesh, acoustic or mechanical-energy qualification",
        cases=results,
        passed=all(r["passed"] for r in results),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify()
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
