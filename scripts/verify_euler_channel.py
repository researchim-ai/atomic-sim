"""Analytic drag, flux spectra, truncation and linearized-mode Euler audits.

This is numerical verification of a component, not nonlinear spatial/acoustic
or experimental qualification. Criteria live in a pre-existing input budget.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from physics.hydraulics.euler_channel import EulerConfig, EulerLiquidChannel
from sdk.experiment import _source_fingerprint


def friction():
    rows = []
    for method in ("BDF", "Radau"):
        for sign in (-1, 1):
            errors = []
            for tolerance in (1e-6, 1e-8, 1e-10):
                m = EulerLiquidChannel(
                    EulerConfig(
                        cells=3, boundary="periodic", initial_velocity_m_s=sign * 2, total_ua_w_k=0
                    )
                )
                m.enable_continuation(2.0)
                maximum = 0.0
                for t in np.linspace(0.1, 2, 20):
                    row = m.advance(
                        float(t) - m.time_s, method=method, rtol=tolerance, max_step_s=0.1
                    )
                    exact = sign * 2 / (1 + 10 * t)
                    maximum = max(
                        maximum,
                        abs(row["max_velocity_m_s"] - exact),
                        abs(row["min_velocity_m_s"] - exact),
                    )
                errors.append(maximum)
                rows.append(
                    dict(
                        method=method,
                        sign=sign,
                        rtol=tolerance,
                        velocity_max_error_m_s=maximum,
                        total_energy_residual_j=abs(row["energy_residual_j"]),
                        momentum_residual_kg_m_s=abs(row["momentum_residual_kg_m_s"]),
                    )
                )
            rows[-1]["refinement_improved"] = bool(errors[-1] < errors[0])
    return rows


def truncation():
    rows = []
    nodes, weights = np.polynomial.legendre.leggauss(12)
    for n in (8, 16, 32):
        m = EulerLiquidChannel(
            EulerConfig(cells=n, boundary="periodic", total_ua_w_k=0, total_loss_coefficient=0)
        )

        def inventory(x):
            phase = 2 * np.pi * x / m.config.total_length_m
            return m.inventory(
                7e6 + 1e4 * np.sin(phase), 500 + 0.2 * np.cos(phase), 1 + 0.05 * np.sin(phase)
            )

        q = []
        for i in range(n):
            q.append(
                sum(w * inventory((i + 0.5 + 0.5 * z) * m.dx) for z, w in zip(nodes, weights)) * 0.5
            )
        state = m._state.copy()
        state[: 3 * n] = np.array(q).ravel()
        rhs = m.derivatives(0, state)[: 3 * n].reshape(n, 3)
        exact = []
        for i in range(n):
            # Physical exact face flux is evaluated directly on the prescribed
            # smooth primitive solution, not through the numerical flux.
            def flux(x):
                phase = 2 * np.pi * x / m.config.total_length_m
                p = 7e6 + 1e4 * np.sin(phase)
                t = 500 + 0.2 * np.cos(phase)
                u = 1 + 0.05 * np.sin(phase)
                from physics.materials.iapws import properties_pt

                s = properties_pt(float(p), float(t))
                a = m.config.area_m2
                return np.array(
                    [
                        s.rho_kg_m3 * u * a,
                        (s.rho_kg_m3 * u * u + p) * a,
                        s.rho_kg_m3 * (s.u_j_kg + 0.5 * u * u) * u * a + p * u * a,
                    ]
                )

            exact.append(flux(i * m.dx) - flux((i + 1) * m.dx))
        error = np.mean(abs(rhs - np.array(exact)), axis=0) / m.dx
        rows.append(dict(cells=n, l1_residual_per_length=error.tolist()))
    for a, b in zip(rows, rows[1:]):
        b["orders"] = np.log2(
            np.array(a["l1_residual_per_length"]) / b["l1_residual_per_length"]
        ).tolist()
    return rows


def linear_modes():
    rows = []
    n = 8
    base = EulerLiquidChannel(
        EulerConfig(
            cells=n,
            boundary="periodic",
            initial_velocity_m_s=0.5,
            total_ua_w_k=0,
            total_loss_coefficient=0,
        )
    )
    q0 = base._state[:3]
    sound = base.cell(q0)[3]
    scales = np.array([q0[0], q0[0] * sound, q0[2]])
    matrix = base.flux(q0, True)[1] * base.dx
    scaled = matrix * scales[None, :] / scales[:, None]
    eigenvalues, eigenvectors = np.linalg.eig(scaled)
    vector = eigenvectors[:, np.argmax(eigenvalues.real)].real
    vector /= vector[0]
    theta = 2 * np.pi / n
    operator = -1j * np.sin(
        theta
    ) / base.dx * scaled - base.config.wave_speed_bound_m_s / base.dx * (
        1 - np.cos(theta)
    ) * np.eye(3)
    phase = np.exp(1j * 2 * np.pi * (np.arange(n) + 0.5) / n)
    duration = 1e-4
    for amplitude in (3e-4, 1.5e-4, 7.5e-5):
        prescribed = q0 + amplitude * np.real(phase[:, None] * vector[None, :]) * scales
        states = [base.cell(q) for q in prescribed]
        c = replace(
            base.config,
            initial_pressure_profile_pa=tuple(s[0].p_pa for s in states),
            initial_temperature_profile_k=tuple(s[0].T_k for s in states),
            initial_velocity_profile_m_s=tuple(s[1] for s in states),
        )
        m = EulerLiquidChannel(c)
        m.advance(duration, method="Radau", rtol=1e-11, max_step_s=1e-5)
        expected = (
            q0
            + amplitude
            * np.real(phase[:, None] * (expm(operator * duration) @ vector)[None, :])
            * scales
        )
        actual = m._state[: 3 * n].reshape(n, 3)
        error = float(np.max(abs(actual - expected) / scales))
        rows.append(
            dict(
                amplitude=amplitude,
                max_scaled_nonlinear_vs_linear_error=error,
                relative_to_amplitude=error / amplitude,
            )
        )
    for a, b in zip(rows, rows[1:]):
        b["amplitude_order"] = float(
            np.log2(
                a["max_scaled_nonlinear_vs_linear_error"]
                / b["max_scaled_nonlinear_vs_linear_error"]
            )
        )
    return dict(
        duration_s=duration,
        continuum_eigenvalues_m_s=sorted(eigenvalues.real.tolist()),
        expected_eigenvalues_m_s=[0.5 - sound, 0.5, 0.5 + sound],
        rows=rows,
        scope="Linearized semidiscrete Fourier mode, including numerical diffusion; not acoustic spatial qualification",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budget", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    budget = json.loads(args.budget.read_text())
    drag = friction()
    space = truncation()
    wave = linear_modes()
    checks = dict(
        friction_finest=all(
            r["velocity_max_error_m_s"] <= budget["friction_velocity_abs_m_s"]
            for r in drag
            if r["rtol"] == 1e-10
        ),
        friction_improves=all(r.get("refinement_improved", True) for r in drag),
        conserved_energy=all(r["total_energy_residual_j"] <= budget["energy_abs_j"] for r in drag),
        conserved_momentum=all(
            r["momentum_residual_kg_m_s"] <= budget["momentum_abs_kg_m_s"] for r in drag
        ),
        truncation_order=all(
            min(r["orders"]) >= budget["smooth_truncation_min_order"] for r in space[1:]
        ),
        linear_mode=wave["rows"][-1]["relative_to_amplitude"]
        <= budget["linear_mode_relative_error"],
        nonlinear_remainder=all(
            r["amplitude_order"] >= budget["nonlinear_remainder_min_order"]
            for r in wave["rows"][1:]
        ),
    )
    report = dict(
        passed=all(checks.values()),
        checks=checks,
        drag=drag,
        truncation=space,
        linear_mode=wave,
        budget_sha256=hashlib.sha256(args.budget.read_bytes()).hexdigest(),
        source_sha256=_source_fingerprint(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        validated=False,
        plant_referenced=False,
        full_nonlinear_spatial_qualification=False,
        acoustic_qualified=False,
    )
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(passed=report["passed"], checks=checks)))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
