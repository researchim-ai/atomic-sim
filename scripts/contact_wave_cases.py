"""Analytic isobaric entropy/contact waves for the conservative IF97 Euler PDE."""

from dataclasses import replace

import numpy as np

from physics.hydraulics.euler_channel import EulerConfig, EulerLiquidChannel


def exact_averages(model, time_s, *, velocity=1.0, amplitude_k=0.01, shape="smooth"):
    nodes, weights = np.polynomial.legendre.leggauss(12)
    length = model.config.total_length_m

    def temperature(x):
        phase = (x - velocity * time_s) % length
        return 500 + amplitude_k * (
            np.sin(2 * np.pi * phase / length)
            if shape == "smooth"
            else (1 if phase < length / 2 else -1)
        )

    rows = []
    for i in range(model.config.cells):
        if shape == "step":
            # Exact integration of the piecewise constant entropy wave.
            begin, end = i * model.dx, (i + 1) * model.dx
            cuts = [begin, end]
            for k in range(-2, 3):
                for edge in (0, length / 2):
                    x = velocity * time_s + edge + k * length
                    if begin < x < end:
                        cuts.append(x)
            cuts = sorted(cuts)
            total = np.zeros(3)
            for a, b in zip(cuts, cuts[1:]):
                total += (
                    (b - a) / model.dx * model.inventory(7e6, temperature((a + b) / 2), velocity)
                )
            rows.append(total)
        else:
            rows.append(
                sum(
                    w * model.inventory(7e6, temperature((i + 0.5 + 0.5 * z) * model.dx), velocity)
                    for z, w in zip(nodes, weights)
                )
                * 0.5
            )
    return np.array(rows)


def configuration(cells, *, velocity=1.0, flux="hllc", amplitude_k=0.01, shape="smooth"):
    base = EulerConfig(
        cells=cells,
        numerical_flux=flux,
        boundary="periodic",
        initial_velocity_m_s=velocity,
        total_loss_coefficient=0,
        total_ua_w_k=0,
    )
    model = EulerLiquidChannel(base)
    averages = exact_averages(model, 0, velocity=velocity, amplitude_k=amplitude_k, shape=shape)
    states = [model.cell(q) for q in averages]
    return replace(
        base,
        initial_pressure_profile_pa=tuple(s[0].p_pa for s in states),
        initial_temperature_profile_k=tuple(s[0].T_k for s in states),
        initial_velocity_profile_m_s=tuple(s[1] for s in states),
    )


def errors(model, time_s, *, velocity=1.0, amplitude_k=0.01, shape="smooth"):
    n = model.config.cells
    exact = exact_averages(model, time_s, velocity=velocity, amplitude_k=amplitude_k, shape=shape)
    actual = model._state[: 3 * n].reshape(n, 3)
    temperatures = np.array([model.cell(q)[0].T_k for q in exact])
    measured = model.rates(model._state)["temperatures"]
    delta = measured - temperatures
    return dict(
        temperature_l1_k=float(np.mean(abs(delta))),
        temperature_l2_k=float(np.sqrt(np.mean(delta**2))),
        temperature_linf_k=float(max(abs(delta))),
        temperature_signed_integral_k_m=float(sum(delta) * model.dx),
        conserved_l1=[float(x) for x in np.mean(abs(actual - exact), axis=0) / model.dx],
        min_temperature_k=float(min(measured)),
        max_temperature_k=float(max(measured)),
        pressure_error_pa=float(max(abs(model.rates(model._state)["pressures"] - 7e6))),
    )
