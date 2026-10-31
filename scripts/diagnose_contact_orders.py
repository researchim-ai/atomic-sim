"""FFT-exact semidiscrete linear upwind diagnostic, not Euler qualification.

Inventories are normalized to one fixed reference volume; linear transport
commutes with that scaling. Actual Euler/EOS coupling still needs raw runs.
"""

import numpy as np

from physics.hydraulics.euler_channel import EulerConfig, EulerLiquidChannel


def main():
    m = EulerLiquidChannel(
        EulerConfig(
            cells=1,
            numerical_flux="hllc",
            boundary="periodic",
            total_ua_w_k=0,
            total_loss_coefficient=0,
        )
    )
    qh = m.inventory(7e6, 500.01, 1)
    ql = m.inventory(7e6, 499.99, 1)
    errs = []
    for n in (16, 32, 64, 128, 256, 512, 1024):
        dx = 3 / n
        q0 = np.tile(ql, (n, 1))
        q0[: n // 2] = qh
        modes = np.fft.fft(q0, axis=0)
        theta = 2 * np.pi * np.fft.fftfreq(n)
        err = 0
        for t in np.linspace(0.03, 0.3, 10):
            numeric = np.fft.ifft(
                modes * np.exp((np.exp(-1j * theta) - 1) * t / dx)[:, None], axis=0
            ).real
            exact = []
            for i in range(n):
                a, b = i * dx, (i + 1) * dx
                hot = sum(max(0, min(b, t + 1.5 + k * 3) - max(a, t + k * 3)) for k in (-1, 0, 1))
                exact.append(ql + (qh - ql) * hot / dx)
            tn = np.array([m.cell(q)[0].T_k for q in numeric])
            te = np.array([m.cell(q)[0].T_k for q in exact])
            err = max(err, float(np.mean(abs(tn - te))))
        errs.append(err)
        print(n, err, None if len(errs) < 2 else np.log2(errs[-2] / err), flush=True)


if __name__ == "__main__":
    main()
