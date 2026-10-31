"""Independent exact periodic advection benchmark; no reactor/EOS qualification."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from physics.hydraulics.reconstruction import conservative_rate
from sdk.experiment import _source_fingerprint


def evolve(values, speed, duration, *, scheme, cfl=0.3):
    u = np.asarray(values, dtype=float).copy()
    n = len(u)
    if not np.isfinite([speed, duration, cfl]).all() or duration < 0 or not 0 < cfl <= 0.5:
        raise ValueError("Invalid advection time/CFL")
    if speed == 0 or duration == 0:
        return u
    steps = int(np.ceil(duration * abs(speed) * n / cfl))
    dt = duration / steps
    flows = np.full(n + 1, speed)

    def rhs(y):
        return n * conservative_rate(y, flows, scheme=scheme, periodic=True)

    # SSPRK3, a different time method from the coupled BDF/Radau model.
    for _ in range(steps):
        a = u + dt * rhs(u)
        b = 0.75 * u + 0.25 * (a + dt * rhs(a))
        u = u / 3 + 2 * (b + dt * rhs(b)) / 3
    return u


def verify():
    cases = []
    for speed in (-1.0, 1.0):
        for scheme in ("donor", "muscl_mc"):
            rows = []
            for n in (32, 64, 128):
                x = (np.arange(n) + 0.5) / n
                # Exact cell averages of a smooth periodic wave, not point samples.
                amplitude = np.sin(np.pi / n) / (np.pi / n)
                u = 1.0 + 0.2 * amplitude * np.sin(2 * np.pi * x)
                result = evolve(u, speed, 0.25, scheme=scheme)
                exact = 1.0 + 0.2 * amplitude * np.sin(2 * np.pi * (x - speed * 0.25))
                rows.append(
                    dict(
                        cells=n,
                        l1_error=float(np.mean(abs(result - exact))),
                        mass_error=float(abs(np.mean(result) - np.mean(u))),
                    )
                )
            orders = [
                float(np.log2(rows[i]["l1_error"] / rows[i + 1]["l1_error"])) for i in range(2)
            ]
            threshold = 0.85 if scheme == "donor" else 1.7
            cases.append(
                dict(
                    speed=speed,
                    scheme=scheme,
                    rows=rows,
                    observed_orders=orders,
                    passed=all(p > threshold for p in orders)
                    and max(r["mass_error"] for r in rows) < 1e-12,
                )
            )
    bounded = []
    for speed in (-1.0, 1.0):
        n = 128
        x = (np.arange(n) + 0.5) / n
        u = np.where((x > 0.25) & (x < 0.5), 1.0, 0.0)
        result = evolve(u, speed, 0.2, scheme="muscl_mc")

        def tv(v):
            return float(sum(abs(np.roll(v, -1) - v)))

        bounded.append(
            dict(
                speed=speed,
                min=float(min(result)),
                max=float(max(result)),
                total_variation_before=tv(u),
                total_variation_after=tv(result),
                mass_error=float(abs(np.mean(result) - np.mean(u))),
            )
        )
    bounds_ok = all(
        r["min"] >= -1e-12
        and r["max"] <= 1 + 1e-12
        and r["total_variation_after"] <= r["total_variation_before"] + 1e-10
        and r["mass_error"] < 1e-12
        for r in bounded
    )
    return dict(
        schema_version=1,
        kind="transport_operator_verification",
        validated=False,
        plant_referenced=False,
        source_sha256=_source_fingerprint(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        equation="u_t + a u_x = 0, periodic unit interval; exact advected cell averages",
        smooth_cases=cases,
        discontinuous_cases=bounded,
        passed=all(c["passed"] for c in cases) and bounds_ok,
        scope="Constant-coefficient scalar transport, SSPRK3 CFL<=0.5; not TVD/second-order qualification of implicit IF97 channel",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = verify()
    with args.output.open("x") as f:
        json.dump(r, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            dict(
                passed=r["passed"],
                orders=[
                    dict(scheme=c["scheme"], speed=c["speed"], orders=c["observed_orders"])
                    for c in r["smooth_cases"]
                ],
            )
        )
    )
    if not r["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
