"""Thermal transient self-convergence on one pipe, with a temporal control.

No exact transient solution or blanket hydrodynamic/acoustic qualification is
claimed. All input bundles must complete and pass hash/ledger/restart checks.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from physics.hydraulics.dynamic_channel import DynamicConfig, DynamicWallChannel
from scripts.compare_isobaric_runs import load
from sdk.experiment import _source_fingerprint

ACCEPTANCE_CRITERIA = dict(
    norm="sampled_Linf_response",
    minimum_order=0.7,
    maximum_temporal_spatial_ratio=0.1,
    maximum_pressure_temporal_pa=1.0,
    maximum_flow_temporal_kg_s=1e-5,
)


def _pipe_signature(spec):
    c = DynamicConfig(**spec["hydraulics"])
    n = c.cells
    if c.junction_lengths_m is None or c.junction_loss_coefficients is None:
        raise ValueError("Explicit fixed pipe geometry is required")
    length = sum(c.junction_lengths_m)
    loss = sum(c.junction_loss_coefficients)
    weights = np.r_[0.5, np.ones(n - 1), 0.5]
    if not (
        np.allclose(c.junction_lengths_m, length / n * weights, rtol=1e-12, atol=1e-14)
        and np.allclose(c.junction_loss_coefficients, loss / n * weights, rtol=1e-12, atol=1e-14)
        and math.isclose(c.total_volume_m3, c.area_m2 * length, rel_tol=1e-12)
    ):
        raise ValueError("Mesh changed the physical pipe or boundary partition")
    if (
        c.heat_w != 0
        or c.initial_temperature_profile_k is None
        or c.initial_wall_temperature_profile_k is None
    ):
        raise ValueError("Expected an unheated pipe with explicit thermal profiles")
    offsets = np.array(c.initial_wall_temperature_profile_k) - c.initial_temperature_profile_k
    if np.allclose(offsets, 5.0, rtol=0, atol=1e-10):
        shape = "uniform"
    elif np.allclose(
        offsets, 10 * np.sin(np.pi * (np.arange(n) + 0.5) / n) ** 2, rtol=0, atol=1e-10
    ):
        shape = "sin2"
    else:
        raise ValueError("Expected uniform or smooth sine-squared 40 kJ wall pulse")
    model = DynamicWallChannel(c)
    d = model.derivatives(0.0, model._state)
    if max(abs(d[2 * n : 3 * n + 1])) > 1e-5 or max(abs(d[: 2 * n : 2])) > 1e-10:
        raise ValueError("Initial hydraulic profile is not equilibrated")
    return dict(
        total_length_m=length,
        total_loss_coefficient=loss,
        total_volume_m3=c.total_volume_m3,
        area_m2=c.area_m2,
        left_pressure_pa=c.left_pressure_pa,
        right_pressure_pa=c.right_pressure_pa,
        left_temperature_k=c.left_temperature_k,
        right_temperature_k=c.right_temperature_k,
        total_heat_capacity_j_k=c.total_heat_capacity_j_k,
        total_ua_w_k=c.total_ua_w_k,
        heat_w=c.heat_w,
        wall_mean_pulse_k=5.0,
        wall_pulse_shape=shape,
        enthalpy_transport=c.enthalpy_transport,
    )


def _norm(a, b, field, response=True):
    initial_a = a[0.0][field] if response else 0.0
    initial_b = b[0.0][field] if response else 0.0
    return max(abs((row[field] - initial_a) - (b[t][field] - initial_b)) for t, row in a.items())


def assess(rows, grids=(4, 8, 16)):
    """Pure arithmetic also covered by independent synthetic regression cases."""
    if len(grids) != 3 or grids[1] != 2 * grids[0] or grids[2] != 2 * grids[1]:
        raise ValueError("Expected three doubling grids")
    if len(rows) != 4 or not rows[0] or 0.0 not in rows[0]:
        raise ValueError("Four traces starting at zero are required")
    ts = np.array(sorted(rows[0]))
    if (
        len(ts) < 2
        or not np.all(np.isfinite(ts))
        or ts[0] != 0.0
        or ts[-1] <= 0.0
        or any(set(r) != set(ts) for r in rows)
    ):
        raise ValueError("Identical nontrivial time grids required")
    fields = ("right_cell_temperature_k", "mean_wall_temperature_k")
    result = {}
    checks = {}
    for field in fields:
        coarse = _norm(rows[0], rows[1], field)
        fine = _norm(rows[1], rows[2], field)
        temporal = _norm(rows[2], rows[3], field)
        order = math.log2(coarse / fine) if coarse > 0 and fine > 0 else None
        result[field] = dict(
            observed_order=order,
            temporal_to_spatial_ratio=temporal / fine if fine > 0 else None,
        )
        result[field][f"difference_{grids[0]}_{grids[1]}_k"] = coarse
        result[field][f"difference_{grids[1]}_{grids[2]}_k"] = fine
        result[field][f"temporal_difference_{grids[2]}_k"] = temporal
        differences = []
        for a, b in ((rows[0], rows[1]), (rows[1], rows[2])):
            delta = np.array(
                [(a[t][field] - a[0.0][field]) - (b[t][field] - b[0.0][field]) for t in ts]
            )
            differences.append(
                dict(
                    l1_k_s=float(trapezoid(abs(delta), ts)),
                    l2_k=float(np.sqrt(trapezoid(delta**2, ts) / (ts[-1] - ts[0]))),
                    signed_integral_k_s=float(trapezoid(delta, ts)),
                    max_difference_time_s=float(ts[int(np.argmax(abs(delta)))]),
                )
            )
        result[field]["additional_diagnostics_not_used_for_acceptance"] = differences
        checks[field] = bool(
            order is not None
            and order > ACCEPTANCE_CRITERIA["minimum_order"]
            and temporal < ACCEPTANCE_CRITERIA["maximum_temporal_spatial_ratio"] * fine
        )
    diagnostic = dict(
        midpoint_pressure_temporal_pa=_norm(rows[2], rows[3], "midpoint_pressure_pa", False),
        flow_temporal_kg_s=max(
            _norm(rows[2], rows[3], field, False)
            for field in ("inlet_mass_flow_kg_s", "outlet_mass_flow_kg_s")
        ),
    )
    checks["temporal_pressure"] = (
        diagnostic["midpoint_pressure_temporal_pa"]
        <= ACCEPTANCE_CRITERIA["maximum_pressure_temporal_pa"]
    )
    checks["temporal_flow"] = (
        diagnostic["flow_temporal_kg_s"] <= ACCEPTANCE_CRITERIA["maximum_flow_temporal_kg_s"]
    )
    return result, diagnostic, checks


def compare(folders):
    if len(folders) != 4:
        raise ValueError("Expected n4, n8, n16 and n16-tight bundles")
    runs = [load(folder, 8.0) for folder in folders]
    cells = [r[0]["hydraulics"]["cells"] for r in runs]
    if cells != [cells[0], 2 * cells[0], 4 * cells[0], 4 * cells[0]]:
        raise ValueError("Expected doubling spatial grids and repeated finest grid")
    if len({r[4]["source_sha256"] for r in runs}) != 1:
        raise ValueError("Spatial/time comparison requires identical runtime sources")
    signatures = [_pipe_signature(r[0]) for r in runs]
    if any(s != signatures[0] for s in signatures[1:]):
        raise ValueError("Physical pipe changed between runs")
    times = set(runs[0][2])
    if any(set(r[2]) != times for r in runs[1:]):
        raise ValueError("Identical sampled times required")
    if len(times) < 20:
        raise ValueError("Transient sampling is too sparse")
    base = runs[2][0]["solver"]
    tight = runs[3][0]["solver"]
    if not (tight["rtol"] < base["rtol"] and tight["max_step_s"] < base["max_step_s"]):
        raise ValueError("Finest grid requires an independent tighter temporal run")
    if any(r[0]["solver"] != runs[0][0]["solver"] for r in runs[:3]):
        raise ValueError("Spatial grids must use the same temporal settings")
    metrics, diagnostics, checks = assess([r[2] for r in runs], tuple(cells[:3]))
    return dict(
        schema_version=1,
        kind="thermal_transient_self_convergence",
        validated=False,
        plant_referenced=False,
        source_sha256=_source_fingerprint(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_hashes=[r[4]["source_sha256"] for r in runs],
        horizon_s=8.0,
        comparison_times=len(times),
        physical_pipe=signatures[0],
        grids=cells,
        acceptance_criteria=dict(ACCEPTANCE_CRITERIA),
        criteria_sha256=hashlib.sha256(
            json.dumps(ACCEPTANCE_CRITERIA, sort_keys=True).encode()
        ).hexdigest(),
        metrics=metrics,
        diagnostics=diagnostics,
        checks=checks,
        passed=all(checks.values()),
        runs={str(p): r[1] for p, r in zip(folders, runs)},
        scope="Thermal response self-convergence, not an exact transient error bound, pressure spatial qualification or mechanical-energy validation",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = compare(args.runs)
    with args.output.open("x") as f:
        json.dump(r, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(passed=r["passed"], metrics=r["metrics"], diagnostics=r["diagnostics"])))
    if not r["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
