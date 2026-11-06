"""Stationary coupled initialization with explicit inventory/control constraints."""

from copy import deepcopy

import numpy as np
from scipy.optimize import least_squares, root

from core.status import DomainError, SolverError


def solve(plant, relative_power=1.0):
    if not np.isfinite(relative_power) or not 0.5 <= relative_power <= 1.1:
        raise ValueError("Stationary power initialization domain0.5..1.1")
    work = deepcopy(plant)
    initial = plant.pack().copy()
    scale = np.maximum(abs(initial), 1.0)
    names = plant.network.volume_names()
    primary = ["core", "hot_leg", "sg_primary", "cold_leg", "pressurizer"]
    coreloop = primary[:-1]
    mass = {name: 13 + 3 * names.index(name) for name in names}
    p_mass = sum(initial[mass[n]] for n in primary)
    s_mass = initial[mass["sg_secondary"]] + initial[-3]
    b_mass = sum(initial[mass[n] + 2] for n in coreloop)
    press = work.network.volumes["pressurizer"]
    s_press = press.mass_kg * press.thermo.s_j_kg_k
    fixed = {
        13 + 3 * len(names) + plant.network.junction_names().index(n): 0.0
        for n in ["spray", "dump", "surge"]
    }
    retained = np.array([i for i in range(len(initial)) if i not in fixed])
    removed_equations = {
        13 + 3 * len(names) + plant.network.junction_names().index(n) for n in ["spray", "dump"]
    } | {mass["pressurizer"]}
    equations = np.array([i for i in range(len(initial)) if i not in removed_equations])
    tolerances = np.full(len(initial), 1e-8)
    tolerances[7:10] = 0.1
    for n, i in mass.items():
        tolerances[i : i + 3] = [1e-6, 1.0, 1e-10]
    offset = 13 + 3 * len(names)
    for j, n in enumerate(work.network.junction_names()):
        link = work.network.junctions[n]
        tolerances[offset + j] = (
            0.01 * link.area_m2 / link.length_m if link.flow_law == "momentum" else 1e-6
        )
    tolerances[-3:] = [1e-6, 1.0, 1e-8]
    calls = 0

    def physical(z):
        state = initial.copy()
        state[retained] = z * scale[retained]
        for i, value in fixed.items():
            state[i] = value
        return state

    def residual(z):
        nonlocal calls
        calls += 1
        if calls > 10000:
            raise SolverError("Coupled initialization work budget exhausted")
        state = physical(z)
        rate = work.derivatives(0.0, state)
        value = rate / tolerances
        value[12] = (state[0] - relative_power) / 1e-10
        value[mass["core"]] = (sum(state[mass[n]] for n in primary) - p_mass) / max(
            1e-6, p_mass * 1e-10
        )
        value[mass["sg_secondary"]] = (state[mass["sg_secondary"]] + state[-3] - s_mass) / max(
            1e-6, s_mass * 1e-10
        )
        value[mass["core"] + 2] = (sum(state[mass[n] + 2] for n in coreloop) - b_mass) / max(
            1e-10, b_mass * 1e-10
        )
        for n in ["pressurizer", "sg_secondary"]:
            i = mass[n] + 2
            value[i] = (state[i] - initial[i]) / max(1e-10, abs(initial[i]) * 1e-10)
        press = work.network.volumes["pressurizer"]
        value[mass["pressurizer"] + 1] = (press.mass_kg * press.thermo.s_j_kg_k - s_press) / max(
            abs(s_press) * 1e-10, 1e-6
        )
        return value[equations]

    result = root(
        residual,
        initial[retained] / scale[retained],
        method="hybr",
        options={"xtol": 1e-10, "maxfev": 2000},
    )
    if np.max(abs(residual(result.x))) > 1.0:
        refined = least_squares(
            residual,
            result.x,
            jac="3-point",
            x_scale="jac",
            diff_step=1e-6,
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            max_nfev=300,
        )
        if np.max(abs(residual(refined.x))) < np.max(abs(residual(result.x))):
            result = refined
    state = physical(result.x)
    defect = float(np.max(abs(residual(result.x))))
    if defect > 1.0:
        raise SolverError(
            f"Coupled nominal dimensional residual failed:{result.message}, max_tolerance_ratio={defect}"
        )
    # Diagnose rank only after solving: scale rows/columns before SVD so
    # conservation anchors and momentum equations have comparable magnitudes.
    step = 1e-6
    columns = []
    for i in range(len(result.x)):
        delta = np.zeros_like(result.x)
        delta[i] = step
        columns.append((residual(result.x + delta) - residual(result.x - delta)) / (2 * step))
    matrix = np.column_stack(columns)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1), 1e-30)[:, None]
    matrix /= np.maximum(np.linalg.norm(matrix, axis=0), 1e-30)[None, :]
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-9))
    if rank != len(result.x):
        raise SolverError(f"Initialization constrained Jacobian rank {rank}/{len(result.x)}")
    work.refresh_algebraic(state)
    rate = work.derivatives(0.0, state)
    # Dimensional balances are final acceptance criteria, independent of optimizer scaling.
    masses = max(abs(rate[mass[n]]) for n in names)
    energies = max(abs(rate[mass[n] + 1]) for n in names)
    momentum = {}
    offset = 13 + 3 * len(names)
    for j, name in enumerate(work.network.junction_names()):
        link = work.network.junctions[name]
        if link.flow_law == "momentum":
            momentum[name] = abs(float(rate[offset + j])) * link.length_m / link.area_m2
    if (
        masses > 1e-6
        or energies > 1.0
        or max(momentum.values()) > 0.01
        or abs(state[0] - relative_power) > 1e-10
    ):
        raise SolverError(
            f"Coupled dimensional initial residual failed:{masses},{energies},{momentum}"
        )
    if not 0 <= state[12] <= 1:
        raise DomainError("Stationary rod setting outside stroke")
    work._reject_if_out_of_domain()
    report = dict(
        time_s=0.0,
        constrained_jacobian_rank=rank,
        constrained_unknowns=len(result.x),
        equilibrated_singular_values=singular.tolist(),
        rank_relative_threshold=1e-9,
        warmup_steps=0,
        requested_relative_power=relative_power,
        evaluations=calls,
        scaled_defect=defect,
        max_mass_rate_kg_s=float(masses),
        max_inventory_energy_rate_w=float(energies),
        momentum_residuals_pa=momentum,
        initial_rod_insertion=float(state[12]),
        max_state_relative_change=float(np.max(abs(state - initial) / scale)),
        constraints=[
            "primary total mass",
            "secondary+hotwell total mass",
            "circulating primary boron",
            "pressurizer and secondary boron",
            "pressurizer entropy anchor",
            "requested fission power",
        ],
        parameter_changes=False,
        validated=False,
        plant_referenced=False,
    )
    plant.unpack(state)
    plant.refresh_algebraic(state)
    return report
