"""Pressure-anchored charging solver for the separate natural-circulation case.

Adapted explicitly from atomic_model_plant_extension.initialization. A prescribed
core pressure determines the initial fill mass; this is initialization, not an
operational change to conserved inventories or a fabricated transient history.
"""

from copy import deepcopy

import numpy as np
from scipy.optimize import OptimizeResult

from core.status import DomainError, SolverError


def charge(plant, relative_power=1.0):
    if not np.isfinite(relative_power) or not 0.01 <= relative_power <= 1.1:
        raise ValueError("Charging power domain0.01..1.1")
    work = deepcopy(plant)
    initial = plant.pack().copy()
    scale = np.maximum(abs(initial), 1.0)
    names = plant.network.volume_names()
    primary = ["core", "hot_leg", "sg_primary", "cold_leg", "pressurizer"]
    coreloop = primary[:-1]
    mass = {name: 13 + 3 * names.index(name) for name in names}
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
        value[mass["core"]] = (work.network.volumes["core"].pressure_pa - 16e6) / 0.01
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

    result = damped_newton(residual, initial[retained] / scale[retained])
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
            "initial fill defined by core pressure16MPa; resulting primary mass is an output",
            "secondary+hotwell total mass",
            "circulating primary boron",
            "pressurizer and secondary boron",
            "pressurizer entropy anchor",
            "requested fission power",
        ],
        parameter_changes=False,
        initialization_kind="pressure-anchored initial charging, not closed-system transient",
        charged_primary_mass_kg=float(sum(state[mass[n]] for n in primary)),
        validated=False,
        plant_referenced=False,
    )
    plant.unpack(state)
    plant.refresh_algebraic(state)
    return report


def damped_newton(fun, start):
    """Reject inadmissible Newton trials by line search; never project stocks."""
    state = start.copy()
    for iteration in range(40):
        value = fun(state)
        if np.max(abs(value)) <= 0.1:
            return OptimizeResult(x=state, success=True, message="domain-aware damped Newton")
        columns = []
        for i in range(len(state)):
            delta = np.zeros_like(state)
            delta[i] = 1e-6
            columns.append((fun(state + delta) - fun(state - delta)) / 2e-6)
        jac = np.column_stack(columns)
        row_scale = np.maximum(np.linalg.norm(jac, axis=1), 1.0)
        direction = np.linalg.solve(jac / row_scale[:, None], -value / row_scale)
        for halves in range(30):
            alpha = 0.5**halves
            try:
                trial = fun(state + alpha * direction)
            except (DomainError, ValueError):
                continue
            if np.linalg.norm(trial / row_scale) < np.linalg.norm(value / row_scale):
                state += alpha * direction
                break
        else:
            raise SolverError("Charging Newton domain/merit line search exhausted")
    raise SolverError("Charging Newton iteration budget exhausted")
