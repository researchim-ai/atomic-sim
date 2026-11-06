"""Source-bound qualification; no experimental or plant promotion."""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from atomic_model_workspace.common import digest

from .cases import slab
from .model import Model
from .reference import continuum_k, dense_k, exact_evolution, two_layer_source
from .solvers import eigenvalue, fixed_source
from .transient import checkpoint, equilibrium_precursors, restore, start, step

CRITERIA = dict(
    eigen_dense_relative=1e-8,
    fixed_dense_relative=1e-10,
    spatial_order_min=1.8,
    temporal_order_min=0.9,
    temporal_finest_relative=0.002,
    linear_residual=1e-10,
    inventory_relative_defect=1e-9,
    critical_population_relative_drift=2e-8,
)


def identity():
    return dict(
        sources={
            str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(
                [*Path(__file__).parent.glob("*.py"), *Path(__file__).parent.glob("data/*.json")]
            )
        },
        runtime=dict(
            python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__
        ),
    )


def qualify():
    from .reference import dense_operators

    static = []
    for hetero in [False, True]:
        model = Model(slab(12, heterogeneous=hetero))
        result = eigenvalue(model)
        kref = dense_k(model)
        q = np.ones((model.g, model.n))
        fixed = fixed_source(model, q)
        loss, fiss, _ = dense_operators(model)
        expected = np.linalg.solve(loss - fiss, (q * model.width).ravel())
        error = float(np.max(abs(np.ravel(fixed["flux"]) - expected)) / np.max(abs(expected)))
        relative = abs(result["k"] - kref) / kref
        if relative > CRITERIA["eigen_dense_relative"] or error > CRITERIA["fixed_dense_relative"]:
            raise ValueError("Independent static reference failed")
        static.append(
            dict(
                heterogeneous=hetero,
                k=result["k"],
                dense_k=kref,
                eigen_relative_error=relative,
                fixed_source_relative_error=error,
                eigen_residual=result["relative_residual"],
            )
        )
    spatial = []
    for problem in ["homogeneous_multigroup_k", "discontinuous_diffusion_source"]:
        errors = []
        for n in [8, 16, 32, 64]:
            config = slab(n)
            if problem == "homogeneous_multigroup_k":
                model = Model(config)
                expected = continuum_k(model)
                error = abs(eigenvalue(model)["k"] - expected) / expected
            else:
                config["diffusion_m"] = [[0.01] * (n // 2) + [0.04] * (n // 2)] * 2
                config["absorption_per_m"] = np.zeros((2, n)).tolist()
                config["nu_fission_per_m"] = np.zeros((2, n)).tolist()
                config["scatter_per_m"] = np.zeros((2, 2, n)).tolist()
                model = Model(config)
                expected = two_layer_source((model.faces[1:] + model.faces[:-1]) / 2)
                actual = np.array(fixed_source(model, np.ones((2, n)))["flux"])
                error = float(np.max(abs(actual - expected)) / np.max(expected))
            errors.append(float(error))
        orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:])).tolist()
        if min(orders) < CRITERIA["spatial_order_min"]:
            raise ValueError("Spatial convergence failed")
        spatial.append(
            dict(problem=problem, meshes=[8, 16, 32, 64], relative_errors=errors, orders=orders)
        )
    transients = []
    for hetero in [False, True]:
        model = Model(slab(8, heterogeneous=hetero))
        phi = np.tile(np.sin(np.pi * (model.faces[:-1] + model.faces[1:]) / 2), (2, 1))
        c = equilibrium_precursors(model, phi) * 0.7
        q = np.full_like(phi, 0.03)
        exact = exact_evolution(model, phi, c, q, 0.04)
        errors, defects, residuals, local_defects = [], [], [], []
        replay_equal = True
        for dt in [0.002, 0.001, 0.0005]:
            state = start(model, phi, c)
            count = round(0.04 / dt)
            saved = None
            for index in range(count):
                state = step(model, state, dt, q)
                if index == count // 2 - 1:
                    saved = checkpoint(model, state)
            resumed = restore(model, saved)
            for _ in range(count - count // 2):
                resumed = step(model, resumed, dt, q)
            replay_equal &= checkpoint(model, resumed) == checkpoint(model, state)
            actual = np.concatenate((np.ravel(state.flux), np.ravel(state.precursors)))
            errors.append(float(np.max(abs(actual - exact)) / np.max(abs(exact))))
            defects.append(max(record["relative_inventory_defect"] for record in state.history))
            residuals.append(max(record["relative_residual"] for record in state.history))
            local_defects.append(
                max(record["max_local_inventory_relative_defect"] for record in state.history)
            )
        orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:])).tolist()
        if (
            min(orders) < CRITERIA["temporal_order_min"]
            or errors[-1] > CRITERIA["temporal_finest_relative"]
            or max(defects) > CRITERIA["inventory_relative_defect"]
            or max(local_defects) > CRITERIA["inventory_relative_defect"]
            or max(residuals) > CRITERIA["linear_residual"]
            or not replay_equal
        ):
            raise ValueError("Transient qualification failed")
        transients.append(
            dict(
                heterogeneous=hetero,
                steps_s=[0.002, 0.001, 0.0005],
                horizon_s=0.04,
                relative_errors=errors,
                orders=orders,
                inventory_relative_defects=defects,
                local_inventory_relative_defects=local_defects,
                residuals=residuals,
                complete_history_restart_equal=replay_equal,
            )
        )
    config = slab(6, True, ("reflecting", "reflecting"))
    config["speed_m_s"] = [[1e7] * 6, [2200.0] * 6]
    k = eigenvalue(Model(config))["k"]
    config["nu_fission_per_m"] = (np.array(config["nu_fission_per_m"]) / k).tolist()
    model = Model(config)
    phi = np.array(eigenvalue(model)["flux"])
    c = equilibrium_precursors(model, phi)
    state = start(model, phi, c)
    for _ in range(20):
        state = step(model, state, 0.001, np.zeros_like(phi))
    drift = max(
        float(np.max(abs(np.array(state.flux) / phi - 1))),
        float(np.max(abs(np.array(state.precursors) / c - 1))),
    )
    if drift > CRITERIA["critical_population_relative_drift"]:
        raise ValueError("Physical-speed critical equilibrium drift failed")
    physical_speed_equilibrium = dict(
        speed_m_s=[1e7, 2200],
        steps=20,
        dt_s=0.001,
        max_population_relative_drift=drift,
        meaning="Stationarity identity only; no physical XS validation",
    )
    return dict(
        schema_version=1,
        source=identity(),
        criteria=CRITERIA,
        status="numerically_verified",
        data_fidelity="synthetic",
        static=static,
        physical_speed_equilibrium=physical_speed_equilibrium,
        spatial=spatial,
        transients=transients,
        validated=False,
        plant_referenced=False,
        exclusions=[
            "Evaluated nuclear-data provenance",
            "OpenMC/inter-code qualification",
            "Thermal feedback and power mapping",
            "Burnup/transport/plant validation",
            "Moving rods and time-dependent materials",
        ],
    )


def audit(path):
    saved = json.loads(Path(path).read_text())
    if set(saved) != {"payload", "sha256"} or digest(saved["payload"]) != saved["sha256"]:
        raise ValueError("Invalid qualification envelope")
    if saved["payload"]["source"] != identity():
        raise ValueError("Qualification source/runtime mismatch")
    fresh = qualify()
    if digest(fresh) != saved["sha256"]:
        raise ValueError("Qualification recomputation differs")
    return fresh
