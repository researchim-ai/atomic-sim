"""Transactional backward Euler, local precursors, full replay-checked checkpoint."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from atomic_model_workspace.common import digest

from .model import array
from .solvers import factor, relative_residual


@dataclass(frozen=True)
class State:
    flux: tuple
    precursors: tuple
    time_s: float
    history: tuple
    initial_flux: tuple
    initial_precursors: tuple
    model_digest: str


def tuples(a):
    return tuple(tuple(float(x) for x in row) for row in a)


def start(model, flux, precursors):
    phi = array(flux, (model.g, model.n), "flux")
    c = array(precursors, (model.j, model.n), "precursors")
    return State(tuples(phi), tuples(c), 0.0, (), tuples(phi), tuples(c), digest(model.config()))


def equilibrium_precursors(model, flux):
    phi = array(flux, (model.g, model.n), "flux")
    return model.beta * (model.nu_fission_per_m * phi).sum(axis=0) / model.decay[:, None]


def step(model, state, dt, source):
    if digest(model.config()) != state.model_digest:
        raise ValueError("State belongs to another material/mesh configuration")
    if not np.isfinite(dt) or dt <= 0 or len(state.history) >= 10000:
        raise ValueError("Invalid timestep or history budget exceeded")
    if not np.isfinite(state.time_s + dt) or state.time_s + dt <= state.time_s:
        raise ValueError("Time resolution exhausted")
    q = array(source, (model.g, model.n), "source")
    old = np.concatenate(
        (
            array(state.flux, (model.g, model.n), "flux").ravel(),
            array(state.precursors, (model.j, model.n), "precursors").ravel(),
        )
    )
    # Sufficient M-matrix condition using positive diagonal similarity from old+1.
    # Reject, never silently shorten a user timestep or clip negative populations.
    weight = np.maximum(old, 1.0)
    population_weight = np.maximum(old, max(float(np.max(old)), 1.0) * 1e-12)
    growth_bound = min(
        float(np.max((model.kinetics @ candidate) / candidate))
        for candidate in (weight, population_weight)
    )
    if dt * max(growth_bound, 0) >= 0.95:
        raise ValueError("Timestep fails sufficient positivity certificate; reduce dt")
    operator = sparse.eye(len(old), format="csc") - dt * model.kinetics
    forcing = np.concatenate(((q * model.speed_m_s).ravel(), np.zeros(model.j * model.n)))
    rhs = old + dt * forcing
    new = factor(operator).solve(rhs)
    residual = relative_residual(operator, new, rhs)
    if not np.isfinite(new).all() or np.any(new < 0) or residual > 1e-10:
        raise ValueError("Rejected population or linear residual")
    split = model.g * model.n
    phi, c = new[:split].reshape(model.g, model.n), new[split:].reshape(model.j, model.n)
    inventory_weight = np.concatenate(
        ((model.width / model.speed_m_s).ravel(), np.tile(model.width, model.j))
    )
    before, after = float(inventory_weight @ old), float(inventory_weight @ new)
    production = float(np.sum(model.nu_fission_per_m * phi * model.width))
    absorption = float(np.sum(model.absorption_per_m * phi * model.width))
    leakage = float(np.sum(model.leakage * phi))
    external = float(np.sum(q * model.width))
    change = dt * (production - absorption - leakage + external)
    defect = after - before - change
    balance_scale = max(abs(before), abs(after), abs(change), 1e-300)
    if abs(defect) / balance_scale > 1e-9:
        raise ValueError("Rejected global neutron/precursor balance")
    from .diagnostics import face_currents

    currents = face_currents(model, phi)
    old_phi = old[:split].reshape(model.g, model.n)
    old_c = old[split:].reshape(model.j, model.n)
    cell_before = ((old_phi / model.speed_m_s).sum(axis=0) + old_c.sum(axis=0)) * model.width
    cell_after = ((phi / model.speed_m_s).sum(axis=0) + c.sum(axis=0)) * model.width
    cell_rate = (
        (model.nu_fission_per_m * phi - model.absorption_per_m * phi + q) * model.width
    ).sum(axis=0) - (currents[:, 1:] - currents[:, :-1]).sum(axis=0)
    local_defect = cell_after - cell_before - dt * cell_rate
    local_scale = np.maximum(
        np.maximum(abs(cell_before), abs(cell_after)), np.maximum(abs(dt * cell_rate), 1e-300)
    )
    local_relative = float(np.max(abs(local_defect) / local_scale))
    if local_relative > 1e-9:
        raise ValueError("Rejected local neutron/precursor balance")
    record = dict(
        dt_s=float(dt),
        source=q.tolist(),
        time_s=state.time_s + dt,
        relative_residual=residual,
        max_local_inventory_relative_defect=local_relative,
        inventory_before=before,
        inventory_after=after,
        fission_rate=production,
        absorption_rate=absorption,
        outward_leakage_rate=leakage,
        external_rate=external,
        inventory_defect=defect,
        relative_inventory_defect=abs(defect) / balance_scale,
    )
    return State(
        tuples(phi),
        tuples(c),
        state.time_s + dt,
        state.history + (record,),
        state.initial_flux,
        state.initial_precursors,
        state.model_digest,
    )


def checkpoint(model, state):
    from .workflow import identity

    if state.model_digest != digest(model.config()):
        raise ValueError("Checkpoint model mismatch")
    payload = dict(
        schema_version=1,
        source=identity(),
        config=model.config(),
        initial_flux=state.initial_flux,
        initial_precursors=state.initial_precursors,
        history=list(state.history),
        flux=state.flux,
        precursors=state.precursors,
        time_s=state.time_s,
    )
    return dict(payload=payload, sha256=digest(payload))


def restore(model, saved):
    from .workflow import identity

    if not isinstance(saved, dict) or set(saved) != {"payload", "sha256"}:
        raise ValueError("Invalid checkpoint envelope")
    p = saved["payload"]
    fields = {
        "schema_version",
        "source",
        "config",
        "initial_flux",
        "initial_precursors",
        "history",
        "flux",
        "precursors",
        "time_s",
    }
    if (
        set(p) != fields
        or p["schema_version"] != 1
        or saved["sha256"] != digest(p)
        or p["source"] != identity()
        or digest(p["config"]) != digest(model.config())
        or not isinstance(p["history"], list)
        or len(p["history"]) > 10000
    ):
        raise ValueError("Checkpoint identity/schema/history mismatch")
    state = start(model, p["initial_flux"], p["initial_precursors"])
    for record in p["history"]:
        state = step(model, state, record["dt_s"], record["source"])
    if digest(checkpoint(model, state)["payload"]) != saved["sha256"]:
        raise ValueError("Checkpoint fails complete history replay")
    return state
