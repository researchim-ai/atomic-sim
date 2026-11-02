"""Transactional adaptive step for a pure state-margin model; event-prefix diagnostics."""

from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from core.network import NetworkError, number
from core.status import DomainError, SolverError


def advance_with_limits(model, dt_s, *, method="Radau", rtol=1e-8, max_step_s=0.1, atol_scale=0.1):
    dt = number(dt_s, "dt_s", positive=True)
    rtol = number(rtol, "rtol", positive=True)
    step = number(max_step_s, "max_step_s", positive=True)
    scale = number(atol_scale, "atol_scale", positive=True)
    if method not in ("Radau", "BDF") or not 1e-12 <= rtol <= 1e-2 or not 1e-6 <= scale <= 1:
        raise NetworkError("Unsupported bounded-model solver policy")
    end = model.time_s + dt
    if not np.isfinite(end) or end <= model.time_s:
        raise NetworkError("Bounded model clock cannot advance")
    tolerances = np.asarray(model.absolute_tolerances, dtype=float)
    if (
        tolerances.shape != model._state.shape
        or not np.isfinite(tolerances).all()
        or np.any(tolerances <= 0)
    ):
        raise NetworkError("Positive absolute tolerance required for every coupled state")
    model.observation()
    # A localized terminal state can be a few root-resolution units past
    # the mathematical surface. Re-enter only if the new vector field
    # demonstrably points into the domain; otherwise retain the same limit.
    for previous in reversed(model._events):
        if previous["t"] < model.time_s:
            break
        if previous["kind"] != "domain_limit":
            continue
        margins = {name: (value, tol) for name, value, _, tol in model._margins(model._state)}
        value, tolerance = margins[previous["criterion"]]
        if value > tolerance:
            continue
        probe_dt = min(dt, 1e-3)
        probe = model._state + probe_dt * model.derivatives(model.time_s, model._state)
        try:
            next_value = {name: value for name, value, _, _ in model._margins(probe)}[
                previous["criterion"]
            ]
        except DomainError:
            next_value = value
        if next_value <= value + tolerance:
            return dict(
                observation=model.observation(),
                boundary=deepcopy(previous),
                solver=dict(
                    schema_version=1,
                    backend="boundary_hold",
                    method=method,
                    start_time_s=model.time_s,
                    end_time_s=model.time_s,
                    max_step_s=step,
                    rtol=rtol,
                    atol_scale=scale,
                    scipy_nfev=0,
                    scipy_njev=0,
                    scipy_nlu=0,
                    mesh_times_s=[model.time_s],
                ),
            )
        break
    previous_state = None
    previous_margins = None

    def margins(y):
        nonlocal previous_state, previous_margins
        if previous_state is None or not np.array_equal(previous_state, y):
            previous_margins = model._margins(y)
            previous_state = np.array(y, copy=True)
        return previous_margins

    events = []
    for index in range(len(margins(model._state))):

        def event(t, y, index=index):
            margin = margins(y)[index]
            # Half of the declared root resolution avoids false terminal
            # events at a stationary boundary represented with EOS roundoff.
            return margin[1] + 0.5 * margin[3]

        event.terminal = True
        event.direction = -1
        events.append(event)
    result = solve_ivp(
        model.derivatives,
        (model.time_s, end),
        model._state,
        method=method,
        rtol=rtol,
        atol=tolerances * scale,
        max_step=step,
        jac=model.jacobian,
        events=events,
        dense_output=False,
    )
    if not result.success:
        raise SolverError(result.message)
    candidate = result.y[:, -1].copy()
    stamp = float(result.t[-1])
    observed = model.observation(candidate, stamp)
    boundary = None
    for index, roots in enumerate(result.t_events):
        if len(roots):
            name, value, unit, tolerance = model._margins(candidate)[index]
            if abs(value) > tolerance:
                raise NetworkError("Bounded model event root residual exceeded")
            boundary = dict(
                kind="domain_limit",
                t=stamp,
                criterion=name,
                residual=float(value),
                unit=unit,
                root_tolerance=tolerance,
            )
            break
    if boundary is not None and len(model._events) >= 10000:
        raise NetworkError("Bounded model event history resource limit exceeded")
    diagnostic = dict(
        schema_version=1,
        backend="scipy.solve_ivp",
        method=method,
        start_time_s=model.time_s,
        end_time_s=stamp,
        max_step_s=step,
        rtol=rtol,
        atol_scale=scale,
        scipy_nfev=int(result.nfev),
        scipy_njev=int(result.njev),
        scipy_nlu=int(result.nlu),
        mesh_times_s=result.t.tolist(),
    )
    model._state, model.time_s = candidate, stamp
    if boundary is not None:
        model._events.append(boundary)
    return dict(observation=observed, boundary=deepcopy(boundary), solver=diagnostic)
