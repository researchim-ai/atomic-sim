"""One-shot transversal temperature events with atomic Radau mode changes.

A conservative physical rate box must exclude a root or prove monotonicity.
Bisection brackets roots of the accepted numerical dense trajectory. Unresolved
transversality or overlapping root intervals is refused, never ordered by input.
"""

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from core.network import NetworkError, identifier, number
from scripts.controlled_reduced_engine import ScheduledReducedGraph, scheduled_request
from scripts.controlled_reduced_engine import signature as controlled_signature
from scripts.nonlinear_thermal_graph import digest
from scripts.solve_hydraulic_graph import _keys, sha
from scripts.solve_resistance_feedback import effective_input

BUDGET = Path("validation/ai05_state_event_budget_v1.json")


def signature():
    value = controlled_signature()
    value["drivers"]["scripts/state_event_radau.py"] = sha(__file__)
    value["budgets"][str(BUDGET)] = sha(BUDGET)
    return value


class StateEventError(NetworkError):
    def __init__(self, criterion, message, **details):
        super().__init__(message)
        self.report = dict(accepted=False, criterion=criterion, message=message, details=details)


def _value(guard, z, counters, budget):
    counters["root"] += 1
    if counters["root"] > budget["maximum_root_evaluations_per_advance"]:
        raise StateEventError("root_budget", "Root evaluation budget exhausted")
    return float(z[guard["index"]] - guard["offset_k"])


def _admit_guards(engine, pending, requested, budget):
    s = engine.system
    k = s.adapter._kernel
    n = s.adapter._n
    full = engine._full
    t = full[:n]
    hyd = dict(
        pressures_pa=dict(zip(k.names, full[n : 2 * n])),
        flows_kg_s={e["id"]: full[2 * n + j] for j, e in enumerate(k.graph["edges"])},
    )
    effective, _ = effective_input(k.base, k.graph, dict(zip(k.names, t)), k.laws)
    original = k.budget["maximum_box_radius_k"]
    unresolved = [guard["id"] for guard in pending]
    try:
        for refinement in range(budget["maximum_guard_box_refinements"]):
            k.budget["maximum_box_radius_k"] = original * 0.5**refinement
            lo, hi, rl, rh = k._box(t, hyd, effective)
            step = requested
            while not (
                np.all(t + step * np.minimum(rl, 0) > lo)
                and np.all(t + step * np.maximum(rh, 0) < hi)
            ):
                step *= 0.5
                if step < engine.budget["minimum_macro_step_s"]:
                    raise StateEventError("interval_resolution", "Cannot admit event interval")
            certified = {}
            unresolved = []
            for guard in pending:
                i = guard["index"]
                threshold = guard["threshold_k"]
                margin = budget["root_value_atol_k"]
                low = t[i] + step * min(rl[i], 0) - margin
                high = t[i] + step * max(rh[i], 0) + margin
                if low <= threshold <= high:
                    sign = (
                        1
                        if rl[i] > budget["minimum_transversal_rate_k_s"]
                        else -1
                        if rh[i] < -budget["minimum_transversal_rate_k_s"]
                        else 0
                    )
                    if sign:
                        certified[guard["id"]] = (float(rl[i]), float(rh[i]))
                    else:
                        unresolved.append(guard["id"])
            if not unresolved:
                return step, (lo, hi), certified
        raise StateEventError(
            "transversality",
            "Cannot exclude or certify a monotone guard crossing",
            guards=unresolved,
        )
    finally:
        k.budget["maximum_box_radius_k"] = original


def _locate(guard, solution, step, start, rate_bounds, counters, budget):
    left = 0.0
    right = step
    direction = guard["direction"]
    gl = _value(guard, solution(0.0), counters, budget)
    gr = _value(guard, solution(step), counters, budget)
    for _ in range(budget["maximum_root_iterations"]):
        if right - left <= budget["root_time_atol_s"] and abs(gr) <= budget["root_value_atol_k"]:
            break
        mid = left + (right - left) * 0.5
        if mid <= left or mid >= right:
            raise StateEventError("root_resolution", "Root bracket cannot progress")
        gm = _value(guard, solution(mid), counters, budget)
        if direction * gm >= 0:
            right, gr = mid, gm
        else:
            left, gl = mid, gm
    else:
        raise StateEventError("root_iterations", "Root iteration budget exhausted")
    if not direction * gl < 0 <= direction * gr:
        raise StateEventError("root_bracket", "Invalid directed root bracket")
    time = start + right
    if time <= start or right < budget["minimum_event_progress_s"]:
        raise StateEventError(
            "zero_time_event", "Event time is not resolvably later than interval start"
        )
    return dict(
        guard_id=guard["id"],
        time_s=time,
        bracket_s=[start + left, time],
        bracket_values_k=[gl, gr],
        root_value_k=gr,
        rate_bounds_k_s=list(rate_bounds),
        threshold_k=guard["threshold_k"],
        direction=direction,
    ), right


def _advance_until(engine, target, pending, counters, budget):
    """Mutates a detached native engine only; entire outer call owns rollback."""
    engine._sources()
    s = engine.system
    d = s.differential
    segments = []
    atol = (
        np.array(
            [
                engine.budget["temperature_atol_k"]
                if unit == "temperature"
                else engine.budget["energy_atol_j"]
                for unit in s.atol_units
            ]
        )
        / s.scales
    )
    while engine._time < target:
        start = engine._time
        requested = min(engine.max_step, target - start)
        step, box, certified = _admit_guards(engine, pending, requested, budget)
        while True:
            counters["attempts"] += 1
            if counters["attempts"] > budget["maximum_attempts_per_advance"]:
                raise StateEventError(
                    "attempt_budget", "Event integration attempt budget exhausted"
                )
            seed = engine._full.copy()
            z = engine._increments.copy()

            def rhs(time, value):
                counters["rhs"] += 1
                if counters["rhs"] > budget["maximum_rhs_evaluations_per_advance"]:
                    raise StateEventError("rhs_budget", "Event integration RHS budget exhausted")
                full = engine._project(value, seed)
                s.require_box(full, box)
                return -s.residual(full, np.zeros_like(full))[d] / s.mass[d] / s.scales

            try:
                solved = solve_ivp(
                    rhs,
                    (0.0, step),
                    z,
                    method="Radau",
                    rtol=engine.budget["rtol"],
                    atol=atol,
                    max_step=step,
                    dense_output=True,
                )
                if not solved.success or solved.t[-1] != step or not np.all(np.isfinite(solved.y)):
                    raise NetworkError("Event Radau trial failed")
                candidate = engine._project(solved.y[:, -1], seed)
                s.require_box(candidate, box)
                break
            except StateEventError:
                raise
            except NetworkError:
                counters["rejected"] += 1
                step *= 0.5
                if step < engine.budget["minimum_macro_step_s"]:
                    raise StateEventError(
                        "segment_failure", "Event Radau segment cannot be resolved"
                    )
        roots = []
        for guard in pending:
            gl = _value(guard, z, counters, budget)
            gr = _value(guard, solved.sol(step), counters, budget)
            direction = guard["direction"]
            if direction * gl < 0 <= direction * gr:
                bounds = certified.get(guard["id"])
                if bounds is None or (
                    bounds[0] <= budget["minimum_transversal_rate_k_s"]
                    if direction == 1
                    else bounds[1] >= -budget["minimum_transversal_rate_k_s"]
                ):
                    raise StateEventError(
                        "uncertified_root",
                        "Numerical root lies outside certified directional crossing",
                    )
                root, local = _locate(guard, solved.sol, step, start, bounds, counters, budget)
                roots.append((root, local))
            elif (
                direction * gl < 0
                and abs(gr) <= budget["root_value_atol_k"]
                and abs(start + step - target) <= 2 * max(math.ulp(target), math.ulp(start))
            ):
                bounds = certified.get(guard["id"])
                if bounds is not None and (
                    bounds[0] > budget["minimum_transversal_rate_k_s"]
                    if direction == 1
                    else bounds[1] < -budget["minimum_transversal_rate_k_s"]
                ):
                    raise StateEventError(
                        "endpoint_contact",
                        "Guard contact at requested endpoint is unresolved",
                        guard=guard["id"],
                        residual_k=gr,
                    )
        roots.sort(key=lambda item: item[0]["bracket_s"][1])
        if len(roots) > 1 and roots[0][0]["bracket_s"][1] + budget["root_time_atol_s"] >= min(
            r[0]["bracket_s"][0] for r in roots[1:]
        ):
            raise StateEventError(
                "ambiguous_roots",
                "Root intervals are not distinguishable",
                roots=[r[0] for r in roots],
            )
        root, committed = roots[0] if roots else (None, step)
        if root is not None and root["time_s"] > target:
            raise StateEventError("event_clock", "Localized event lies beyond requested clock")
        final_z = solved.sol(committed)
        final_full = engine._project(final_z, seed)
        s.require_box(final_full, box)
        if abs(s.energy_error(final_z)) > budget["maximum_energy_error_j"]:
            raise StateEventError("energy", "Event interval energy budget exceeded")
        next_time = root["time_s"] if root else min(target, start + step)
        if next_time <= start:
            raise StateEventError("clock", "Event macro clock cannot progress")
        delta = np.diff(solved.t)
        segments.append(
            dict(
                start_s=start,
                end_s=next_time,
                trial_duration_s=step,
                committed_duration_s=committed,
                maximum_trial_internal_step_s=float(max(delta)),
                accepted_trial_internal_steps=len(delta),
            )
        )
        engine._increments, engine._full, engine._time = final_z, final_full, next_time
        if root:
            for guard in pending:
                if (
                    guard["id"] != root["guard_id"]
                    and abs(_value(guard, final_z, counters, budget)) <= budget["root_value_atol_k"]
                ):
                    raise StateEventError(
                        "zero_time_cascade",
                        "Another guard is unresolved at the event state",
                        guard=guard["id"],
                    )
            engine._sources()
            return root, segments
    engine._sources()
    return None, segments


class StateEventGraph:
    def __init__(self, request, *, max_macro_step_s=2.0):
        _keys(request, ("format", "schema_version", "model_input", "guards"))
        if (
            request["format"] != "atomic-state-event-graph"
            or type(request["schema_version"]) is not int
            or request["schema_version"] != 1
        ):
            raise NetworkError("Expected state-event graph schema1")
        self.budget = json.loads(BUDGET.read_text())
        self._base = deepcopy(request["model_input"])
        self._runtime = ScheduledReducedGraph(
            scheduled_request(self._base, []), max_macro_step_s=max_macro_step_s
        )
        self.max_step = self._runtime.max_step
        self.identity = digest(dict(request=request, max_macro_step_s=self.max_step))
        self.captured = signature()
        self._guards = self._parse(request["guards"])
        self._by_id = {g["id"]: g for g in self._guards}
        self._fired = []

    def _parse(self, guards):
        if not isinstance(guards, list) or len(guards) > self.budget["maximum_guards"]:
            raise NetworkError("State guard budget exceeded")
        runtime = self._runtime
        result = []
        ids = set()
        keys = set()
        for guard in guards:
            _keys(guard, ("id", "node", "threshold_k", "direction", "changes"))
            name = identifier(guard["id"])
            node = guard["node"]
            if name in ids or not isinstance(node, str) or node not in runtime._names:
                raise NetworkError("Duplicate guard or unknown guard node")
            direction = guard["direction"]
            threshold = number(guard["threshold_k"], "threshold_k", positive=True)
            if type(direction) is not int or direction not in (-1, 1):
                raise NetworkError("Guard direction must be signed integer")
            k = runtime._engine.system.adapter._kernel
            if not k._lower < threshold < k._upper:
                raise NetworkError("Guard threshold exceeds fluid domain")
            key = (node, threshold, direction)
            if key in keys:
                raise NetworkError("Duplicate directional threshold; combine commands in one batch")
            i = runtime._names.index(node)
            offset = threshold - runtime._origin[i]
            if abs(offset) <= self.budget["root_value_atol_k"]:
                raise NetworkError("Initial state is unresolved at guard threshold")
            changes = runtime._parser._parse_timeline(
                [dict(time_s=1.0, changes=guard["changes"])], runtime._parser._kernel
            )[0]["changes"]
            result.append(
                dict(
                    id=name,
                    node=node,
                    index=i,
                    threshold_k=threshold,
                    direction=direction,
                    offset_k=float(offset),
                    changes=changes,
                )
            )
            ids.add(name)
            keys.add(key)
        return tuple(result)

    def _sources(self):
        if signature() != self.captured:
            raise StateEventError("source_drift", "State-event sources/budgets changed")

    def _snapshot(self, runtime, fired):
        consumed = {r["guard_id"] for r in fired}
        return dict(
            time_s=runtime._engine._time,
            fired=deepcopy(fired),
            pending_guards=[g["id"] for g in self._guards if g["id"] not in consumed],
            model=runtime.snapshot(),
        )

    def snapshot(self):
        return self._snapshot(self._runtime, self._fired)

    def advance(self, duration_s, *, stop_at_event=False):
        self._sources()
        if type(stop_at_event) is not bool:
            raise NetworkError("stop_at_event must be boolean")
        duration = number(duration_s, "duration_s", positive=True)
        target = self._runtime._engine._time + duration
        if (
            duration > self.budget["maximum_duration_s"]
            or not math.isfinite(target)
            or target <= self._runtime._engine._time
        ):
            raise NetworkError("Invalid state-event duration or clock")
        runtime = deepcopy(self._runtime)
        fired = deepcopy(self._fired)
        segments = []
        switches = []
        counters = dict(rhs=0, root=0, attempts=0, rejected=0)
        while runtime._engine._time < target:
            consumed = {r["guard_id"] for r in fired}
            pending = [g for g in self._guards if g["id"] not in consumed]
            root, part = _advance_until(runtime._engine, target, pending, counters, self.budget)
            segments.extend(part)
            if root is None:
                break
            guard = self._by_id[root["guard_id"]]
            if (
                fired
                and root["time_s"] - fired[-1]["time_s"] < self.budget["minimum_event_progress_s"]
            ):
                raise StateEventError(
                    "event_separation", "Events cannot be ordered at supported time resolution"
                )
            record = dict(**root, changes_sha256=digest(guard["changes"]))
            history = [
                dict(time_s=r["time_s"], changes=self._by_id[r["guard_id"]]["changes"])
                for r in [*fired, record]
            ]
            runtime._set_timeline(history)
            request = scheduled_request(self._base, history)
            runtime._request = request
            runtime.identity = digest(dict(request=request, max_macro_step_s=self.max_step))
            engine, active, receipt, trace = runtime._switch(
                runtime._engine, runtime._active, runtime._cursor
            )
            runtime._engine, runtime._active = engine, active
            runtime._cursor += 1
            runtime._receipts.append(receipt)
            runtime._entry_time = root["time_s"]
            runtime._entry_temperatures = deepcopy(trace["right"]["state"]["temperatures_k"])
            fired.append(record)
            switches.append(dict(root=deepcopy(record), **trace))
            if stop_at_event:
                break
        final = self._snapshot(runtime, fired)
        self._sources()
        self._runtime, self._fired = runtime, fired
        return dict(
            **final,
            segments=segments,
            switches=switches,
            evaluations=counters,
            requested_time_s=target,
            stopped_at_event=stop_at_event and bool(switches),
        )

    def checkpoint(self):
        payload = dict(
            schema_version=1,
            model="state-event-radau-v1",
            input_sha256=self.identity,
            signature=deepcopy(self.captured),
            state=self.snapshot(),
            runtime_checkpoint=self._runtime.checkpoint(),
        )
        return dict(payload=payload, sha256=digest(payload))

    def restore(self, checkpoint):
        self._sources()
        _keys(checkpoint, ("payload", "sha256"))
        p = checkpoint["payload"]
        if digest(p) != checkpoint["sha256"]:
            raise NetworkError("State-event checkpoint digest mismatch")
        _keys(
            p,
            ("schema_version", "model", "input_sha256", "signature", "state", "runtime_checkpoint"),
        )
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["model"] != "state-event-radau-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
        ):
            raise NetworkError("State-event checkpoint input/model/source mismatch")
        state = p["state"]
        _keys(state, ("time_s", "fired", "pending_guards", "model"))
        time = number(state["time_s"], "time_s")
        fired = state["fired"]
        if time < 0 or not isinstance(fired, list) or len(fired) > len(self._guards):
            raise NetworkError("State-event checkpoint history malformed")
        seen = set()
        previous = 0.0
        history = []
        for record in fired:
            _keys(
                record,
                (
                    "guard_id",
                    "time_s",
                    "bracket_s",
                    "bracket_values_k",
                    "root_value_k",
                    "rate_bounds_k_s",
                    "threshold_k",
                    "direction",
                    "changes_sha256",
                ),
            )
            name = record["guard_id"]
            if not isinstance(name, str) or name not in self._by_id or name in seen:
                raise NetworkError("State-event checkpoint guard history mismatch")
            guard = self._by_id[name]
            when = number(record["time_s"], "event_time_s")
            if when - previous < self.budget["minimum_event_progress_s"] or when > time:
                raise NetworkError("State-event checkpoint event clock mismatch")

            def pair(key):
                value = record[key]
                if not isinstance(value, list) or len(value) != 2:
                    raise NetworkError("State-event checkpoint root pair malformed")
                return [number(v, key) for v in value]

            lo, hi = pair("bracket_s")
            gl, gr = pair("bracket_values_k")
            rl, rh = pair("rate_bounds_k_s")
            value = number(record["root_value_k"], "root_value_k")
            direction = guard["direction"]
            if (
                lo > hi
                or lo < previous
                or hi != when
                or hi - lo > self.budget["root_time_atol_s"] + 2 * math.ulp(when)
                or not direction * gl < 0 <= direction * gr
                or value != gr
                or abs(value) > self.budget["root_value_atol_k"]
                or rl > rh
                or (
                    rl <= self.budget["minimum_transversal_rate_k_s"]
                    if direction == 1
                    else rh >= -self.budget["minimum_transversal_rate_k_s"]
                )
                or record["threshold_k"] != guard["threshold_k"]
                or type(record["direction"]) is not int
                or record["direction"] != direction
                or record["changes_sha256"] != digest(guard["changes"])
            ):
                raise NetworkError("State-event checkpoint root certificate mismatch")
            history.append(dict(time_s=when, changes=guard["changes"]))
            seen.add(name)
            previous = when
        runtime = ScheduledReducedGraph(
            scheduled_request(self._base, history), max_macro_step_s=self.max_step
        )
        runtime.restore(deepcopy(p["runtime_checkpoint"]))
        fresh = self._snapshot(runtime, fired)
        if fresh != state:
            raise NetworkError("State-event checkpoint observation mismatch")
        self._sources()
        self._runtime, self._fired = runtime, deepcopy(fired)
        return fresh
