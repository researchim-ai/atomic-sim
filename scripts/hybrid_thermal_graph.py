"""Atomic calendar, directional/arming guards and explicit-state online mode control.

One fixed-property reduced thermal/hydraulic graph family. No physical state
resets, arbitrary callbacks, topology changes, or unresolved root coalescing.
"""

import json
import math
import threading
from copy import deepcopy
from pathlib import Path

from core.network import NetworkError, identifier, number
from scripts.controlled_reduced_engine import ScheduledReducedGraph, scheduled_request
from scripts.hysteretic_radau import BUDGET as EVENT_BUDGET
from scripts.hysteretic_radau import _mask
from scripts.nonlinear_thermal_graph import digest
from scripts.solve_hydraulic_graph import _keys, sha
from scripts.state_event_radau import StateEventError, _advance_until
from scripts.state_event_radau import signature as event_signature

BUDGET = Path("validation/ai05_hybrid_graph_budget_v1.json")


def signature():
    value = event_signature()
    value["drivers"]["scripts/hybrid_thermal_graph.py"] = sha(__file__)
    value["drivers"]["scripts/hysteretic_radau.py"] = sha("scripts/hysteretic_radau.py")
    value["budgets"][str(BUDGET)] = sha(BUDGET)
    value["budgets"][str(EVENT_BUDGET)] = sha(EVENT_BUDGET)
    return value


def integer(value, label, lo, hi):
    if type(value) is not int or not lo <= value <= hi:
        raise NetworkError("Invalid " + label)
    return value


def finite_json(value, limit):
    # Serialize first: disallow nonfinite state, implicit custom Python objects,
    # non-string keys and JSON transformations that would break exact restart.
    raw = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    if len(raw.encode()) > limit or json.loads(raw) != value:
        raise NetworkError("Controller state is oversized or not canonical JSON")

    def check(v):
        if isinstance(v, dict):
            if any(type(k) is not str for k in v):
                raise NetworkError("Controller state keys must be strings")
            for item in v.values():
                check(item)
        elif isinstance(v, list):
            for item in v:
                check(item)
        elif type(v) not in (str, int, float, bool, type(None)):
            raise NetworkError("Controller state is not JSON")

    check(value)
    return json.loads(raw)


class HybridThermalGraph:
    def __init__(self, request, *, max_macro_step_s=2.0):
        _keys(
            request,
            ("format", "schema_version", "model_input", "initial_mode", "modes", "calendar"),
        )
        if (
            request["format"] != "atomic-hybrid-thermal-graph"
            or type(request["schema_version"]) is not int
            or request["schema_version"] != 1
        ):
            raise NetworkError("Expected hybrid thermal graph schema1")
        self.budget = json.loads(BUDGET.read_text())
        self.event_budget = json.loads(EVENT_BUDGET.read_text())
        self._request = deepcopy(request)
        self._frame = ScheduledReducedGraph(
            scheduled_request(request["model_input"], []), max_macro_step_s=max_macro_step_s
        )
        self.max_step = self._frame.max_step
        self._modes = self._parse_modes(request["modes"])
        initial = identifier(request["initial_mode"])
        if initial not in self._modes:
            raise NetworkError("Unknown initial mode")
        self._calendar = self._parse_calendar(request["calendar"])
        self.identity = digest(dict(request=request, max_macro_step_s=self.max_step))
        self.captured = signature()
        self._lock = threading.RLock()
        engine = self._build(initial, self._frame._engine)
        self._runtime = dict(
            engine=engine,
            mode=initial,
            entry_time=0.0,
            cursor=0,
            history=[],
            controller_state=None,
            revision=0,
            calls=[],
        )

    def _parse_modes(self, values):
        if not isinstance(values, list) or not 1 <= len(values) <= self.budget["maximum_modes"]:
            raise NetworkError("Hybrid mode budget")
        result, ids, mask = {}, set(), None
        kernel = self._frame._parser._kernel
        for mode in values:
            _keys(mode, ("id", "commands", "transitions"))
            name = identifier(mode["id"])
            if name in result:
                raise NetworkError("Duplicate hybrid mode")
            commands = self._frame._parser._parse_timeline(
                [dict(time_s=1.0, changes=mode["commands"])], kernel
            )[0]["changes"]
            current = _mask(commands)
            if mask is not None and current != mask:
                raise NetworkError("Modes require the same complete actuator mask")
            mask = current
            transitions = mode["transitions"]
            if not isinstance(transitions, list):
                raise NetworkError("Transitions must be a list")
            guards = []
            for g in transitions:
                _keys(
                    g, ("id", "node", "threshold_k", "direction", "to", "priority", "arm_delay_s")
                )
                gid = identifier(g["id"])
                if gid.startswith("online_") or gid in ids or g["node"] not in self._frame._names:
                    raise NetworkError("Duplicate guard or unknown node")
                ids.add(gid)
                threshold = number(g["threshold_k"], "threshold_k", positive=True)
                if not kernel._lower < threshold < kernel._upper:
                    raise NetworkError("Threshold outside temperature domain")
                direction = integer(g["direction"], "direction", -1, 1)
                if direction == 0:
                    raise NetworkError("Guard direction must be nonzero")
                delay = number(g["arm_delay_s"], "arm_delay_s")
                if not 0 <= delay <= self.budget["maximum_duration_s"]:
                    raise NetworkError("Invalid arm delay")
                guards.append(
                    dict(
                        id=gid,
                        node=g["node"],
                        threshold_k=threshold,
                        direction=direction,
                        to=identifier(g["to"]),
                        priority=integer(g["priority"], "priority", -1000, 1000),
                        arm_delay_s=delay,
                        index=self._frame._names.index(g["node"]),
                        offset_k=threshold
                        - float(self._frame._origin[self._frame._names.index(g["node"])]),
                    )
                )
            result[name] = dict(commands=commands, transitions=guards)
        if len(ids) > self.budget["maximum_transitions"] or any(
            g["to"] not in result for m in result.values() for g in m["transitions"]
        ):
            raise NetworkError("Transition budget or unknown target mode")
        return result

    def _parse_calendar(self, values):
        if not isinstance(values, list) or len(values) > self.budget["maximum_calendar_events"]:
            raise NetworkError("Hybrid calendar budget")
        result, ids = [], set()
        for event in values:
            _keys(event, ("id", "time_s", "to", "priority"))
            name = identifier(event["id"])
            when = number(event["time_s"], "time_s")
            if (
                name.startswith("online_")
                or name in ids
                or name in {g["id"] for m in self._modes.values() for g in m["transitions"]}
                or when < 0
                or event["to"] not in self._modes
            ):
                raise NetworkError("Invalid calendar entry")
            ids.add(name)
            result.append(
                dict(
                    id=name,
                    time_s=when,
                    to=event["to"],
                    priority=integer(event["priority"], "priority", -1000, 1000),
                )
            )
        return sorted(result, key=lambda e: (e["time_s"], e["id"]))

    def _sources(self):
        if signature() != self.captured:
            raise StateEventError("source_drift", "Hybrid sources/budgets changed")

    def _build(self, mode, previous):
        active = self._frame._parser._patch(
            self._request["model_input"], self._modes[mode]["commands"]
        )
        candidate = self._frame._new_engine(active, previous.snapshot()["state"]["temperatures_k"])
        candidate._increments = previous._increments.copy()
        candidate._full = candidate._project(candidate._increments, candidate._full)
        candidate._time = previous._time
        before, after = previous.snapshot(), candidate.snapshot()
        if (
            before["increments"] != after["increments"]
            or before["state"]["temperatures_k"] != after["state"]["temperatures_k"]
            or before["state"]["integrated_external_energy_j"]
            != after["state"]["integrated_external_energy_j"]
        ):
            raise StateEventError("mode_impulse", "Mode switch changed a physical stock")
        return candidate

    def _observe(self, state):
        engine = state["engine"].snapshot()
        return dict(
            time_s=engine["time_s"],
            mode=state["mode"],
            mode_entry_time_s=state["entry_time"],
            revision=state["revision"],
            calendar_cursor=state["cursor"],
            controller_state=deepcopy(state["controller_state"]),
            history=deepcopy(state["history"]),
            engine=engine,
            validated=False,
            plant_referenced=False,
        )

    def snapshot(self):
        with self._lock:
            self._sources()
            return self._observe(self._runtime)

    def _commit_event(self, state, contenders, kind, counters, root=None):
        if (
            len(state["history"]) >= self.budget["maximum_history_entries"]
            or counters["switches"] >= self.budget["maximum_switches_per_advance"]
        ):
            raise StateEventError("switch_budget", "Hybrid event history budget exhausted")
        contenders = sorted(contenders, key=lambda c: (-c["priority"], c["id"]))
        if len(contenders) > 1 and contenders[0]["priority"] == contenders[1]["priority"]:
            raise StateEventError(
                "priority_tie", "Simultaneous events require a unique highest priority"
            )
        winner = contenders[0]
        left = state["engine"].snapshot()
        previous_mode = state["mode"]
        state["engine"] = self._build(winner["to"], state["engine"])
        state["mode"], state["entry_time"] = winner["to"], state["engine"]._time
        right = state["engine"].snapshot()
        record = dict(
            sequence=len(state["history"]) + 1,
            time_s=state["entry_time"],
            kind=kind,
            source_mode=previous_mode,
            target_mode=state["mode"],
            winner=winner["id"],
            contenders=[dict(id=c["id"], priority=c["priority"], to=c["to"]) for c in contenders],
            stocks_sha256=digest(left["increments"]),
            left_state=left["state"],
            right_state=right["state"],
            root=deepcopy(root),
            parent_sha256=state["history"][-1]["sha256"] if state["history"] else self.identity,
        )
        record["sha256"] = digest(record)
        state["history"].append(record)
        counters["switches"] += 1

    def _arm(self, state, guard):
        arm = state["entry_time"] + guard["arm_delay_s"]
        if not math.isfinite(arm) or (guard["arm_delay_s"] > 0 and arm <= state["entry_time"]):
            raise NetworkError("Unrepresentable guard arm clock")
        return arm

    def _due(self, state, counters, online=None):
        now = state["engine"]._time
        calendar = []
        while (
            state["cursor"] < len(self._calendar)
            and self._calendar[state["cursor"]]["time_s"] == now
        ):
            calendar.append(self._calendar[state["cursor"]])
            state["cursor"] += 1
        guards = [
            g
            for g in self._modes[state["mode"]]["transitions"]
            if self._arm(state, g) <= now
            and g["direction"] * (float(state["engine"]._full[g["index"]]) - g["threshold_k"]) >= 0
        ]
        contenders = calendar + guards + ([online] if online else [])
        if contenders:
            self._commit_event(
                state, contenders, "boundary" if calendar or online else "guard_level", counters
            )
            return True
        return False

    def _execute(self, duration, command):
        self._sources()
        duration = number(duration, "duration_s", positive=True)
        state = deepcopy(self._runtime)
        target = state["engine"]._time + duration
        if (
            len(state["calls"]) >= self.budget["maximum_calls"]
            or duration > self.budget["maximum_duration_s"]
            or not math.isfinite(target)
            or target <= state["engine"]._time
        ):
            raise NetworkError("Hybrid duration/call budget exceeded")
        counters = dict(rhs=0, root=0, attempts=0, rejected=0, switches=0)
        online = None
        if command is not None:
            if command["expected_revision"] != state["revision"]:
                raise NetworkError("Stale online revision")
            online = dict(
                id="online_" + str(state["revision"]),
                to=command["mode"],
                priority=command["priority"],
            )
            state["controller_state"] = finite_json(
                command["controller_state"], self.budget["maximum_controller_state_bytes"]
            )
        parts = []
        while True:
            if self._due(state, counters, online):
                online = None
                continue
            online = None
            now = state["engine"]._time
            if now >= target:
                break
            bound = target
            if state["cursor"] < len(self._calendar):
                bound = min(bound, self._calendar[state["cursor"]]["time_s"])
            active = []
            for guard in self._modes[state["mode"]]["transitions"]:
                arm = self._arm(state, guard)
                if arm > now:
                    bound = min(bound, arm)
                else:
                    active.append(guard)
            # Identical predicates are one physical root with declared competing actions.
            groups = {}
            for guard in active:
                key = (guard["node"], guard["threshold_k"], guard["direction"])
                groups.setdefault(key, []).append(guard)
            pending = [v[0] for v in groups.values()]
            root, segments = _advance_until(
                state["engine"], bound, pending, counters, self.event_budget
            )
            parts.extend(segments)
            if root:
                contenders = next(v for v in groups.values() if v[0]["id"] == root["guard_id"])
                if root["bracket_s"][0] < bound <= root["bracket_s"][1] and bound < target:
                    raise StateEventError(
                        "boundary_root_overlap", "Root bracket overlaps a calendar/arming boundary"
                    )
                self._commit_event(state, contenders, "guard_root", counters, root)
        final = self._observe(state)
        if (
            abs(final["engine"]["state"]["energy_residual_j"])
            > self.budget["maximum_energy_error_j"]
        ):
            raise StateEventError("energy", "Hybrid energy budget exceeded")
        self._sources()
        state["calls"].append(dict(duration_s=duration, command=deepcopy(command)))
        state["revision"] += 1
        final = self._observe(state)
        result = dict(**final, segments=parts, evaluations=counters)
        self._checkpoint_state(state)
        self._sources()
        self._runtime = state
        return result

    def advance(self, duration_s):
        with self._lock:
            return self._execute(duration_s, None)

    def step(self, mode, duration_s, *, expected_revision, controller_state, priority=0):
        with self._lock:
            if mode not in self._modes:
                raise NetworkError("Unknown online mode")
            integer(expected_revision, "expected_revision", 0, self.budget["maximum_calls"])
            integer(priority, "priority", -1000, 1000)
            return self._execute(
                duration_s,
                dict(
                    mode=mode,
                    expected_revision=expected_revision,
                    controller_state=finite_json(
                        controller_state, self.budget["maximum_controller_state_bytes"]
                    ),
                    priority=priority,
                ),
            )

    def checkpoint(self):
        with self._lock:
            self._sources()
            return self._checkpoint_state(self._runtime)

    def _checkpoint_state(self, state):
        payload = dict(
            schema_version=1,
            model="hybrid-thermal-radau-v1",
            input_sha256=self.identity,
            signature=deepcopy(self.captured),
            calls=deepcopy(state["calls"]),
            observation=self._observe(state),
            native_checkpoint=state["engine"].checkpoint(),
        )
        checkpoint = dict(payload=payload, sha256=digest(payload))
        finite_json(checkpoint, self.budget["maximum_checkpoint_bytes"])
        return checkpoint

    def restore(self, checkpoint):
        with self._lock:
            self._sources()
            # Full accepted-call replay, not trust in a rehashed final-state snapshot.
            finite_json(checkpoint, self.budget["maximum_checkpoint_bytes"])
            _keys(checkpoint, ("payload", "sha256"))
            payload = checkpoint["payload"]
            if (
                checkpoint["sha256"] != digest(payload)
                or payload.get("signature") != self.captured
                or payload.get("input_sha256") != self.identity
            ):
                raise NetworkError("Hybrid checkpoint digest/source/input mismatch")
            calls = payload.get("calls")
            if not isinstance(calls, list) or len(calls) > self.budget["maximum_calls"]:
                raise NetworkError("Hybrid replay call budget")
            candidate = HybridThermalGraph(self._request, max_macro_step_s=self.max_step)
            for call in calls:
                _keys(call, ("duration_s", "command"))
                command = call["command"]
                if command is None:
                    candidate.advance(call["duration_s"])
                else:
                    _keys(command, ("mode", "expected_revision", "controller_state", "priority"))
                    candidate.step(
                        command["mode"],
                        call["duration_s"],
                        expected_revision=command["expected_revision"],
                        controller_state=command["controller_state"],
                        priority=command["priority"],
                    )
            if candidate.checkpoint() != checkpoint:
                raise NetworkError("Hybrid checkpoint full replay differs")
            self._sources()
            self._runtime = candidate._runtime
            return self._observe(self._runtime)
