"""Repeatable temperature transitions in an explicit finite-mode Radau graph.

Every mode specifies the same complete actuator mask. Entering a mode requires
all outgoing guards to be strictly on the pre-crossing side. Minimum residence
is certified by the root bracket; it never suppresses or delays a crossing.
"""

import json
import math
from copy import deepcopy
from pathlib import Path

from core.network import NetworkError, identifier, number
from scripts.controlled_reduced_engine import ScheduledReducedGraph, scheduled_request
from scripts.nonlinear_thermal_graph import digest
from scripts.solve_hydraulic_graph import _keys, sha
from scripts.state_event_radau import StateEventError, _advance_until
from scripts.state_event_radau import signature as event_signature

BUDGET = Path("validation/ai05_hysteresis_budget_v1.json")


def signature():
    value = event_signature()
    value["drivers"]["scripts/hysteretic_radau.py"] = sha(__file__)
    value["budgets"][str(BUDGET)] = sha(BUDGET)
    return value


def _mask(commands):
    return tuple(
        sorted(
            (group, name, field)
            for group, mapping in commands.items()
            for name, fields in mapping.items()
            for field in fields
        )
    )


class HystereticRadauGraph:
    def __init__(self, request, *, max_macro_step_s=2.0):
        _keys(
            request,
            (
                "format",
                "schema_version",
                "model_input",
                "initial_mode",
                "minimum_residence_s",
                "modes",
            ),
        )
        if (
            request["format"] != "atomic-hysteretic-graph"
            or type(request["schema_version"]) is not int
            or request["schema_version"] != 1
        ):
            raise NetworkError("Expected hysteretic graph schema1")
        self.budget = json.loads(BUDGET.read_text())
        self._base = deepcopy(request["model_input"])
        self._frame = ScheduledReducedGraph(
            scheduled_request(self._base, []), max_macro_step_s=max_macro_step_s
        )
        self.max_step = self._frame.max_step
        self._initial_mode = identifier(request["initial_mode"])
        self.minimum_residence = number(
            request["minimum_residence_s"], "minimum_residence_s", positive=True
        )
        if self.minimum_residence < self.budget["minimum_residence_s"]:
            raise NetworkError("Residence constraint is below supported event resolution")
        self._modes, self._by_id = self._parse(request["modes"])
        self.identity = digest(dict(request=request, max_macro_step_s=self.max_step))
        self.captured = signature()
        self._mode = self._initial_mode
        self._entry_time = 0.0
        self._history = []
        self._entry_temperatures = {
            name: float(self._frame._origin[i]) for i, name in enumerate(self._frame._names)
        }
        z = self._frame._engine._increments.copy()
        self._engine = self._build(self._mode, self._entry_temperatures, z, 0.0)

    def _parse(self, modes):
        if not isinstance(modes, list) or not 1 <= len(modes) <= self.budget["maximum_modes"]:
            raise NetworkError("Mode count exceeds contract")
        result = {}
        by_id = {}
        mask = None
        total = 0
        k = self._frame._engine.system.adapter._kernel
        for mode in modes:
            _keys(mode, ("id", "commands", "transitions"))
            name = identifier(mode["id"])
            if name in result:
                raise NetworkError("Duplicate mode ID")
            commands = self._frame._parser._parse_timeline(
                [dict(time_s=1.0, changes=mode["commands"])], self._frame._parser._kernel
            )[0]["changes"]
            current_mask = _mask(commands)
            if mask is not None and current_mask != mask:
                raise NetworkError("All modes require the same complete actuator mask")
            mask = current_mask
            transitions = mode["transitions"]
            if (
                not isinstance(transitions, list)
                or len(transitions) > self.budget["maximum_active_guards"]
            ):
                raise NetworkError("Mode transition count exceeds contract")
            normalized = []
            keys = set()
            for transition in transitions:
                _keys(transition, ("id", "node", "threshold_k", "direction", "to"))
                tid = identifier(transition["id"])
                target = identifier(transition["to"])
                node = transition["node"]
                if tid in by_id or not isinstance(node, str) or node not in self._frame._names:
                    raise NetworkError("Duplicate transition or unknown guard node")
                threshold = number(transition["threshold_k"], "threshold_k", positive=True)
                direction = transition["direction"]
                if (
                    type(direction) is not int
                    or direction not in (-1, 1)
                    or not k._lower < threshold < k._upper
                ):
                    raise NetworkError("Invalid transition direction or threshold")
                if target == name:
                    raise NetworkError("Self-transitions without state reset are unsupported")
                key = (node, threshold, direction)
                if key in keys:
                    raise NetworkError("Duplicate outgoing directional threshold")
                i = self._frame._names.index(node)
                guard = dict(
                    id=tid,
                    node=node,
                    index=i,
                    threshold_k=threshold,
                    direction=direction,
                    offset_k=float(threshold - self._frame._origin[i]),
                    source_mode=name,
                    target_mode=target,
                )
                normalized.append(guard)
                by_id[tid] = guard
                keys.add(key)
                total += 1
            result[name] = dict(commands=commands, transitions=tuple(normalized))
        if total > self.budget["maximum_transitions"] or self._initial_mode not in result:
            raise NetworkError("Unknown initial mode or excessive transition graph")
        if any(g["target_mode"] not in result for g in by_id.values()):
            raise NetworkError("Transition target does not exist")
        reachable = {self._initial_mode}
        while True:
            extended = reachable | {
                g["target_mode"] for name in reachable for g in result[name]["transitions"]
            }
            if extended == reachable:
                break
            reachable = extended
        if reachable != set(result):
            raise NetworkError("Declared mode graph contains unreachable modes")
        return result, by_id

    def _sources(self):
        if signature() != self.captured:
            raise StateEventError("source_drift", "Hysteretic Radau sources/budgets changed")

    def _entry_check(self, mode, temperatures):
        for guard in self._modes[mode]["transitions"]:
            residual = (
                number(temperatures[guard["node"]], "entry_temperature_k") - guard["threshold_k"]
            )
            if guard["direction"] * residual >= -self.budget["entry_guard_margin_k"]:
                raise StateEventError(
                    "entry_guard",
                    "Destination guard is active or unresolved at mode entry",
                    mode=mode,
                    guard=guard["id"],
                    residual_k=residual,
                )

    def _build(self, mode, temperatures, increments, time):
        self._entry_check(mode, temperatures)
        active = self._frame._parser._patch(self._base, self._modes[mode]["commands"])
        engine = self._frame._new_engine(active, temperatures)
        engine._increments = increments.copy()
        engine._full = engine._project(engine._increments, engine._full)
        engine._time = time
        return engine

    def _snapshot(self, engine, mode, entry_time, entry_temperatures, history):
        return dict(
            time_s=engine._time,
            mode=mode,
            mode_entry_time_s=entry_time,
            mode_entry_temperatures_k=deepcopy(entry_temperatures),
            transition_count=len(history),
            history=deepcopy(history),
            active_transitions=[g["id"] for g in self._modes[mode]["transitions"]],
            active_commands_sha256=digest(self._modes[mode]["commands"]),
            engine=engine.snapshot(),
        )

    def snapshot(self):
        return self._snapshot(
            self._engine, self._mode, self._entry_time, self._entry_temperatures, self._history
        )

    def advance(self, duration_s, *, stop_at_transition=False):
        self._sources()
        if type(stop_at_transition) is not bool:
            raise NetworkError("stop_at_transition must be boolean")
        duration = number(duration_s, "duration_s", positive=True)
        target = self._engine._time + duration
        if (
            duration > self.budget["maximum_duration_s"]
            or not math.isfinite(target)
            or target <= self._engine._time
        ):
            raise NetworkError("Invalid hysteretic duration or clock")
        engine = deepcopy(self._engine)
        mode = self._mode
        entry_time = self._entry_time
        entry_temperatures = deepcopy(self._entry_temperatures)
        history = deepcopy(self._history)
        segments = []
        switches = []
        counters = dict(rhs=0, root=0, attempts=0, rejected=0)
        while engine._time < target:
            pending = list(self._modes[mode]["transitions"])
            root, part = _advance_until(engine, target, pending, counters, self.budget)
            segments.extend(part)
            if root is None:
                break
            if (
                len(switches) >= self.budget["maximum_switches_per_advance"]
                or len(history) >= self.budget["maximum_history_entries"]
            ):
                raise StateEventError(
                    "switch_budget", "Transition history or per-call switch budget exhausted"
                )
            guard = self._by_id[root["guard_id"]]
            if root["bracket_s"][0] - entry_time < self.minimum_residence:
                raise StateEventError(
                    "minimum_residence",
                    "Root bracket does not establish minimum mode residence",
                    mode=mode,
                    entry_time_s=entry_time,
                    root_bracket_s=root["bracket_s"],
                    minimum_residence_s=self.minimum_residence,
                )
            left = engine.snapshot()
            next_mode = guard["target_mode"]
            temperatures = left["state"]["temperatures_k"]
            candidate = self._build(next_mode, temperatures, engine._increments, engine._time)
            right = candidate.snapshot()
            if left["increments"] != right["increments"] or any(
                left["state"][key] != right["state"][key]
                for key in (
                    "temperatures_k",
                    "temperature_increments_k",
                    "integrated_external_energy_j",
                    "energy_residual_j",
                )
            ):
                raise StateEventError(
                    "mode_impulse", "Mode transition changed a differential stock"
                )
            record = dict(
                **root,
                sequence=len(history) + 1,
                source_mode=mode,
                target_mode=next_mode,
                source_commands_sha256=digest(self._modes[mode]["commands"]),
                target_commands_sha256=digest(self._modes[next_mode]["commands"]),
            )
            history.append(record)
            switches.append(dict(transition=deepcopy(record), left=left, right=right))
            engine = candidate
            mode = next_mode
            entry_time = root["time_s"]
            entry_temperatures = deepcopy(temperatures)
            if stop_at_transition:
                break
        final = self._snapshot(engine, mode, entry_time, entry_temperatures, history)
        if (
            abs(final["engine"]["state"]["energy_residual_j"])
            > self.budget["maximum_energy_error_j"]
        ):
            raise StateEventError("energy", "Hysteretic global energy budget exceeded")
        self._sources()
        self._engine, self._mode, self._entry_time, self._entry_temperatures, self._history = (
            engine,
            mode,
            entry_time,
            entry_temperatures,
            history,
        )
        return dict(
            **final,
            segments=segments,
            switches=switches,
            evaluations=counters,
            requested_time_s=target,
            stopped_at_transition=stop_at_transition and bool(switches),
        )

    def checkpoint(self):
        payload = dict(
            schema_version=1,
            model="hysteretic-radau-v1",
            input_sha256=self.identity,
            signature=deepcopy(self.captured),
            state=self.snapshot(),
            native_checkpoint=self._engine.checkpoint(),
        )
        return dict(payload=payload, sha256=digest(payload))

    def restore(self, checkpoint):
        self._sources()
        _keys(checkpoint, ("payload", "sha256"))
        p = checkpoint["payload"]
        if digest(p) != checkpoint["sha256"]:
            raise NetworkError("Hysteretic checkpoint digest mismatch")
        _keys(
            p,
            ("schema_version", "model", "input_sha256", "signature", "state", "native_checkpoint"),
        )
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["model"] != "hysteretic-radau-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
        ):
            raise NetworkError("Hysteretic checkpoint input/model/source mismatch")
        state = p["state"]
        _keys(
            state,
            (
                "time_s",
                "mode",
                "mode_entry_time_s",
                "mode_entry_temperatures_k",
                "transition_count",
                "history",
                "active_transitions",
                "active_commands_sha256",
                "engine",
            ),
        )
        time = number(state["time_s"], "time_s")
        entry_time = number(state["mode_entry_time_s"], "entry_time_s")
        history = state["history"]
        count = state["transition_count"]
        if (
            time < 0
            or not isinstance(history, list)
            or type(count) is not int
            or count != len(history)
            or count > self.budget["maximum_history_entries"]
        ):
            raise NetworkError("Hysteretic checkpoint history malformed")
        mode = self._initial_mode
        previous = 0.0
        for sequence, record in enumerate(history, 1):
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
                    "sequence",
                    "source_mode",
                    "target_mode",
                    "source_commands_sha256",
                    "target_commands_sha256",
                ),
            )
            tid = record["guard_id"]
            if not isinstance(tid, str) or tid not in self._by_id:
                raise NetworkError("Hysteretic checkpoint transition ID mismatch")
            guard = self._by_id[tid]
            target = guard["target_mode"]
            when = number(record["time_s"], "event_time_s")
            if (
                type(record["sequence"]) is not int
                or record["sequence"] != sequence
                or record["source_mode"] != mode
                or guard["source_mode"] != mode
                or record["target_mode"] != target
                or when <= previous
                or when > time
            ):
                raise NetworkError("Hysteretic checkpoint transition path mismatch")

            def pair(key):
                values = record[key]
                if not isinstance(values, list) or len(values) != 2:
                    raise NetworkError("Hysteretic root pair malformed")
                return [number(v, key) for v in values]

            lo, hi = pair("bracket_s")
            gl, gr = pair("bracket_values_k")
            rl, rh = pair("rate_bounds_k_s")
            value = number(record["root_value_k"], "root_value_k")
            direction = guard["direction"]
            if (
                lo > hi
                or lo - previous < self.minimum_residence
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
                or record["source_commands_sha256"] != digest(self._modes[mode]["commands"])
                or record["target_commands_sha256"] != digest(self._modes[target]["commands"])
            ):
                raise NetworkError("Hysteretic checkpoint root/residence certificate mismatch")
            mode = target
            previous = when
        if state["mode"] != mode or entry_time != previous or entry_time > time:
            raise NetworkError("Hysteretic checkpoint active mode/entry mismatch")
        temperatures = state["mode_entry_temperatures_k"]
        if not isinstance(temperatures, dict) or set(temperatures) != set(self._frame._names):
            raise NetworkError("Hysteretic checkpoint entry map malformed")
        temperatures = {
            name: number(temperatures[name], "entry_temperature_k", positive=True)
            for name in self._frame._names
        }
        initial = {name: float(self._frame._origin[i]) for i, name in enumerate(self._frame._names)}
        if not history and temperatures != initial:
            raise NetworkError("Hysteretic initial entry frame mismatch")
        self._entry_check(mode, temperatures)
        if history:
            guard = self._by_id[history[-1]["guard_id"]]
            entry_value = temperatures[guard["node"]] - guard["threshold_k"]
            if abs(entry_value - history[-1]["root_value_k"]) > 4 * math.ulp(guard["threshold_k"]):
                raise NetworkError("Hysteretic checkpoint entry/root mismatch")
        active = self._frame._parser._patch(self._base, self._modes[mode]["commands"])
        engine = self._frame._new_engine(active, temperatures)
        engine.restore(deepcopy(p["native_checkpoint"]))
        fresh = self._snapshot(engine, mode, entry_time, temperatures, history)
        if fresh != state:
            raise NetworkError("Hysteretic checkpoint observation mismatch")
        if time == entry_time and fresh["engine"]["state"]["temperatures_k"] != temperatures:
            raise NetworkError("Hysteretic checkpoint entry state mismatch")
        if (
            abs(fresh["engine"]["state"]["energy_residual_j"])
            > self.budget["maximum_energy_error_j"]
        ):
            raise NetworkError("Hysteretic checkpoint energy mismatch")
        self._sources()
        self._engine, self._mode, self._entry_time, self._entry_temperatures, self._history = (
            engine,
            mode,
            entry_time,
            temperatures,
            deepcopy(history),
        )
        return fresh
