"""Atomic scheduled and online controls for the registered reduced Radau graph.

Warm construction and the original differential reference frame are distinct.
Nested native checkpoints require this outer frame contract for restoration;
standalone legacy/native handoff is deliberately not exposed here.
"""

import bisect
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from core.network import NetworkError, number
from scripts.nonlinear_thermal_graph import digest
from scripts.reduced_residual_engine import ReducedResidualEngine
from scripts.registered_graph_system import graph_request
from scripts.registered_graph_system import signature as reduced_signature
from scripts.scheduled_thermal_graph import ScheduledThermalGraph
from scripts.scheduled_thermal_graph import signature as scheduled_signature
from scripts.solve_hydraulic_graph import _keys, sha

BUDGET = Path("validation/ai05_controlled_radau_budget_v1.json")


def signature():
    value = reduced_signature()
    other = scheduled_signature()
    for group in ("drivers", "budgets"):
        value[group].update(other[group])
    value["drivers"]["scripts/controlled_reduced_engine.py"] = sha(__file__)
    value["budgets"][str(BUDGET)] = sha(BUDGET)
    return value


def scheduled_request(model, events):
    return dict(
        format="atomic-scheduled-thermal-graph",
        schema_version=1,
        model_input=deepcopy(model),
        timeline=deepcopy(events),
    )


class ScheduledReducedGraph:
    def __init__(self, request, *, max_macro_step_s=2.0):
        _keys(request, ("format", "schema_version", "model_input", "timeline"))
        if (
            request["format"] != "atomic-scheduled-thermal-graph"
            or type(request["schema_version"]) is not int
            or request["schema_version"] != 1
        ):
            raise NetworkError("Expected scheduled graph schema1")
        self.budget = json.loads(BUDGET.read_text())
        self._base = deepcopy(request["model_input"])
        self._request = deepcopy(request)
        self._parser = ScheduledThermalGraph(scheduled_request(self._base, []))
        self._set_timeline(request["timeline"])
        initial_trial = deepcopy(self._base)
        initial_trial["initial_temperatures_k"] = dict(
            zip(self._parser.names, map(float, self._parser.offset))
        )
        self._engine = ReducedResidualEngine(
            graph_request(initial_trial), max_macro_step_s=max_macro_step_s
        )
        self.max_step = self._engine.max_step
        self._origin = self._engine.system.origin[self._engine.system.differential].copy()
        self._mass = self._engine.system.mass.copy()
        self._scales = self._engine.system.scales.copy()
        self._names = tuple(self._engine.system.adapter._names)
        self.identity = digest(dict(request=request, max_macro_step_s=self.max_step))
        self.captured = signature()
        self._cursor = 0
        self._receipts = []
        self._active = deepcopy(self._base)
        self._entry_time = 0.0
        self._entry_temperatures = {
            name: float(self._origin[i]) for i, name in enumerate(self._names)
        }
        if self._times and self._times[0] == 0.0:
            self._engine, self._active, receipt, _ = self._switch(self._engine, self._active, 0)
            self._receipts.append(receipt)
            self._cursor = 1

    def _set_timeline(self, events):
        if not isinstance(events, list) or len(events) > self.budget["maximum_events"]:
            raise NetworkError("Controlled event budget exceeded")
        self._timeline = self._parser._parse_timeline(events, self._parser._kernel)
        self._times = tuple(e["time_s"] for e in self._timeline)
        self._parser._timeline = self._timeline
        self._parser._times = self._times

    def _sources(self):
        if signature() != self.captured:
            raise NetworkError("Controlled Radau source/budget drift")

    def _new_engine(self, active, entry_temperatures):
        if not isinstance(entry_temperatures, dict) or set(entry_temperatures) != set(self._names):
            raise NetworkError("Controlled entry temperature map mismatch")
        temperatures = {
            name: number(entry_temperatures[name], "entry_temperature_k", positive=True)
            for name in self._names
        }
        trial = deepcopy(active)
        trial["initial_temperatures_k"] = temperatures
        candidate = ReducedResidualEngine(graph_request(trial), max_macro_step_s=self.max_step)
        if (
            tuple(candidate.system.adapter._names) != self._names
            or not np.array_equal(candidate.system.mass, self._mass)
            or not np.array_equal(candidate.system.scales, self._scales)
        ):
            raise NetworkError("Command changed controlled inventory frame")
        # The outer checkpoint binds this frame. Attach/restore the original
        # increments before publishing this private construction candidate.
        candidate.system.origin[candidate.system.differential] = self._origin
        return candidate

    def _switch(self, engine, active, index):
        event = self._timeline[index]
        left = engine.snapshot()
        if left["time_s"] != event["time_s"]:
            raise NetworkError("Controlled event clock mismatch")
        updated = self._parser._patch(active, event["changes"])
        candidate = self._new_engine(updated, left["state"]["temperatures_k"])
        candidate._increments = engine._increments.copy()
        candidate._full = candidate._project(candidate._increments, candidate._full)
        candidate._time = engine._time
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
            raise NetworkError("Control introduced a thermal/energy impulse")
        receipt = self._parser._receipt(index, active, updated)
        return (
            candidate,
            updated,
            receipt,
            dict(
                receipt=deepcopy(receipt),
                changes=deepcopy(event["changes"]),
                left=left,
                right=right,
            ),
        )

    def _snapshot(self, engine, cursor, active, receipts, entry_time, entry_temperatures):
        return dict(
            time_s=engine._time,
            active_event_count=cursor,
            active_model_sha256=digest(active),
            event_receipts=deepcopy(receipts),
            entry_time_s=entry_time,
            entry_temperatures_k=deepcopy(entry_temperatures),
            engine=engine.snapshot(),
        )

    def snapshot(self):
        return self._snapshot(
            self._engine,
            self._cursor,
            self._active,
            self._receipts,
            self._entry_time,
            self._entry_temperatures,
        )

    def advance(self, duration_s):
        self._sources()
        duration = number(duration_s, "duration_s", positive=True)
        start = self._engine._time
        target = start + duration
        if (
            duration > self.budget["maximum_duration_s"]
            or not math.isfinite(target)
            or target <= start
        ):
            raise NetworkError("Invalid controlled duration or unrepresentable clock")
        engine = deepcopy(self._engine)
        active = deepcopy(self._active)
        receipts = deepcopy(self._receipts)
        cursor = self._cursor
        entry_time = self._entry_time
        entry_temperatures = deepcopy(self._entry_temperatures)
        segments = []
        switches = []
        rhs_count = 0
        attempt_count = 0

        def integrate(end):
            nonlocal rhs_count, attempt_count
            if len(segments) >= self.budget["maximum_segments_per_advance"]:
                raise NetworkError("Controlled segment budget exhausted")
            remaining_rhs = self.budget["maximum_rhs_evaluations_per_advance"] - rhs_count
            remaining_attempts = self.budget["maximum_attempts_per_advance"] - attempt_count
            if min(remaining_rhs, remaining_attempts) <= 0:
                raise NetworkError("Controlled aggregate integration budget exhausted")
            before = engine._time
            original = deepcopy(engine.budget)
            engine.budget["maximum_rhs_evaluations_per_advance"] = min(
                original["maximum_rhs_evaluations_per_advance"], remaining_rhs
            )
            engine.budget["maximum_attempts"] = min(
                original["maximum_attempts"], remaining_attempts
            )
            try:
                result = engine.advance(end - before)
            finally:
                engine.budget = original
            adjustment = end - engine._time
            bound = self.budget["maximum_clock_alignment_ulps"] * max(
                math.ulp(before), math.ulp(end), math.ulp(end - before)
            )
            if abs(adjustment) > bound:
                raise NetworkError("Controlled absolute clock alignment exceeded")
            engine._time = end
            result["time_s"] = end
            rhs_count += result["rhs_evaluations"]
            attempt_count += len(result["segments"]) + result["rejected_macro_steps"]
            segments.append(
                dict(
                    start_s=before,
                    end_s=end,
                    clock_alignment_s=adjustment,
                    active_event_count=cursor,
                    result=result,
                )
            )

        while cursor < len(self._timeline) and self._times[cursor] <= target:
            time = self._times[cursor]
            if time <= engine._time:
                raise NetworkError("Controlled event would be replayed")
            integrate(time)
            engine, active, receipt, trace = self._switch(engine, active, cursor)
            entry_time = time
            entry_temperatures = deepcopy(trace["right"]["state"]["temperatures_k"])
            receipts.append(receipt)
            switches.append(trace)
            cursor += 1
        if engine._time < target:
            integrate(target)
        final = self._snapshot(engine, cursor, active, receipts, entry_time, entry_temperatures)
        if (
            abs(final["engine"]["state"]["energy_residual_j"])
            > self.budget["maximum_energy_error_j"]
        ):
            raise NetworkError("Controlled energy budget exceeded")
        self._sources()
        self._engine, self._active, self._cursor, self._receipts = engine, active, cursor, receipts
        self._entry_time, self._entry_temperatures = entry_time, entry_temperatures
        return dict(
            **final,
            segments=segments,
            switches=switches,
            rhs_evaluations=rhs_count,
            integration_attempts=attempt_count,
        )

    def checkpoint(self):
        payload = dict(
            schema_version=1,
            model="scheduled-reduced-radau-v1",
            input_sha256=self.identity,
            signature=deepcopy(self.captured),
            state=self.snapshot(),
            engine_checkpoint=self._engine.checkpoint(),
        )
        return dict(payload=payload, sha256=digest(payload))

    def restore(self, checkpoint):
        self._sources()
        _keys(checkpoint, ("payload", "sha256"))
        p = checkpoint["payload"]
        if digest(p) != checkpoint["sha256"]:
            raise NetworkError("Controlled checkpoint digest mismatch")
        _keys(
            p,
            ("schema_version", "model", "input_sha256", "signature", "state", "engine_checkpoint"),
        )
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["model"] != "scheduled-reduced-radau-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
        ):
            raise NetworkError("Controlled checkpoint input/model/source mismatch")
        state = p["state"]
        _keys(
            state,
            (
                "time_s",
                "active_event_count",
                "active_model_sha256",
                "event_receipts",
                "entry_time_s",
                "entry_temperatures_k",
                "engine",
            ),
        )
        time = number(state["time_s"], "time_s")
        cursor = state["active_event_count"]
        if time < 0 or type(cursor) is not int or cursor != bisect.bisect_right(self._times, time):
            raise NetworkError("Controlled checkpoint clock/cursor mismatch")
        entry_time = number(state["entry_time_s"], "entry_time_s")
        if entry_time != (self._times[cursor - 1] if cursor else 0.0):
            raise NetworkError("Controlled checkpoint entry clock mismatch")
        active = deepcopy(self._base)
        receipts = []
        for index in range(cursor):
            updated = self._parser._patch(active, self._timeline[index]["changes"])
            receipts.append(self._parser._receipt(index, active, updated))
            active = updated
        if digest(active) != state["active_model_sha256"] or receipts != state["event_receipts"]:
            raise NetworkError("Controlled checkpoint active commands mismatch")
        if entry_time == 0 and state["entry_temperatures_k"] != {
            name: float(self._origin[i]) for i, name in enumerate(self._names)
        }:
            raise NetworkError("Controlled initial entry frame mismatch")
        engine = self._new_engine(active, state["entry_temperatures_k"])
        engine.restore(deepcopy(p["engine_checkpoint"]))
        fresh = self._snapshot(
            engine, cursor, active, receipts, entry_time, state["entry_temperatures_k"]
        )
        if fresh != state:
            raise NetworkError("Controlled checkpoint state/frame mismatch")
        if (
            abs(fresh["engine"]["state"]["energy_residual_j"])
            > self.budget["maximum_energy_error_j"]
        ):
            raise NetworkError("Controlled checkpoint energy mismatch")
        self._sources()
        self._engine, self._active, self._cursor, self._receipts = engine, active, cursor, receipts
        self._entry_time, self._entry_temperatures = (
            entry_time,
            deepcopy(state["entry_temperatures_k"]),
        )
        return fresh


class OnlineReducedGraph:
    def __init__(self, model_input, *, max_macro_step_s=2.0):
        self._base = deepcopy(model_input)
        self._model = ScheduledReducedGraph(
            scheduled_request(self._base, []), max_macro_step_s=max_macro_step_s
        )
        self.max_step = self._model.max_step
        self.identity = digest(dict(model_input=model_input, max_macro_step_s=self.max_step))
        self._history = []
        self._revision = 0
        self.budget = json.loads(BUDGET.read_text())
        self.captured = signature()

    @property
    def revision(self):
        return self._revision

    def _sources(self):
        if signature() != self.captured:
            raise NetworkError("Online Radau source/budget drift")

    def observation(self):
        return dict(
            revision=self.revision,
            history_sha256=digest(self._history),
            model=self._model.snapshot(),
        )

    def step(self, changes, duration_s, *, expected_revision):
        self._sources()
        if type(expected_revision) is not int or expected_revision != self.revision:
            raise NetworkError("Stale online Radau command revision")
        if self.revision >= self.budget["maximum_events"]:
            raise NetworkError("Online Radau command budget exhausted")
        duration = number(duration_s, "duration_s", positive=True)
        time = self._model._engine._time
        if time + duration > self.budget["maximum_episode_s"]:
            raise NetworkError("Online Radau episode horizon exceeded")
        candidate = deepcopy(self._model)
        candidate.budget = deepcopy(self.budget)
        candidate._set_timeline(
            [*deepcopy(self._history), dict(time_s=time, changes=deepcopy(changes))]
        )
        history = deepcopy(list(candidate._timeline))
        request = scheduled_request(self._base, history)
        candidate._request = request
        candidate.identity = digest(dict(request=request, max_macro_step_s=self.max_step))
        engine, active, receipt, trace = candidate._switch(
            candidate._engine, candidate._active, candidate._cursor
        )
        candidate._engine, candidate._active = engine, active
        candidate._cursor += 1
        candidate._receipts.append(receipt)
        candidate._entry_time = time
        candidate._entry_temperatures = deepcopy(trace["right"]["state"]["temperatures_k"])
        integrated = candidate.advance(duration)
        observation = dict(
            revision=self.revision + 1, history_sha256=digest(history), model=candidate.snapshot()
        )
        report = dict(observation=observation, command_switch=trace, integration=integrated)
        self._sources()
        self._model, self._history = candidate, history
        self._revision += 1
        return report

    def checkpoint(self):
        payload = dict(
            schema_version=1,
            model="online-reduced-radau-v1",
            input_sha256=self.identity,
            signature=deepcopy(self.captured),
            revision=self.revision,
            history=deepcopy(self._history),
            scheduled_checkpoint=self._model.checkpoint(),
        )
        return dict(payload=payload, sha256=digest(payload))

    def restore(self, checkpoint):
        self._sources()
        _keys(checkpoint, ("payload", "sha256"))
        p = checkpoint["payload"]
        if digest(p) != checkpoint["sha256"]:
            raise NetworkError("Online Radau checkpoint digest mismatch")
        _keys(
            p,
            (
                "schema_version",
                "model",
                "input_sha256",
                "signature",
                "revision",
                "history",
                "scheduled_checkpoint",
            ),
        )
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["model"] != "online-reduced-radau-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
        ):
            raise NetworkError("Online Radau checkpoint input/model/source mismatch")
        revision, history = p["revision"], p["history"]
        if (
            type(revision) is not int
            or not isinstance(history, list)
            or revision != len(history)
            or not 0 <= revision <= self.budget["maximum_events"]
        ):
            raise NetworkError("Online Radau checkpoint revision/history mismatch")
        candidate = ScheduledReducedGraph(
            scheduled_request(self._base, history), max_macro_step_s=self.max_step
        )
        state = candidate.restore(deepcopy(p["scheduled_checkpoint"]))
        time = state["time_s"]
        if (
            time > self.budget["maximum_episode_s"]
            or state["active_event_count"] != revision
            or (revision == 0 and time != 0)
            or (revision > 0 and (history[0]["time_s"] != 0 or history[-1]["time_s"] >= time))
        ):
            raise NetworkError("Online Radau checkpoint has future/unheld/missing commands")
        result = dict(revision=revision, history_sha256=digest(history), model=state)
        self._sources()
        self._model, self._history, self._revision = candidate, deepcopy(history), revision
        return result
