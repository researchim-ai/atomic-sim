"""Atomic piecewise-constant schedules for the quasi-steady nonlinear graph.

Events are right-continuous, ordered exact float timestamps, never epsilon-
coalesced. Reinitialization preserves the original thermal/energy frame and
validates hydraulics at actual current temperatures. This is a source-bound
adapter to the existing engineering kernel, not a general DAE event engine.
"""

import argparse
import bisect
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from core.network import NetworkError, number
from scripts.nonlinear_thermal_graph import NonlinearThermalGraph, digest
from scripts.nonlinear_thermal_graph import signature as kernel_signature
from scripts.solve_hydraulic_graph import _keys, sha

BUDGET = Path("validation/ai05_scheduled_graph_budget_v1.json")
GROUPS = ("thermal_nodes", "hydraulic_nodes", "hydraulic_edges")
THERMAL_FIELDS = ("heat_power_w", "ambient_temperature_k", "supply_temperature_k")


def signature():
    result = kernel_signature()
    result["drivers"]["scripts/scheduled_thermal_graph.py"] = sha(__file__)
    result["budgets"][str(BUDGET)] = sha(BUDGET)
    return result


class ScheduledGraphError(NetworkError):
    def __init__(self, criterion, message, **details):
        super().__init__(message)
        self.report = dict(accepted=False, criterion=criterion, message=message, details=details)


class ScheduledThermalGraph:
    def __init__(self, spec):
        _keys(spec, ("format", "schema_version", "model_input", "timeline"))
        if (
            spec["format"] != "atomic-scheduled-thermal-graph"
            or type(spec["schema_version"]) is not int
            or spec["schema_version"] != 1
        ):
            raise NetworkError("Expected atomic-scheduled-thermal-graph schema 1")
        self.identity = digest(spec)
        self.budget = json.loads(BUDGET.read_text())
        self._template = deepcopy(spec["model_input"])
        base = NonlinearThermalGraph(self._template)
        self.names = list(base.names)
        self.offset = base.offset.copy()
        self.capacity = base.capacity.copy()
        self.scale = base.scale
        self.initial_energy = base.initial_energy
        self.max_step = base.max_step
        self._node_index = {n["id"]: i for i, n in enumerate(base.graph["nodes"])}
        self._edge_index = {e["id"]: i for i, e in enumerate(base.graph["edges"])}
        self._timeline = self._parse_timeline(spec["timeline"], base)
        self._times = tuple(e["time_s"] for e in self._timeline)
        self.captured = signature()
        self._kernel = base
        self._active = deepcopy(self._template)
        self._cursor = 0
        self._receipts = []
        while self._cursor < len(self._timeline) and self._timeline[self._cursor]["time_s"] == 0:
            self._kernel, self._active, receipt, _ = self._switch(
                self._kernel, self._active, self._cursor
            )
            self._receipts.append(receipt)
            self._cursor += 1

    def _parse_timeline(self, events, base):
        if not isinstance(events, list) or len(events) > self.budget["maximum_events"]:
            raise NetworkError("Timeline must be a bounded list")
        result = []
        previous = -1.0
        for event in events:
            _keys(event, ("time_s", "changes"))
            time = number(event["time_s"], "event_time_s")
            if time < 0 or time <= previous:
                raise NetworkError(
                    "Event times must be nonnegative and strictly increasing; simultaneous commands share one batch"
                )
            previous = time
            changes = event["changes"]
            _keys(changes, (), GROUPS)
            if not changes:
                raise NetworkError("Event changes must not be empty")
            normalized = {}
            for group, mapping in changes.items():
                if not isinstance(mapping, dict) or not mapping:
                    raise NetworkError("Command group requires a nonempty object")
                normalized[group] = {}
                for name, fields in mapping.items():
                    if group == "hydraulic_edges":
                        if name not in self._edge_index:
                            raise NetworkError("Unknown commanded edge")
                        allowed = ("pressure_gain_pa",)
                    else:
                        if name not in self._node_index:
                            raise NetworkError("Unknown commanded node")
                        node = base.graph["nodes"][self._node_index[name]]
                        allowed = (
                            THERMAL_FIELDS
                            if group == "thermal_nodes"
                            else (
                                ("pressure_pa",) if "pressure_pa" in node else ("injection_kg_s",)
                            )
                        )
                    _keys(fields, (), allowed)
                    if not fields:
                        raise NetworkError("Command field map must not be empty")
                    values = {}
                    for key, value in fields.items():
                        value = number(
                            value,
                            key,
                            positive=key == "pressure_pa" or key.endswith("temperature_k"),
                        )
                        if (
                            key == "supply_temperature_k"
                            and not base.cfg["min_temperature_k"]
                            < value
                            < base.cfg["max_temperature_k"]
                        ):
                            raise NetworkError("Scheduled supply temperature exceeds fluid domain")
                        if key == "pressure_pa" and not node.get(
                            "min_pressure_pa", 0
                        ) < value < node.get("max_pressure_pa", math.inf):
                            raise NetworkError(
                                "Scheduled pressure boundary exceeds declared domain"
                            )
                        values[key] = value
                    normalized[group][name] = values
            result.append(dict(time_s=time, changes=normalized))
        return tuple(result)

    def _patch(self, active, changes):
        result = deepcopy(active)
        for group, mapping in changes.items():
            for name, fields in mapping.items():
                if group == "thermal_nodes":
                    destination = result["steady_input"]["thermal"]["nodes"][name]
                elif group == "hydraulic_nodes":
                    destination = result["steady_input"]["hydraulic"]["nodes"][
                        self._node_index[name]
                    ]
                else:
                    destination = result["steady_input"]["hydraulic"]["edges"][
                        self._edge_index[name]
                    ]
                destination.update(deepcopy(fields))
        return result

    def _build(self, active, state, time, proposal):
        trial = deepcopy(active)
        actual = self.offset + state[:-1]
        trial["initial_temperatures_k"] = dict(zip(self.names, map(float, actual)))
        kernel = NonlinearThermalGraph(trial)
        if (
            kernel.names != self.names
            or not np.array_equal(kernel.capacity, self.capacity)
            or kernel.scale != self.scale
        ):
            raise ScheduledGraphError(
                "inventory_frame", "Control reinitialization changed inventory frame"
            )
        kernel.offset = self.offset.copy()
        kernel.initial_energy = self.initial_energy
        kernel.identity = digest(active)
        kernel._state = state.copy()
        kernel._time = time
        kernel._proposal = proposal
        kernel._evaluate(actual)
        return kernel

    def _receipt(self, index, before, after):
        event = self._timeline[index]
        return dict(
            event_index=index,
            time_s=event["time_s"],
            changes_sha256=digest(event["changes"]),
            before_model_sha256=digest(before),
            after_model_sha256=digest(after),
        )

    def _switch(self, kernel, active, index):
        event = self._timeline[index]
        if kernel._time != event["time_s"]:
            raise ScheduledGraphError(
                "event_clock", "Event must be applied at its declared timestamp"
            )
        left = kernel.snapshot()
        updated = self._patch(active, event["changes"])
        try:
            candidate = self._build(updated, kernel._state, kernel._time, self.max_step)
            right = candidate.snapshot()
        except NetworkError as error:
            raise ScheduledGraphError(
                "event_reinitialization",
                "Command batch is incompatible with the actual state; whole request rolled back",
                event_index=index,
                time_s=event["time_s"],
                cause=getattr(error, "report", dict(message=str(error))),
            ) from error
        continuity = (
            "time_s",
            "temperatures_k",
            "temperature_increments_k",
            "energy_integral_scaled_k",
            "integrated_external_energy_j",
            "stored_energy_j",
            "energy_residual_j",
        )
        if any(left[k] != right[k] for k in continuity):
            raise ScheduledGraphError(
                "event_impulse", "Control-only event changed a thermal stock or energy integral"
            )
        receipt = self._receipt(index, active, updated)
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

    def _snapshot(self, kernel, cursor, active, receipts):
        return dict(
            time_s=kernel._time,
            active_event_count=cursor,
            active_model_sha256=digest(active),
            event_receipts=deepcopy(receipts),
            state=kernel.snapshot(),
        )

    def snapshot(self):
        return self._snapshot(self._kernel, self._cursor, self._active, self._receipts)

    def rates(self):
        return self._kernel.rates()

    def _integrate_to(self, kernel, end, remaining_attempts):
        start = kernel._time
        duration = end - start
        if duration <= 0:
            raise ScheduledGraphError("event_clock", "Integration interval must progress")
        original_limit = kernel.budget["maximum_attempts"]
        kernel.budget["maximum_attempts"] = min(original_limit, remaining_attempts)
        try:
            result = kernel.advance(duration)
        except NetworkError as error:
            criterion = (
                "attempt_budget"
                if getattr(error, "report", {}).get("criterion") == "attempt_budget"
                else "segment_failure"
            )
            raise ScheduledGraphError(
                criterion,
                "Scheduled segment failed; whole request rolled back",
                start_s=start,
                end_s=end,
                cause=getattr(error, "report", dict(message=str(error))),
            ) from error
        finally:
            kernel.budget["maximum_attempts"] = original_limit
        adjustment = end - kernel._time
        bound = self.budget["maximum_clock_alignment_ulps"] * max(
            math.ulp(start), math.ulp(end), math.ulp(duration)
        )
        if abs(adjustment) > bound:
            raise ScheduledGraphError(
                "clock_alignment",
                "Absolute event clock differs beyond floating-point alignment budget",
            )
        kernel._time = end
        result["time_s"] = end
        return dict(start_s=start, end_s=end, clock_alignment_s=adjustment, result=result)

    def advance(self, duration_s):
        duration = number(duration_s, "duration_s", positive=True)
        target = self._kernel._time + duration
        if (
            duration > self.budget["maximum_duration_s"]
            or not math.isfinite(target)
            or target <= self._kernel._time
        ):
            raise NetworkError("Duration or representable time budget exceeded")
        if signature() != self.captured:
            raise ScheduledGraphError("source_drift", "Scheduled model sources changed")
        active = deepcopy(self._active)
        cursor = self._cursor
        receipts = deepcopy(self._receipts)
        working = self._build(
            active, self._kernel._state, self._kernel._time, self._kernel._proposal
        )
        switches = []
        segments = []
        attempts = 0

        def integrate(end):
            nonlocal attempts
            if len(segments) >= self.budget["maximum_segments_per_advance"]:
                raise ScheduledGraphError(
                    "segment_budget", "Scheduled segment budget exhausted; rolled back"
                )
            remaining_attempts = self.budget["maximum_attempts_per_advance"] - attempts
            if remaining_attempts <= 0:
                raise ScheduledGraphError(
                    "attempt_budget", "Combined integration attempt budget exhausted; rolled back"
                )
            segment = self._integrate_to(working, end, remaining_attempts)
            attempts += segment["result"]["accepted_steps"] + segment["result"]["rejected_steps"]
            segment.update(active_event_count=cursor, active_model_sha256=digest(active))
            segments.append(segment)

        while cursor < len(self._timeline) and self._timeline[cursor]["time_s"] <= target:
            event = self._timeline[cursor]
            if event["time_s"] <= working._time:
                raise ScheduledGraphError("cursor", "An already-applied event would be repeated")
            integrate(event["time_s"])
            working, active, receipt, trace = self._switch(working, active, cursor)
            receipts.append(receipt)
            switches.append(trace)
            cursor += 1
        if working._time < target:
            integrate(target)
        if len(segments) > self.budget["maximum_segments_per_advance"]:
            raise ScheduledGraphError(
                "segment_budget", "Scheduled segment budget exhausted; rolled back"
            )
        final = self._snapshot(working, cursor, active, receipts)
        if abs(final["state"]["energy_residual_j"]) > self.budget["maximum_energy_error_j"]:
            raise ScheduledGraphError(
                "integrated_energy", "Scheduled energy budget exceeded; rolled back"
            )
        if signature() != self.captured:
            raise ScheduledGraphError(
                "source_drift", "Sources changed during scheduled advance; rolled back"
            )
        self._kernel = working
        self._cursor = cursor
        self._active = active
        self._receipts = receipts
        return dict(**final, segments=segments, switches=switches, integration_attempts=attempts)

    def checkpoint(self):
        payload = dict(
            schema_version=1,
            model="scheduled-thermal-graph-v1",
            input_sha256=self.identity,
            signature=deepcopy(self.captured),
            state=self.snapshot(),
        )
        return dict(payload=payload, sha256=digest(payload))

    def restore(self, checkpoint):
        _keys(checkpoint, ("payload", "sha256"))
        p = checkpoint["payload"]
        if digest(p) != checkpoint["sha256"]:
            raise NetworkError("Scheduled checkpoint digest mismatch")
        _keys(p, ("schema_version", "model", "input_sha256", "signature", "state"))
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["model"] != "scheduled-thermal-graph-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
            or signature() != self.captured
        ):
            raise NetworkError("Scheduled checkpoint model/input/source mismatch")
        s = p["state"]
        _keys(s, ("time_s", "active_event_count", "active_model_sha256", "event_receipts", "state"))
        time = number(s["time_s"], "checkpoint_time_s")
        cursor = s["active_event_count"]
        if time < 0 or type(cursor) is not int or cursor != bisect.bisect_right(self._times, time):
            raise NetworkError("Scheduled checkpoint time/cursor is not right-continuous")
        active = deepcopy(self._template)
        receipts = []
        for index in range(cursor):
            updated = self._patch(active, self._timeline[index]["changes"])
            receipts.append(self._receipt(index, active, updated))
            active = updated
        if digest(active) != s["active_model_sha256"] or receipts != s["event_receipts"]:
            raise NetworkError("Scheduled checkpoint active commands/receipts mismatch")
        state = s["state"]
        _keys(state, tuple(self._kernel.snapshot()))
        if state["time_s"] != time:
            raise NetworkError("Scheduled checkpoint contains two different clocks")
        if not isinstance(state["temperature_increments_k"], dict) or set(
            state["temperature_increments_k"]
        ) != set(self.names):
            raise NetworkError("Scheduled checkpoint increment map mismatch")
        increment = np.array(
            [
                number(state["temperature_increments_k"][k], "temperature_increment_k")
                for k in self.names
            ]
        )
        energy = number(state["energy_integral_scaled_k"], "scaled_energy")
        proposal = number(state["step_proposal_s"], "step_proposal_s", positive=True)
        candidate = self._build(active, np.r_[increment, energy], time, proposal)
        payload = dict(
            schema_version=1,
            model="nonlinear-thermal-graph-v1",
            input_sha256=candidate.identity,
            signature=deepcopy(candidate.captured),
            state=state,
        )
        candidate.restore(dict(payload=payload, sha256=digest(payload)))
        final = self._snapshot(candidate, cursor, active, receipts)
        if abs(final["state"]["energy_residual_j"]) > self.budget["maximum_energy_error_j"]:
            raise NetworkError("Scheduled checkpoint exceeds global energy budget")
        if signature() != self.captured:
            raise ScheduledGraphError("source_drift", "Sources changed during restore")
        self._kernel = candidate
        self._cursor = cursor
        self._active = active
        self._receipts = receipts
        return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.input.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "request.json").write_text(json.dumps(spec, indent=2) + "\n")
    try:
        model = ScheduledThermalGraph(spec)
        if args.resume:
            model.restore(json.loads(args.resume.read_text()))
        report = dict(
            passed=True,
            initial=model.snapshot(),
            final=model.advance(args.duration_s),
            signature=model.captured,
            validated=False,
            plant_referenced=False,
        )
        (args.output / "checkpoint.json").write_text(
            json.dumps(model.checkpoint(), indent=2, allow_nan=False) + "\n"
        )
        code = 0
    except NetworkError as error:
        report = dict(passed=False, diagnostics=getattr(error, "report", dict(message=str(error))))
        code = 2
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    manifest = dict(
        signature=signature(),
        files={p.name: sha(p) for p in sorted(args.output.iterdir())},
        exit_code=code,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
