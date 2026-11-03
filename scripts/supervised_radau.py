"""Serialized supervisory mode control with calendar, online and thermal events.

No implicit event priorities: uncertain calendar/root ordering is refused.
No-op commands are acknowledged without rearming or resetting mode residence.
External commands select declared modes; they never modify arbitrary physics.
"""

import bisect
import json
import math
from copy import deepcopy
from pathlib import Path

from core.network import NetworkError, identifier, number
from scripts.hysteretic_radau import HystereticRadauGraph, _mask
from scripts.hysteretic_radau import signature as mode_signature
from scripts.nonlinear_thermal_graph import digest
from scripts.solve_hydraulic_graph import _keys, sha
from scripts.state_event_radau import StateEventError, _advance_until

BUDGET = Path("validation/ai05_supervision_budget_v1.json")


def signature():
    value = mode_signature()
    value["drivers"]["scripts/supervised_radau.py"] = sha(__file__)
    value["budgets"][str(BUDGET)] = sha(BUDGET)
    return value


class _SupervisoryProgram(HystereticRadauGraph):
    """Use declared external entry modes as additional reachability roots.

    Each closed reachable subgraph is validated by the original mode parser. No
    synthetic transitions are inserted, and global masks/ID/resource limits remain.
    """

    def __init__(self, request, seeds, **kwargs):
        self._external_seeds = tuple(seeds)
        super().__init__(request, **kwargs)

    def _parse(self, modes):
        if not isinstance(modes, list) or not 1 <= len(modes) <= self.budget["maximum_modes"]:
            raise NetworkError("Supervised mode count exceeds contract")
        raw = {}
        ids = set()
        edges = {}
        for mode in modes:
            _keys(mode, ("id", "commands", "transitions"))
            name = identifier(mode["id"])
            if name in raw or not isinstance(mode["transitions"], list):
                raise NetworkError("Duplicate mode or malformed transition list")
            raw[name] = mode
            edges[name] = []
            for transition in mode["transitions"]:
                _keys(transition, ("id", "node", "threshold_k", "direction", "to"))
                tid = identifier(transition["id"])
                target = identifier(transition["to"])
                if tid in ids:
                    raise NetworkError("Duplicate global transition ID")
                ids.add(tid)
                edges[name].append(target)
        seeds = {self._initial_mode, *self._external_seeds}
        if not seeds <= set(raw) or any(
            t not in raw for targets in edges.values() for t in targets
        ):
            raise NetworkError("Unknown supervised entry/target mode")
        normalized = {}
        by_id = {}
        saved = self._initial_mode
        try:
            for seed in sorted(seeds):
                reachable = {seed}
                while True:
                    extended = reachable | {target for node in reachable for target in edges[node]}
                    if extended == reachable:
                        break
                    reachable = extended
                self._initial_mode = seed
                part, guards = super()._parse(
                    [mode for name, mode in raw.items() if name in reachable]
                )
                normalized.update(part)
                by_id.update(guards)
        finally:
            self._initial_mode = saved
        if set(normalized) != set(raw):
            raise NetworkError(
                "Mode is unreachable from autonomous or declared supervisory entries"
            )
        if (
            len(by_id) > self.budget["maximum_transitions"]
            or len({_mask(m["commands"]) for m in normalized.values()}) != 1
        ):
            raise NetworkError("Global supervised transition count or actuator mask mismatch")
        return normalized, by_id


class SupervisedRadauGraph:
    def __init__(self, request, *, max_macro_step_s=2.0):
        _keys(request, ("format", "schema_version", "controller", "schedule", "online_modes"))
        if (
            request["format"] != "atomic-supervised-mode-graph"
            or type(request["schema_version"]) is not int
            or request["schema_version"] != 1
        ):
            raise NetworkError("Expected supervised graph schema1")
        self.budget = json.loads(BUDGET.read_text())
        self._request = deepcopy(request)
        schedule = request["schedule"]
        online = request["online_modes"]
        if (
            not isinstance(schedule, list)
            or len(schedule) > self.budget["maximum_schedule_entries"]
            or not isinstance(online, list)
        ):
            raise NetworkError("Malformed supervised schedule/permissions")
        normalized = []
        ids = set()
        previous = 0.0
        for item in schedule:
            _keys(item, ("id", "time_s", "to"))
            eid = identifier(item["id"])
            target = identifier(item["to"])
            time = number(item["time_s"], "calendar_time_s", positive=True)
            if eid in ids or time <= previous:
                raise NetworkError("Calendar IDs/timestamps must be unique and strictly ordered")
            normalized.append(dict(id=eid, time_s=time, to=target))
            ids.add(eid)
            previous = time
        if any(not isinstance(v, str) for v in online) or len(set(online)) != len(online):
            raise NetworkError("Online mode permissions require unique names")
        self._online_modes = frozenset(identifier(v) for v in online)
        self._schedule = tuple(normalized)
        self._times = tuple(e["time_s"] for e in normalized)
        seeds = [e["to"] for e in normalized] + list(online)
        self._program = _SupervisoryProgram(
            deepcopy(request["controller"]), seeds, max_macro_step_s=max_macro_step_s
        )
        self.max_step = self._program.max_step
        self.identity = digest(dict(request=request, max_macro_step_s=self.max_step))
        self.captured = signature()
        self._engine = deepcopy(self._program._engine)
        self._mode = self._program._mode
        self._entry_time = 0.0
        self._entry_temperatures = deepcopy(self._program._entry_temperatures)
        self._history = []
        self._cursor = 0
        self._revision = 0

    @property
    def revision(self):
        return self._revision

    def _sources(self):
        if signature() != self.captured:
            raise StateEventError("source_drift", "Supervised Radau sources/budgets changed")

    def _near_calendar(self, root):
        gap = self.budget["minimum_calendar_root_gap_s"]
        lo, hi = root["bracket_s"]
        index = bisect.bisect_left(self._times, lo - gap)
        return index < len(self._times) and self._times[index] <= hi + gap

    def _command_id(self, value):
        if (
            not isinstance(value, str)
            or not 1 <= len(value) <= self.budget["maximum_command_id_length"]
            or not value.isascii()
            or any(not (c.isalnum() or c in "-_.:") for c in value)
        ):
            raise NetworkError("Invalid online command ID")
        return value

    def _budget_event(self, counters, changed):
        if (
            len(self._history) >= self.budget["maximum_history_entries"]
            or counters["events"] >= self.budget["maximum_events_per_advance"]
            or (changed and counters["switches"] >= self.budget["maximum_switches_per_advance"])
        ):
            raise StateEventError(
                "event_budget", "Supervisory event/switch/history budget exhausted"
            )

    def _change(self, target, time, age_lower):
        left = self._engine.snapshot()
        changed = target != self._mode
        if changed:
            if age_lower < self._program.minimum_residence:
                raise StateEventError(
                    "minimum_residence",
                    "Supervisory change violates mode residence",
                    mode=self._mode,
                    target=target,
                    age_lower_s=age_lower,
                )
            temperatures = left["state"]["temperatures_k"]
            engine = self._program._build(target, temperatures, self._engine._increments, time)
            right = engine.snapshot()
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
                    "mode_impulse", "Supervisory command changed a differential stock"
                )
            self._engine = engine
            self._entry_time = time
            self._entry_temperatures = deepcopy(temperatures)
        else:
            right = deepcopy(left)
        self._mode = target
        return left, right, changed

    def _record(
        self, kind, source, target, when, age_lower, changed, commit_revision, extra, counters
    ):
        record = dict(
            kind=kind,
            time_s=when,
            sequence=len(self._history) + 1,
            source_mode=source,
            target_mode=target,
            changed=changed,
            commit_revision=commit_revision,
            residence_lower_bound_s=age_lower,
            source_commands_sha256=digest(self._program._modes[source]["commands"]),
            target_commands_sha256=digest(self._program._modes[target]["commands"]),
            entry_frame_sha256=digest(self._entry_temperatures),
            **extra,
        )
        self._history.append(record)
        counters["events"] += 1
        counters["switches"] += int(changed)
        return record

    def _force(self, kind, target, commit_revision, extra, counters):
        if target not in self._program._modes:
            raise NetworkError("Unknown supervisory mode")
        engine = self._engine
        for guard in self._program._modes[self._mode]["transitions"]:
            value = engine._increments[guard["index"]] - guard["offset_k"]
            if abs(value) <= self.budget["root_value_atol_k"] or guard["direction"] * value >= 0:
                raise StateEventError(
                    "ambiguous_event_order",
                    "Supervisory command meets unresolved active guard",
                    guard=guard["id"],
                )
        source = self._mode
        time = engine._time
        age = time - self._entry_time
        changed = target != source
        self._budget_event(counters, changed)
        left, right, changed = self._change(target, time, age)
        record = self._record(
            kind, source, target, time, age, changed, commit_revision, extra, counters
        )
        return dict(event=deepcopy(record), left=left, right=right)

    def _state_change(self, root, commit_revision, counters):
        if self._near_calendar(root):
            raise StateEventError(
                "ambiguous_event_order",
                "State root and calendar deadline are not distinguishable",
                root=root,
            )
        guard = self._program._by_id[root["guard_id"]]
        source = self._mode
        target = guard["target_mode"]
        if guard["source_mode"] != source:
            raise NetworkError("Inactive state transition selected")
        self._budget_event(counters, True)
        age = root["bracket_s"][0] - self._entry_time
        left, right, changed = self._change(target, root["time_s"], age)
        extra = {key: value for key, value in root.items() if key != "time_s"}
        record = self._record(
            "state", source, target, root["time_s"], age, changed, commit_revision, extra, counters
        )
        return dict(event=deepcopy(record), left=left, right=right)

    def _run(self, target, commit_revision, counters, stop):
        segments = []
        events = []
        while self._engine._time < target:
            deadline = self._schedule[self._cursor] if self._cursor < len(self._schedule) else None
            bound = min(target, deadline["time_s"]) if deadline else target
            try:
                root, part = _advance_until(
                    self._engine,
                    bound,
                    list(self._program._modes[self._mode]["transitions"]),
                    counters,
                    self.budget,
                )
            except StateEventError as error:
                if (
                    deadline
                    and bound == deadline["time_s"]
                    and error.report["criterion"] == "endpoint_contact"
                ):
                    raise StateEventError(
                        "ambiguous_event_order",
                        "Calendar endpoint meets unresolved state root",
                        calendar=deadline,
                        cause=error.report,
                    ) from error
                raise
            segments.extend(part)
            if root is not None:
                event = self._state_change(root, commit_revision, counters)
                events.append(event)
                if stop:
                    break
                continue
            if self._engine._time != bound:
                raise NetworkError("Supervisory interval did not reach deadline")
            if deadline and bound == deadline["time_s"]:
                event = self._force(
                    "calendar",
                    deadline["to"],
                    commit_revision,
                    dict(calendar_id=deadline["id"], calendar_index=self._cursor),
                    counters,
                )
                self._cursor += 1
                events.append(event)
                if stop and event["event"]["changed"]:
                    break
        return segments, events

    def _snapshot(self):
        return dict(
            time_s=self._engine._time,
            revision=self._revision,
            mode=self._mode,
            mode_entry_time_s=self._entry_time,
            mode_entry_temperatures_k=deepcopy(self._entry_temperatures),
            calendar_cursor=self._cursor,
            event_count=len(self._history),
            mode_transition_count=sum(r["changed"] for r in self._history),
            online_command_count=sum(r["kind"] == "online" for r in self._history),
            history=deepcopy(self._history),
            active_transitions=[g["id"] for g in self._program._modes[self._mode]["transitions"]],
            active_commands_sha256=digest(self._program._modes[self._mode]["commands"]),
            engine=self._engine.snapshot(),
        )

    def snapshot(self):
        return self._snapshot()

    def _install(self, candidate):
        (
            self._engine,
            self._mode,
            self._entry_time,
            self._entry_temperatures,
            self._history,
            self._cursor,
            self._revision,
        ) = (
            candidate._engine,
            candidate._mode,
            candidate._entry_time,
            candidate._entry_temperatures,
            candidate._history,
            candidate._cursor,
            candidate._revision,
        )

    def _transaction(self, duration, command=None, stop=False):
        self._sources()
        duration = number(duration, "duration_s", positive=True)
        target = self._engine._time + duration
        if (
            duration > self.budget["maximum_duration_s"]
            or not math.isfinite(target)
            or target <= self._engine._time
        ):
            raise NetworkError("Invalid supervisory duration or clock")
        candidate = deepcopy(self)
        commit = self.revision + 1
        counters = dict(rhs=0, root=0, attempts=0, rejected=0, events=0, switches=0)
        prefix = []
        if command is not None:
            mode, cid = command
            prefix.append(
                candidate._force(
                    "online",
                    mode,
                    commit,
                    dict(command_id=cid, observed_revision=self.revision),
                    counters,
                )
            )
        segments, events = candidate._run(target, commit, counters, stop)
        events = prefix + events
        candidate._revision = commit
        result = candidate._snapshot()
        if (
            abs(result["engine"]["state"]["energy_residual_j"])
            > self.budget["maximum_energy_error_j"]
        ):
            raise StateEventError("energy", "Supervised energy budget exceeded")
        self._sources()
        self._install(candidate)
        return dict(
            **result,
            segments=segments,
            events=events,
            evaluations=counters,
            requested_time_s=target,
            stopped_at_transition=stop and any(e["event"]["changed"] for e in events),
        )

    def advance(self, duration_s, *, stop_at_transition=False):
        if type(stop_at_transition) is not bool:
            raise NetworkError("stop_at_transition must be boolean")
        return self._transaction(duration_s, stop=stop_at_transition)

    def step(self, target_mode, duration_s, *, expected_revision, command_id):
        self._sources()
        if type(expected_revision) is not int or expected_revision != self.revision:
            raise StateEventError(
                "stale_revision", "Online command used stale observation revision"
            )
        if not isinstance(target_mode, str) or target_mode not in self._online_modes:
            raise StateEventError("online_mode", "Target mode is not enabled for online commands")
        cid = self._command_id(command_id)
        if any(r["kind"] == "online" and r["command_id"] == cid for r in self._history):
            raise StateEventError("duplicate_command", "Online command ID was already accepted")
        return self._transaction(duration_s, command=(target_mode, cid))

    def checkpoint(self):
        payload = dict(
            schema_version=1,
            model="supervised-radau-v1",
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
            raise NetworkError("Supervised checkpoint digest mismatch")
        _keys(
            p,
            ("schema_version", "model", "input_sha256", "signature", "state", "native_checkpoint"),
        )
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["model"] != "supervised-radau-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
        ):
            raise NetworkError("Supervised checkpoint input/model/source mismatch")
        state = p["state"]
        _keys(
            state,
            (
                "time_s",
                "revision",
                "mode",
                "mode_entry_time_s",
                "mode_entry_temperatures_k",
                "calendar_cursor",
                "event_count",
                "mode_transition_count",
                "online_command_count",
                "history",
                "active_transitions",
                "active_commands_sha256",
                "engine",
            ),
        )
        time = number(state["time_s"], "time_s")
        revision = state["revision"]
        history = state["history"]
        if (
            time < 0
            or type(revision) is not int
            or revision < 0
            or not isinstance(history, list)
            or len(history) > self.budget["maximum_history_entries"]
        ):
            raise NetworkError("Supervised checkpoint revision/history malformed")
        if (revision == 0) != (time == 0) or (revision == 0 and history):
            raise NetworkError("Supervised checkpoint revision clock mismatch")
        mode = self._program._initial_mode
        entry_time = 0.0
        previous = 0.0
        cursor = 0
        commands = set()
        changes = 0
        last_commit = 0
        last_changed = None
        entry_temperatures = {
            name: float(self._program._frame._origin[i])
            for i, name in enumerate(self._program._frame._names)
        }
        frame_hash = digest(entry_temperatures)
        common = (
            "kind",
            "time_s",
            "sequence",
            "source_mode",
            "target_mode",
            "changed",
            "commit_revision",
            "residence_lower_bound_s",
            "source_commands_sha256",
            "target_commands_sha256",
            "entry_frame_sha256",
        )
        for sequence, record in enumerate(history, 1):
            kind = record.get("kind") if isinstance(record, dict) else None
            extra = {
                "state": (
                    "guard_id",
                    "bracket_s",
                    "bracket_values_k",
                    "root_value_k",
                    "rate_bounds_k_s",
                    "threshold_k",
                    "direction",
                ),
                "calendar": ("calendar_id", "calendar_index"),
                "online": ("command_id", "observed_revision"),
            }.get(kind)
            if extra is None:
                raise NetworkError("Unknown supervised checkpoint event kind")
            _keys(record, common + extra)
            when = number(record["time_s"], "event_time_s")
            target = record["target_mode"]
            commit = record["commit_revision"]
            changed = record["changed"]
            if (
                type(record["sequence"]) is not int
                or record["sequence"] != sequence
                or record["source_mode"] != mode
                or not isinstance(target, str)
                or target not in self._program._modes
                or type(changed) is not bool
                or changed != (target != mode)
                or when < previous
                or when > time
                or type(commit) is not int
                or not last_commit <= commit <= revision
                or commit <= 0
            ):
                raise NetworkError("Supervised checkpoint path/revision mismatch")
            age = when - entry_time
            if kind == "state":
                tid = record["guard_id"]
                if not isinstance(tid, str) or tid not in self._program._by_id:
                    raise NetworkError("Unknown checkpoint state guard")
                guard = self._program._by_id[tid]
                if guard["source_mode"] != mode or guard["target_mode"] != target or not changed:
                    raise NetworkError("Supervised state-transition path mismatch")

                def pair(key):
                    values = record[key]
                    if not isinstance(values, list) or len(values) != 2:
                        raise NetworkError("Malformed supervisory root pair")
                    return [number(v, key) for v in values]

                lo, hi = pair("bracket_s")
                gl, gr = pair("bracket_values_k")
                rl, rh = pair("rate_bounds_k_s")
                value = number(record["root_value_k"], "root_value_k")
                direction = guard["direction"]
                age = lo - entry_time
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
                    or self._near_calendar(record)
                ):
                    raise NetworkError("Supervised root/collision certificate mismatch")
            elif kind == "calendar":
                if cursor >= len(self._schedule):
                    raise NetworkError("Unexpected checkpoint calendar event")
                declared = self._schedule[cursor]
                if (
                    type(record["calendar_index"]) is not int
                    or record["calendar_index"] != cursor
                    or record["calendar_id"] != declared["id"]
                    or when != declared["time_s"]
                    or target != declared["to"]
                ):
                    raise NetworkError("Supervised calendar prefix mismatch")
                cursor += 1
            else:
                cid = self._command_id(record["command_id"])
                observed = record["observed_revision"]
                if (
                    cid in commands
                    or target not in self._online_modes
                    or type(observed) is not int
                    or observed != commit - 1
                    or commit <= last_commit
                ):
                    raise NetworkError("Supervised online receipt/revision mismatch")
                commands.add(cid)
            if number(record["residence_lower_bound_s"], "residence_lower_bound_s") != age or (
                changed and age < self._program.minimum_residence
            ):
                raise NetworkError("Supervised residence certificate mismatch")
            if record["source_commands_sha256"] != digest(
                self._program._modes[mode]["commands"]
            ) or record["target_commands_sha256"] != digest(
                self._program._modes[target]["commands"]
            ):
                raise NetworkError("Supervised actuator mask/configuration mismatch")
            if changed:
                entry_time = when
                frame_hash = record["entry_frame_sha256"]
                last_changed = record
                changes += 1
            elif record["entry_frame_sha256"] != frame_hash:
                raise NetworkError("No-op command reset checkpoint entry frame")
            mode = target
            previous = when
            last_commit = commit
        if (
            type(state["calendar_cursor"]) is not int
            or state["calendar_cursor"] != cursor
            or cursor != bisect.bisect_right(self._times, time)
            or type(state["event_count"]) is not int
            or state["event_count"] != len(history)
            or type(state["mode_transition_count"]) is not int
            or state["mode_transition_count"] != changes
            or type(state["online_command_count"]) is not int
            or state["online_command_count"] != len(commands)
            or state["mode"] != mode
            or number(state["mode_entry_time_s"], "mode_entry_time_s") != entry_time
        ):
            raise NetworkError("Supervised checkpoint counters/mode mismatch")
        temperatures = state["mode_entry_temperatures_k"]
        if not isinstance(temperatures, dict) or set(temperatures) != set(entry_temperatures):
            raise NetworkError("Supervised checkpoint entry map mismatch")
        temperatures = {
            name: number(temperatures[name], "entry_temperature_k", positive=True)
            for name in entry_temperatures
        }
        if digest(temperatures) != frame_hash:
            raise NetworkError("Supervised checkpoint entry frame hash mismatch")
        if last_changed and last_changed["kind"] == "state":
            guard = self._program._by_id[last_changed["guard_id"]]
            if abs(
                temperatures[guard["node"]] - guard["threshold_k"] - last_changed["root_value_k"]
            ) > 4 * math.ulp(guard["threshold_k"]):
                raise NetworkError("Supervised checkpoint entry/root mismatch")
        candidate = deepcopy(self)
        active = candidate._program._frame._parser._patch(
            candidate._program._base, candidate._program._modes[mode]["commands"]
        )
        candidate._program._entry_check(mode, temperatures)
        engine = candidate._program._frame._new_engine(active, temperatures)
        engine.restore(deepcopy(p["native_checkpoint"]))
        (
            candidate._engine,
            candidate._mode,
            candidate._entry_time,
            candidate._entry_temperatures,
            candidate._history,
            candidate._cursor,
            candidate._revision,
        ) = engine, mode, entry_time, temperatures, deepcopy(history), cursor, revision
        for guard in candidate._program._modes[mode]["transitions"]:
            if guard["direction"] * (engine._increments[guard["index"]] - guard["offset_k"]) >= 0:
                raise NetworkError("Checkpoint skipped an active state guard")
        fresh = candidate._snapshot()
        if (
            fresh != state
            or abs(fresh["engine"]["state"]["energy_residual_j"])
            > self.budget["maximum_energy_error_j"]
        ):
            raise NetworkError("Supervised checkpoint observation/energy mismatch")
        self._sources()
        self._install(candidate)
        return fresh
