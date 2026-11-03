"""Atomic sampled controller/plant episodes with source-bound RNG checkpoints.

The policy is an explicit two-mode sampled Schmitt relay, not an arbitrary AI
agent. One local Random.random draw is consumed per observation, even at zero
noise. No process-global RNG or external controller process is modified.
"""

import json
import math
import platform
import random
from copy import deepcopy
from pathlib import Path

from core.network import NetworkError, identifier, number
from scripts.nonlinear_thermal_graph import digest
from scripts.solve_hydraulic_graph import _keys, sha
from scripts.state_event_radau import StateEventError
from scripts.supervised_radau import SupervisedRadauGraph
from scripts.supervised_radau import signature as plant_signature

BUDGET = Path("validation/ai05_closed_loop_budget_v1.json")


def signature():
    value = plant_signature()
    value["drivers"]["scripts/closed_loop_radau.py"] = sha(__file__)
    value["budgets"][str(BUDGET)] = sha(BUDGET)
    value["controller_runtime"] = dict(
        implementation=platform.python_implementation(),
        python=platform.python_version(),
        rng="python-random-MT19937-state-v3-uniform-one-draw",
    )
    return value


def _rng_state(rng):
    return json.loads(json.dumps(rng.getstate()))


class ClosedLoopRadau:
    def __init__(self, request, *, max_macro_step_s=2.0):
        _keys(request, ("format", "schema_version", "plant", "policy"))
        if (
            request["format"] != "atomic-closed-loop-radau"
            or type(request["schema_version"]) is not int
            or request["schema_version"] != 1
        ):
            raise NetworkError("Expected closed-loop schema1")
        self.budget = json.loads(BUDGET.read_text())
        self._request = deepcopy(request)
        policy = request["policy"]
        _keys(
            policy,
            (
                "kind",
                "sensor_node",
                "low_k",
                "high_k",
                "heat_mode",
                "cool_mode",
                "initial_decision",
                "sample_period_s",
                "noise_amplitude_k",
                "seed",
            ),
        )
        if policy["kind"] != "sampled_schmitt_v1":
            raise NetworkError("Unsupported controller policy")
        self._node = identifier(policy["sensor_node"])
        self._heat, self._cool = identifier(policy["heat_mode"]), identifier(policy["cool_mode"])
        if self._heat == self._cool or policy["initial_decision"] not in (self._heat, self._cool):
            raise NetworkError("Policy requires two distinct modes and declared initial memory")
        self._low = number(policy["low_k"], "low_k", positive=True)
        self._high = number(policy["high_k"], "high_k", positive=True)
        self._period = number(policy["sample_period_s"], "sample_period_s", positive=True)
        self._noise = number(policy["noise_amplitude_k"], "noise_amplitude_k")
        self._seed = policy["seed"]
        if (
            self._low >= self._high
            or self._period < self.budget["minimum_sample_period_s"]
            or not 0 <= self._noise <= self.budget["maximum_noise_amplitude_k"]
            or type(self._seed) is not int
            or not 0 <= self._seed < 2**64
        ):
            raise NetworkError("Invalid relay band, sample period, noise or seed")
        self._plant = SupervisedRadauGraph(
            deepcopy(request["plant"]), max_macro_step_s=max_macro_step_s
        )
        state = self._plant.snapshot()
        if self._node not in state["engine"]["state"]["temperatures_k"]:
            raise NetworkError("Unknown sensor node")
        if not {self._heat, self._cool} <= self._plant._online_modes:
            raise NetworkError("Both policy modes must be enabled online")
        self._initial_plant_sha256 = digest(state)
        self._decision = policy["initial_decision"]
        self._rng = random.Random(self._seed)
        self._ticks, self._revision = 0, 0
        self._history = []
        self.identity = digest(dict(request=request, max_macro_step_s=self._plant.max_step))
        self.captured = signature()

    @property
    def revision(self):
        return self._revision

    def _sources(self):
        if signature() != self.captured:
            raise StateEventError("source_drift", "Closed-loop sources, budgets or runtime changed")

    def _choose(self, measured, previous):
        if measured <= self._low:
            return self._heat
        if measured >= self._high:
            return self._cool
        return previous

    def snapshot(self):
        return dict(
            tick_count=self._ticks,
            revision=self._revision,
            decision=self._decision,
            next_sample_time_s=self._ticks * self._period,
            rng_sha256=digest(_rng_state(self._rng)),
            history=deepcopy(self._history),
            plant=self._plant.snapshot(),
        )

    def _tick(self, commit_revision):
        before = self._plant.snapshot()
        when = self._ticks * self._period
        end = (self._ticks + 1) * self._period
        if before["time_s"] != when or not math.isfinite(end) or end <= when:
            raise NetworkError("Controller tick clock cannot progress exactly")
        truth = before["engine"]["state"]["temperatures_k"][self._node]
        unit = self._rng.random()
        measured = truth + self._noise * (2 * unit - 1)
        target = self._choose(measured, self._decision)
        cid = f"controller:tick-{self._ticks + 1}"
        result = self._plant.step(
            target, end - when, expected_revision=before["revision"], command_id=cid
        )
        for event in result["events"]:
            record = event["event"]
            if record["kind"] == "state":
                lo, hi = record["bracket_s"]
                gap = self.budget["minimum_sample_root_gap_s"]
                if any(lo - gap <= boundary <= hi + gap for boundary in (when, end)):
                    raise StateEventError(
                        "ambiguous_sample_order",
                        "State root and sampling clock are not distinguishable",
                        root=record,
                    )
        after = self._plant.snapshot()
        if after["time_s"] != end:
            raise NetworkError("Controller tick missed endpoint")
        self._history.append(
            dict(
                sequence=self._ticks + 1,
                commit_revision=commit_revision,
                time_s=when,
                end_time_s=end,
                previous_decision=self._decision,
                decision=target,
                true_temperature_k=truth,
                uniform_draw=unit,
                measured_temperature_k=measured,
                command_id=cid,
                observed_plant_revision=before["revision"],
                plant_before_sha256=digest(before),
                plant_after_sha256=digest(after),
            )
        )
        self._decision = target
        self._ticks += 1
        return dict(
            **result["evaluations"],
            maximum_trial_internal_step_s=max(
                segment["maximum_trial_internal_step_s"] for segment in result["segments"]
            ),
        )

    def run(self, ticks, *, expected_revision):
        self._sources()
        if type(expected_revision) is not int or expected_revision != self._revision:
            raise StateEventError("stale_revision", "Closed-loop observation revision is stale")
        if (
            type(ticks) is not int
            or not 1 <= ticks <= self.budget["maximum_ticks_per_transaction"]
            or self._ticks + ticks > self.budget["maximum_episode_ticks"]
        ):
            raise NetworkError("Closed-loop tick budget exceeded")
        candidate = deepcopy(self)
        counters = dict(rhs=0, root=0, attempts=0, events=0, switches=0)
        limits = dict(
            rhs="maximum_batch_rhs_evaluations",
            root="maximum_batch_root_evaluations",
            attempts="maximum_batch_attempts",
            events="maximum_batch_events",
            switches="maximum_batch_switches",
        )
        maximum_step = 0.0
        for _ in range(ticks):
            used = candidate._tick(self._revision + 1)
            maximum_step = max(maximum_step, used["maximum_trial_internal_step_s"])
            for key in counters:
                counters[key] += used[key]
                if counters[key] > self.budget[limits[key]]:
                    raise StateEventError(
                        "controller_work_budget",
                        "Closed-loop shared work budget exhausted",
                        resource=key,
                        used=counters[key],
                    )
        candidate._revision += 1
        result = candidate.snapshot()
        self._sources()
        self._install(candidate)
        return dict(**result, work=dict(**counters, maximum_trial_internal_step_s=maximum_step))

    def step(self, *, expected_revision):
        return self.run(1, expected_revision=expected_revision)

    def _install(self, candidate):
        self._plant, self._decision, self._rng, self._ticks, self._revision, self._history = (
            candidate._plant,
            candidate._decision,
            candidate._rng,
            candidate._ticks,
            candidate._revision,
            candidate._history,
        )

    def checkpoint(self):
        self._sources()
        payload = dict(
            schema_version=1,
            model="closed-loop-radau-v1",
            input_sha256=self.identity,
            signature=deepcopy(self.captured),
            state=self.snapshot(),
            rng_state=_rng_state(self._rng),
            plant_checkpoint=self._plant.checkpoint(),
        )
        return dict(payload=payload, sha256=digest(payload))

    def restore(self, checkpoint):
        self._sources()
        _keys(checkpoint, ("payload", "sha256"))
        p = checkpoint["payload"]
        if digest(p) != checkpoint["sha256"]:
            raise NetworkError("Closed-loop checkpoint digest mismatch")
        _keys(
            p,
            (
                "schema_version",
                "model",
                "input_sha256",
                "signature",
                "state",
                "rng_state",
                "plant_checkpoint",
            ),
        )
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["model"] != "closed-loop-radau-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
        ):
            raise NetworkError("Closed-loop input/source/runtime mismatch")
        state = p["state"]
        _keys(
            state,
            (
                "tick_count",
                "revision",
                "decision",
                "next_sample_time_s",
                "rng_sha256",
                "history",
                "plant",
            ),
        )
        ticks, revision, history = state["tick_count"], state["revision"], state["history"]
        if (
            type(ticks) is not int
            or not 0 <= ticks <= self.budget["maximum_episode_ticks"]
            or type(revision) is not int
            or not 0 <= revision <= ticks
            or (revision == 0) != (ticks == 0)
            or not isinstance(history, list)
            or len(history) != ticks
        ):
            raise NetworkError("Closed-loop tick/revision/history mismatch")
        candidate = deepcopy(self)
        candidate._plant.restore(deepcopy(p["plant_checkpoint"]))
        for event in candidate._plant.snapshot()["history"]:
            if event["kind"] == "state":
                nearest = round(event["time_s"] / self._period)
                boundary = nearest * self._period
                lo, hi = event["bracket_s"]
                gap = self.budget["minimum_sample_root_gap_s"]
                if 0 <= nearest <= ticks and lo - gap <= boundary <= hi + gap:
                    raise NetworkError("Checkpoint has ambiguous sample/root ordering")
        rng = random.Random(self._seed)
        decision = self._request["policy"]["initial_decision"]
        previous_hash = self._initial_plant_sha256
        online = [r for r in candidate._plant.snapshot()["history"] if r["kind"] == "online"]
        if len(online) != ticks:
            raise NetworkError("Policy/plant receipt counts differ")
        keys = (
            "sequence",
            "commit_revision",
            "time_s",
            "end_time_s",
            "previous_decision",
            "decision",
            "true_temperature_k",
            "uniform_draw",
            "measured_temperature_k",
            "command_id",
            "observed_plant_revision",
            "plant_before_sha256",
            "plant_after_sha256",
        )
        last_commit, batch_ticks = 0, 0
        for i, (record, receipt) in enumerate(zip(history, online)):
            _keys(record, keys)
            commit = record["commit_revision"]
            if (
                type(commit) is not int
                or commit not in (last_commit, last_commit + 1)
                or commit <= 0
            ):
                raise NetworkError("Closed-loop transaction revision path mismatch")
            batch_ticks = batch_ticks + 1 if commit == last_commit else 1
            if batch_ticks > self.budget["maximum_ticks_per_transaction"]:
                raise NetworkError("Checkpoint transaction tick budget exceeded")
            last_commit = commit
            truth = number(record["true_temperature_k"], "true_temperature_k", positive=True)
            unit = rng.random()
            measured = truth + self._noise * (2 * unit - 1)
            target = self._choose(measured, decision)
            cid = f"controller:tick-{i + 1}"
            if (
                type(record["sequence"]) is not int
                or record["sequence"] != i + 1
                or number(record["time_s"], "time_s") != i * self._period
                or number(record["end_time_s"], "end_time_s") != (i + 1) * self._period
                or record["previous_decision"] != decision
                or record["decision"] != target
                or number(record["uniform_draw"], "uniform_draw") != unit
                or number(record["measured_temperature_k"], "measured_temperature_k") != measured
                or record["command_id"] != cid
                or type(record["observed_plant_revision"]) is not int
                or record["observed_plant_revision"] != i
                or record["plant_before_sha256"] != previous_hash
                or receipt["command_id"] != cid
                or receipt["time_s"] != i * self._period
                or receipt["observed_revision"] != i
                or receipt["target_mode"] != target
            ):
                raise NetworkError("Closed-loop observation/memory/RNG/receipt mismatch")
            after_hash = record["plant_after_sha256"]
            if (
                not isinstance(after_hash, str)
                or len(after_hash) != 64
                or any(c not in "0123456789abcdef" for c in after_hash)
            ):
                raise NetworkError("Malformed plant observation hash")
            previous_hash, decision = after_hash, target
        plant = candidate._plant.snapshot()
        if (
            last_commit != revision
            or p["rng_state"] != _rng_state(rng)
            or state["rng_sha256"] != digest(_rng_state(rng))
            or state["decision"] != decision
            or plant != state["plant"]
            or plant["revision"] != ticks
            or number(state["next_sample_time_s"], "next_sample_time_s") != ticks * self._period
            or plant["time_s"] != ticks * self._period
            or digest(plant) != previous_hash
        ):
            raise NetworkError("Closed-loop final clock, RNG or plant mismatch")
        candidate._rng, candidate._decision = rng, decision
        candidate._ticks, candidate._revision, candidate._history = (
            ticks,
            revision,
            deepcopy(history),
        )
        fresh = candidate.snapshot()
        if fresh != state:
            raise NetworkError("Closed-loop checkpoint observation mismatch")
        self._sources()
        self._install(candidate)
        return fresh
