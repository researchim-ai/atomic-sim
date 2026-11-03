"""Atomic sampled PI episodes over a finite-inventory thermal EOS assembly.

The sensor filter is a sampled EWMA; power is held between observations.
Conditional integration prevents accumulation into amplitude/slew saturation.
No continuous-controller, asynchronous external-agent or hybrid-root claim.
"""

import json
import math
import re
import threading
from copy import deepcopy

from atomic_model_runtime.resources import resource
from core.network import NetworkError, number
from scripts.controlled_eos_assembly import ControlledEOSAssembly, _locked
from scripts.controlled_eos_assembly import signature as plant_signature
from scripts.inertial_graph import InertialGraphError, digest
from scripts.solve_hydraulic_graph import _keys, sha

BUDGET = resource("validation/ai05_closed_loop_pi_budget_v1.json")
MEMORY = ("filtered_k", "integral_w", "output_w")
DECISION = (
    "filtered_k",
    "error_k",
    "integral_before_w",
    "integral_after_w",
    "output_before_w",
    "output_w",
    "unconstrained_w",
    "lower_w",
    "upper_w",
    "integral_held",
)


def signature():
    result = plant_signature()
    result["drivers"]["scripts/closed_loop_eos_pi.py"] = sha(__file__)
    result["budgets"][str(BUDGET)] = sha(BUDGET)
    return result


def pi_sample(policy, memory, truth):
    """Pure discrete control law; inputs are validated by the episode wrapper."""
    dt = policy["sample_period_s"]
    tau = policy["filter_tau_s"]
    alpha = 1.0 if tau == 0 else -math.expm1(-dt / tau)
    filtered = truth if tau == 0 else memory["filtered_k"] + alpha * (truth - memory["filtered_k"])
    error = policy["setpoint_k"] - filtered
    old_i = memory["integral_w"]
    old_u = memory["output_w"]
    candidate_i = old_i + policy["ki_w_k_s"] * dt * error
    lo = max(policy["output_min_w"], old_u - policy["slew_w_s"] * dt)
    hi = min(policy["output_max_w"], old_u + policy["slew_w_s"] * dt)
    raw = policy["bias_w"] + policy["kp_w_k"] * error + candidate_i
    if not all(math.isfinite(v) for v in [filtered, error, candidate_i, lo, hi, raw]):
        raise NetworkError("Nonfinite sampled PI arithmetic")
    limited = min(hi, max(lo, raw))
    held = ((raw > limited and error > 0) or (raw < limited and error < 0)) and candidate_i != old_i
    integral = old_i if held else candidate_i
    raw = policy["bias_w"] + policy["kp_w_k"] * error + integral
    if not math.isfinite(raw):
        raise NetworkError("Nonfinite PI output")
    output = min(hi, max(lo, raw))
    decision = dict(
        filtered_k=filtered,
        error_k=error,
        integral_before_w=old_i,
        integral_after_w=integral,
        output_before_w=old_u,
        output_w=output,
        unconstrained_w=raw,
        lower_w=lo,
        upper_w=hi,
        integral_held=held,
    )
    return dict(filtered_k=filtered, integral_w=integral, output_w=output), decision


class ClosedLoopEOSPI:
    def __init__(self, request, *, max_step_s=0.0005):
        _keys(request, ("format", "schema_version", "plant", "policy"))
        if (
            request["format"] != "atomic-closed-loop-eos-pi"
            or type(request["schema_version"]) is not int
            or request["schema_version"] != 1
        ):
            raise NetworkError("Expected closed-loop EOS PI schema1")
        self.budget = json.loads(BUDGET.read_text())
        self.request = deepcopy(request)
        p = request["policy"]
        _keys(
            p,
            (
                "kind",
                "sensor",
                "actuator",
                "setpoint_k",
                "sample_period_s",
                "filter_tau_s",
                "kp_w_k",
                "ki_w_k_s",
                "bias_w",
                "output_min_w",
                "output_max_w",
                "slew_w_s",
                "initial_integral_w",
                "initial_output_w",
            ),
        )
        if (
            p["kind"] != "sampled_pi_conditional_v1"
            or not isinstance(p["sensor"], str)
            or not isinstance(p["actuator"], str)
        ):
            raise NetworkError("Unsupported PI policy/ports")
        self._policy = {
            k: (v if k in ["kind", "sensor", "actuator"] else number(v, k)) for k, v in p.items()
        }
        p = self._policy
        if (
            not self.budget["minimum_sample_period_s"]
            <= p["sample_period_s"]
            <= self.budget["maximum_sample_period_s"]
            or not 0 <= p["filter_tau_s"] <= self.budget["maximum_filter_tau_s"]
            or p["kp_w_k"] < 0
            or p["ki_w_k_s"] < 0
            or p["slew_w_s"] <= 0
            or p["output_min_w"] >= p["output_max_w"]
            or max(abs(p["output_min_w"]), abs(p["output_max_w"]))
            > self.budget["maximum_absolute_output_w"]
            or not p["output_min_w"] <= p["initial_output_w"] <= p["output_max_w"]
        ):
            raise NetworkError("Invalid sampled PI domain, gains or limits")
        self._plant = ControlledEOSAssembly(deepcopy(request["plant"]), max_step_s=max_step_s)
        initial = self._plant.snapshot()
        if (
            p["sensor"] not in initial["port_temperatures_k"]
            or p["actuator"] not in initial["active_heat_power_w"]
        ):
            raise NetworkError("Unknown sensor or actuator")
        domains = [e.channel.system.domain for e in self._plant.circuits]
        if not p["sensor"].startswith("wall."):
            domains = [domains[["primary", "secondary"].index(p["sensor"].split(".")[0])]]
        self.sensor_min = max(d["min_temperature_k"] for d in domains)
        self.sensor_max = min(d["max_temperature_k"] for d in domains)
        if (
            not self.sensor_min <= p["setpoint_k"] <= self.sensor_max
            or p["initial_output_w"] != initial["active_heat_power_w"][p["actuator"]]
        ):
            raise NetworkError("PI target or initial actuator inconsistent with plant")
        self.initial_truth = initial["port_temperatures_k"][p["sensor"]]
        self.initial_memory = dict(
            filtered_k=self.initial_truth,
            integral_w=p["initial_integral_w"],
            output_w=p["initial_output_w"],
        )
        self.initial_controls = deepcopy(initial["active_heat_power_w"])
        self.initial_plant_sha256 = digest(initial)
        self._memory = deepcopy(self.initial_memory)
        self._ticks = self._revision = 0
        self._history = []
        self.identity = digest(dict(request=request, max_step_s=self._plant.max_step))
        self.captured = signature()
        self._lock = threading.RLock()

    def __deepcopy__(self, memo):
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key != "_lock":
                setattr(result, key, deepcopy(value, memo))
        result._lock = threading.RLock()
        return result

    @property
    def revision(self):
        return self._revision

    def _sources(self):
        if signature() != self.captured:
            raise InertialGraphError(
                "source_drift", "Closed-loop PI sources/budget/runtime changed"
            )

    @_locked
    def snapshot(self):
        return dict(
            tick_count=self._ticks,
            revision=self.revision,
            next_sample_time_s=self._ticks * self._policy["sample_period_s"],
            controller=deepcopy(self._memory),
            history=deepcopy(self._history),
            plant=self._plant.snapshot(),
        )

    def _tick(self, commit_revision):
        p = self._policy
        before = self._plant.snapshot()
        when = self._ticks * p["sample_period_s"]
        end = (self._ticks + 1) * p["sample_period_s"]
        if before["time_s"] != when or not math.isfinite(end) or end <= when:
            raise NetworkError("PI sampling clock cannot progress exactly")
        truth = before["port_temperatures_k"][p["sensor"]]
        memory, decision = pi_sample(p, self._memory, truth)
        powers = before["active_heat_power_w"].copy()
        powers[p["actuator"]] = memory["output_w"]
        result = self._plant.step(
            powers,
            end - when,
            expected_revision=before["revision"],
            command_id=f"pi.tick-{self._ticks + 1:06d}",
        )
        after = self._plant.snapshot()
        if after["time_s"] != end:
            raise NetworkError("PI plant/sample clock mismatch")
        self._history.append(
            dict(
                sequence=self._ticks + 1,
                sample_time_s=when,
                commit_revision=commit_revision,
                truth_k=truth,
                **decision,
                plant_before_sha256=digest(before),
                plant_after_sha256=digest(after),
            )
        )
        self._memory = memory
        self._ticks += 1
        return result["work"]

    @_locked
    def run(self, ticks, *, expected_revision):
        self._sources()
        if type(expected_revision) is not int or expected_revision != self.revision:
            raise InertialGraphError("stale_revision", "Closed-loop PI observation is stale")
        if (
            type(ticks) is not int
            or not 1 <= ticks <= self.budget["maximum_ticks_per_transaction"]
            or self._ticks + ticks > self.budget["maximum_episode_ticks"]
        ):
            raise NetworkError("PI tick budget exceeded")
        c = deepcopy(self)
        rhs = substeps = 0
        maxstep = stability = 0.0
        for _ in range(ticks):
            remaining_rhs = self.budget["maximum_batch_rhs_evaluations"] - rhs
            remaining_steps = self.budget["maximum_batch_substeps"] - substeps
            if remaining_rhs <= 0 or remaining_steps <= 0:
                raise InertialGraphError("controller_work_budget", "PI batch work budget exhausted")
            # Pass the remaining integration allowance into the nested transaction.
            # Only tighten limits; restore the original declared budget afterwards.
            declared = c._plant.budget
            c._plant.budget = dict(
                declared,
                maximum_rhs_evaluations=min(declared["maximum_rhs_evaluations"], remaining_rhs),
                maximum_substeps=min(declared["maximum_substeps"], remaining_steps),
            )
            try:
                work = c._tick(self.revision + 1)
            finally:
                c._plant.budget = declared
            rhs += work["rhs"]
            substeps += work["substeps"]
            maxstep = max(maxstep, work["maximum_internal_step_s"])
            stability = max(stability, work["maximum_stability_fraction"])
            if (
                rhs > self.budget["maximum_batch_rhs_evaluations"]
                or substeps > self.budget["maximum_batch_substeps"]
            ):
                raise InertialGraphError("controller_work_budget", "PI batch work budget exceeded")
        c._revision += 1
        result = c.snapshot()
        self._sources()
        self._install(c)
        return dict(
            **result,
            work=dict(
                rhs=rhs,
                substeps=substeps,
                maximum_internal_step_s=maxstep,
                maximum_stability_fraction=stability,
            ),
        )

    def step(self, *, expected_revision):
        return self.run(1, expected_revision=expected_revision)

    def _install(self, c):
        self._plant, self._memory, self._ticks, self._revision, self._history = (
            c._plant,
            c._memory,
            c._ticks,
            c._revision,
            c._history,
        )

    @_locked
    def checkpoint(self):
        self._sources()
        p = dict(
            schema_version=1,
            model="closed-loop-eos-pi-v1",
            input_sha256=self.identity,
            signature=deepcopy(self.captured),
            state=self.snapshot(),
            plant_checkpoint=self._plant.checkpoint(),
        )
        return dict(payload=p, sha256=digest(p))

    @_locked
    def restore(self, checkpoint):
        self._sources()
        _keys(checkpoint, ("payload", "sha256"))
        p = checkpoint["payload"]
        if digest(p) != checkpoint["sha256"]:
            raise NetworkError("PI checkpoint digest mismatch")
        _keys(
            p, ("schema_version", "model", "input_sha256", "signature", "state", "plant_checkpoint")
        )
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["model"] != "closed-loop-eos-pi-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
        ):
            raise NetworkError("PI checkpoint identity mismatch")
        state = p["state"]
        _keys(
            state,
            ("tick_count", "revision", "next_sample_time_s", "controller", "history", "plant"),
        )
        ticks, rev, history = state["tick_count"], state["revision"], state["history"]
        if (
            type(ticks) is not int
            or type(rev) is not int
            or not 0 <= ticks <= self.budget["maximum_episode_ticks"]
            or not 0 <= rev <= ticks
            or (rev == 0) != (ticks == 0)
            or not isinstance(history, list)
            or len(history) != ticks
        ):
            raise NetworkError("PI checkpoint clock/history mismatch")
        c = deepcopy(self)
        c._plant.restore(p["plant_checkpoint"])
        plant = c._plant.snapshot()
        period = self._policy["sample_period_s"]
        if (
            plant["time_s"] != ticks * period
            or plant["revision"] != ticks
            or len(plant["history"]) != ticks
        ):
            raise NetworkError("PI checkpoint plant clock mismatch")
        memory = deepcopy(self.initial_memory)
        previous_sha = self.initial_plant_sha256
        last_commit = 0
        batch = 0
        for seq, (r, receipt) in enumerate(zip(history, plant["history"]), 1):
            _keys(
                r,
                (
                    "sequence",
                    "sample_time_s",
                    "commit_revision",
                    "truth_k",
                    *DECISION,
                    "plant_before_sha256",
                    "plant_after_sha256",
                ),
            )
            commit = r["commit_revision"]
            truth = number(r["truth_k"], "sensor truth")
            if (
                type(r["sequence"]) is not int
                or r["sequence"] != seq
                or number(r["sample_time_s"], "sample time") != (seq - 1) * period
                or type(commit) is not int
                or commit not in [last_commit, last_commit + 1]
                or not 1 <= commit <= rev
                or not self.sensor_min <= truth <= self.sensor_max
                or r["plant_before_sha256"] != previous_sha
                or not isinstance(r["plant_after_sha256"], str)
                or not re.fullmatch("[0-9a-f]{64}", r["plant_after_sha256"])
            ):
                raise NetworkError("PI receipt chain mismatch")
            if seq == 1 and truth != self.initial_truth:
                raise NetworkError("PI initial observation mismatch")
            batch = batch + 1 if commit == last_commit else 1
            if batch > self.budget["maximum_ticks_per_transaction"]:
                raise NetworkError("PI transaction partition exceeds budget")
            memory, decision = pi_sample(self._policy, memory, truth)
            if any(
                type(r[key]) is not type(decision[key]) or r[key] != decision[key]
                for key in DECISION
            ):
                raise NetworkError("PI recurrence mismatch")
            controls = self.initial_controls.copy()
            controls[self._policy["actuator"]] = memory["output_w"]
            if (
                receipt["sequence"] != seq
                or receipt["time_s"] != (seq - 1) * period
                or receipt["commit_revision"] != seq
                or receipt["command_id"] != f"pi.tick-{seq:06d}"
                or receipt["heat_power_w"] != controls
            ):
                raise NetworkError("PI/plant command history mismatch")
            previous_sha = r["plant_after_sha256"]
            last_commit = commit
        if last_commit != rev or previous_sha != digest(plant):
            raise NetworkError("PI final plant/history mismatch")
        c._memory = memory
        c._ticks = ticks
        c._revision = rev
        c._history = deepcopy(history)
        if c.snapshot() != state:
            raise NetworkError("PI checkpoint observation mismatch")
        self._sources()
        self._install(c)
        return c.snapshot()
