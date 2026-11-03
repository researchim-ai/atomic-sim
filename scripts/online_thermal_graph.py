"""Revision-checked, transactional sample-and-hold control of thermal graphs.

Controllers remain outside the model. A failed step commits neither its command
nor any integration work. Checkpoints bind the original model and committed
command history; hashes provide consistency, not publisher authentication.
"""

import json
from copy import deepcopy
from pathlib import Path

from core.network import NetworkError, number
from scripts.nonlinear_thermal_graph import digest
from scripts.scheduled_thermal_graph import ScheduledThermalGraph
from scripts.scheduled_thermal_graph import signature as scheduled_signature
from scripts.solve_hydraulic_graph import _keys, sha

BUDGET = Path("validation/ai05_online_graph_budget_v1.json")


def signature():
    result = scheduled_signature()
    result["drivers"]["scripts/online_thermal_graph.py"] = sha(__file__)
    result["budgets"][str(BUDGET)] = sha(BUDGET)
    return result


def wrap(model, history):
    return dict(
        format="atomic-scheduled-thermal-graph",
        schema_version=1,
        model_input=deepcopy(model),
        timeline=deepcopy(history),
    )


class OnlineThermalGraph:
    """One command batch per positive-duration step; all observations detached."""

    def __init__(self, model_input):
        self._base = deepcopy(model_input)
        self.identity = digest(self._base)
        self._model = ScheduledThermalGraph(wrap(self._base, []))
        self._history = []
        self._revision = 0
        self.budget = json.loads(BUDGET.read_text())
        self.captured = signature()

    @property
    def revision(self):
        return self._revision

    def _sources(self):
        if signature() != self.captured:
            raise NetworkError("Online model source/budget drift")

    def observation(self):
        return dict(
            revision=self.revision,
            history_sha256=digest(self._history),
            model=self._model.snapshot(),
        )

    def step(self, changes, duration_s, *, expected_revision):
        self._sources()
        if type(expected_revision) is not int or expected_revision != self.revision:
            raise NetworkError("Stale or invalid online command revision")
        if self.revision >= self.budget["maximum_commands"]:
            raise NetworkError("Online command budget exhausted")
        duration = number(duration_s, "duration_s", positive=True)
        time = self._model.snapshot()["time_s"]
        if time + duration > self.budget["maximum_episode_s"]:
            raise NetworkError("Online episode horizon exceeded")
        # Stage a detached machine. Only this adapter may extend a committed
        # prefix, exactly at the current clock; scheduled inputs remain immutable.
        candidate = deepcopy(self._model)
        events = [*deepcopy(self._history), dict(time_s=time, changes=deepcopy(changes))]
        timeline = candidate._parse_timeline(events, candidate._kernel)
        candidate._timeline = timeline
        candidate._times = tuple(e["time_s"] for e in timeline)
        candidate.identity = digest(wrap(self._base, list(timeline)))
        kernel, active, receipt, switch = candidate._switch(
            candidate._kernel, candidate._active, candidate._cursor
        )
        candidate._kernel, candidate._active = kernel, active
        candidate._cursor += 1
        candidate._receipts.append(receipt)
        result = candidate.advance(duration)
        history = deepcopy(list(timeline))
        observation = dict(
            revision=self.revision + 1, history_sha256=digest(history), model=candidate.snapshot()
        )
        report = dict(observation=observation, command_switch=switch, integration=result)
        self._sources()
        self._model, self._history = candidate, history
        self._revision += 1
        return report

    def checkpoint(self):
        payload = dict(
            schema_version=1,
            model="online-thermal-graph-v1",
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
            raise NetworkError("Online checkpoint digest mismatch")
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
            or p["model"] != "online-thermal-graph-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
        ):
            raise NetworkError("Online checkpoint input/model/source mismatch")
        revision, history = p["revision"], p["history"]
        if (
            type(revision) is not int
            or not isinstance(history, list)
            or revision != len(history)
            or not 0 <= revision <= self.budget["maximum_commands"]
        ):
            raise NetworkError("Online checkpoint history/revision mismatch")
        candidate = ScheduledThermalGraph(wrap(self._base, history))
        state = candidate.restore(p["scheduled_checkpoint"])
        time = state["time_s"]
        if (
            time > self.budget["maximum_episode_s"]
            or state["active_event_count"] != revision
            or (revision == 0 and time != 0)
            or (revision > 0 and (history[0]["time_s"] != 0 or history[-1]["time_s"] >= time))
        ):
            raise NetworkError("Online checkpoint has future, unheld or missing commands")
        observation = dict(revision=revision, history_sha256=digest(history), model=state)
        self._sources()
        self._model, self._history, self._revision = candidate, deepcopy(history), revision
        return observation
