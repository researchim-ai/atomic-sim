"""One observation/action contract shared by Python and Gymnasium clients."""

import hashlib
import math
from copy import deepcopy
from pathlib import Path
from threading import RLock

from atomic_model_controlled_plant import Session
from atomic_model_training.schema import state_schema
from atomic_model_training.spec import TIERS, admit, integer
from atomic_model_workspace.common import digest
from sdk.api import validate_commands


def _positive(value, maximum):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError("Expected finite positive number <= " + str(maximum))
    return float(value)


class Episode:
    """Deterministic full-state episode; seed is metadata, not a physical perturbation.

    Commands are persistent absolute settings except the core's latched flags.
    Solver/domain failures propagate; they are never converted into RL rewards.
    """

    def __init__(self, *, duration_s=0.1, max_steps=100, tier="training", seed=0):
        duration_s = _positive(duration_s, 5)
        integer(max_steps, 1, 256)
        integer(seed, 0, 2**32 - 1)
        if tier not in TIERS:
            raise ValueError("Unknown solver tier")
        self.config = dict(duration_s=duration_s, max_steps=max_steps, tier=tier, seed=seed)
        self._session = Session()
        self._lock = RLock()

    def observe(self):
        with self._lock:
            session = self._session
            plant = session.plant
            terminated = bool(plant.scram_latched)
            result = dict(
                schema_version=1,
                model_id="pwr-reduced-flow-controlled",
                revision=session.revision,
                time_s=float(plant.time_s),
                state=list(map(float, plant.pack())),
                state_schema=state_schema(plant),
                thermal_power_w=float(plant.thermal_power_w_from_state(plant.pack())),
                nominal_fission_power_w=float(plant.fission_power_nom),
                terminated=terminated,
                truncated=session.revision >= self.config["max_steps"],
                validated=False,
                plant_referenced=False,
            )
            result["observation_id"] = digest(dict(observation=result, config=self.config))
            return result

    def step(self, commands, *, observation_id):
        with self._lock:
            previous = self.observe()
            if observation_id != previous["observation_id"]:
                raise ValueError("Stale observation")
            if previous["terminated"] or previous["truncated"]:
                raise ValueError("Episode finished; create or reset an episode")
            if not isinstance(commands, dict):
                raise ValueError("Commands must be an object")
            commands = validate_commands(commands)
            transition = self._session.advance(
                self.config["duration_s"],
                commands=commands,
                expected_revision=previous["revision"],
                **TIERS[self.config["tier"]],
            )
            return dict(
                observation=self.observe(),
                transition=transition["transition"],
                balances=self._session.ledger.report(),
            )

    def checkpoint(self):
        with self._lock:
            payload = dict(
                schema_version=1,
                config=deepcopy(self.config),
                adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                core=self._session.checkpoint(),
            )
            return dict(payload=payload, sha256=digest(payload))

    @classmethod
    def restore(cls, saved):
        if not isinstance(saved, dict) or set(saved) != {"payload", "sha256"}:
            raise ValueError("Invalid episode checkpoint")
        payload = saved["payload"]
        if digest(payload) != saved["sha256"] or set(payload) != {
            "schema_version",
            "config",
            "adapter_sha256",
            "core",
        }:
            raise ValueError("Invalid episode checkpoint payload")
        if (
            payload["schema_version"] != 1
            or payload["adapter_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ):
            raise ValueError("Episode adapter changed")
        result = cls(**payload["config"])
        result._session = Session.restore(payload["core"], max_replay_steps=256)
        history = result._session.history
        if len(history) > result.config["max_steps"] or any(
            entry["duration_s"] != result.config["duration_s"]
            or any(entry["options"][k] != v for k, v in TIERS[result.config["tier"]].items())
            for entry in history
        ):
            raise ValueError("Episode/core configuration mismatch")
        return result

    def training_case(self, *, case_id, family, split="validation"):
        """Export the accepted control schedule for the existing campaign runner.

        This is a replay case, not a statistical claim of independent validation.
        """
        with self._lock:
            return admit(
                dict(
                    id=case_id,
                    family=family,
                    split=split,
                    seed=self.config["seed"],
                    tier=self.config["tier"],
                    steps=[
                        dict(duration_s=e["duration_s"], commands=deepcopy(e["commands"]))
                        for e in self._session.history
                    ],
                )
            )
