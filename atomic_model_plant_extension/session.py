"""Transactional reduced-PWR API with source-bound, full-history restart.

Native nonlinear-solver state is deliberately rebuilt on restart. Commands and
accepted output steps are replayed; no opaque SciPy object is deserialized.
"""

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from threading import RLock

from atomic_model_workspace.common import digest
from plants.ledger import HorizonLedger
from sdk.api import apply_commands
from sdk.experiment import _source_fingerprint

from .initialization import solve
from .pwr import SourceBoundPWR

OPTIONS = dict(method="Radau", rtol=1e-6, atol=1e-5, first_step_s=0.5, jacobian_relative_step=2e-6)


def identity():
    root = Path(__file__).parent
    return dict(
        foundation_sha256=_source_fingerprint(),
        files={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        },
    )


class Session:
    """Each accepted transaction has a revision, commands, options and state hash."""

    def __init__(self, *, height_m=10.0, maximum_internal_step_s=2.0):
        if (
            type(maximum_internal_step_s) not in (int, float)
            or not math.isfinite(maximum_internal_step_s)
            or not 0.01 <= maximum_internal_step_s <= 2.0
        ):
            raise ValueError("Internal step domain .01..2 s")
        self.config = dict(height_m=height_m, maximum_internal_step_s=maximum_internal_step_s)
        self.plant = SourceBoundPWR(height_m)
        self.plant.maximum_internal_step_s = maximum_internal_step_s
        self.initialization = solve(self.plant)
        self.ledger = HorizonLedger()
        self.ledger.attach(self.plant)
        self.history = []
        self.revision = 0
        self.lock = RLock()

    def state_hash(self):
        p = self.plant
        return digest(
            dict(
                time_s=p.time_s,
                state=list(map(float, p.pack())),
                snapshot={
                    k: (list(map(float, v)) if k == "state" else v)
                    for k, v in vars(p.snapshot()).items()
                },
                feed_open=p._feed_open,
                ledger=self.ledger.checkpoint(),
                energy=deepcopy(p.last_energy_ledger),
            )
        )

    def advance(self, duration_s, *, commands=None, expected_revision=None, **options):
        with self.lock:
            if expected_revision is not None and (
                type(expected_revision) is not int or expected_revision != self.revision
            ):
                raise ValueError("Stale session revision")
            if (
                type(duration_s) not in (int, float)
                or not math.isfinite(duration_s)
                or not 0 < duration_s <= 5
            ):
                raise ValueError("Output step domain (0,5] s")
            unknown = set(options) - set(OPTIONS)
            if unknown:
                raise ValueError("Unknown solver options:" + str(sorted(unknown)))
            settings = dict(OPTIONS, **options)
            if settings["method"] not in ("Radau", "BDF"):
                raise ValueError("Unsupported solver")
            for name in set(OPTIONS) - {"method"}:
                value = settings[name]
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError("Invalid positive solver option:" + name)
            commands = {} if commands is None else deepcopy(commands)
            if not isinstance(commands, dict):
                raise ValueError("Commands must be an object")
            p = self.plant
            reset_solver_before_step = p._hybrid is None
            saved = p.snapshot()
            ledger = self.ledger.checkpoint()
            mode = p._feed_open
            diagnostics = {
                k: deepcopy(getattr(p, k))
                for k in ("last_energy_ledger", "last_motion_events", "last_protection_event_s")
            }
            try:
                apply_commands(p, commands)
                observation = p.advance(float(duration_s), **settings)
                entry = dict(
                    reset_solver_before_step=reset_solver_before_step,
                    duration_s=float(duration_s),
                    commands=commands,
                    options=settings,
                    state_sha256=self.state_hash(),
                    events=deepcopy(p.last_motion_events),
                    protection_event_s=p.last_protection_event_s,
                    requested_throttle=p.throttle,
                    effective_throttle=p.effective_steam_throttle,
                )
            except Exception:
                p.restore(saved)
                p.refresh_algebraic(p.pack())
                p._feed_open = mode
                p._hybrid = None
                p._hybrid_signature = None
                p._continuation = None
                p._continuation_signature = None
                for k, v in diagnostics.items():
                    setattr(p, k, v)
                self.ledger.rollback(ledger)
                raise
            self.history.append(entry)
            self.revision += 1
            return dict(revision=self.revision, observation=observation, transition=deepcopy(entry))

    def checkpoint(self):
        with self.lock:
            payload = dict(
                schema_version=1,
                model_id=self.plant.passport.model_id,
                source=identity(),
                config=self.config,
                revision=self.revision,
                history=deepcopy(self.history),
                final_state_sha256=self.state_hash(),
                validated=False,
                plant_referenced=False,
            )
            payload = json.loads(json.dumps(payload, allow_nan=False))
            return dict(payload=payload, sha256=digest(payload))

    @classmethod
    def restore(cls, saved, *, max_replay_steps=10000):
        if (
            not isinstance(saved, dict)
            or set(saved) != {"payload", "sha256"}
            or digest(saved["payload"]) != saved["sha256"]
        ):
            raise ValueError("Invalid plant checkpoint envelope")
        p = saved["payload"]
        if (
            p["source"] != identity()
            or p["schema_version"] != 1
            or p["model_id"] != "pwr-reduced-seven-bank"
        ):
            raise ValueError("Checkpoint source/model mismatch")
        if (
            not isinstance(p["history"], list)
            or len(p["history"]) > max_replay_steps
            or p["revision"] != len(p["history"])
        ):
            raise ValueError("Checkpoint replay budget/revision mismatch")
        result = cls(**p["config"])
        for entry in p["history"]:
            if type(entry.get("reset_solver_before_step")) is not bool:
                raise ValueError("Missing native-solver reset journal")
            if entry["reset_solver_before_step"]:
                result.plant._hybrid = None
                result.plant._hybrid_signature = None
                result.plant._continuation = None
                result.plant._continuation_signature = None
            result.advance(entry["duration_s"], commands=entry["commands"], **entry["options"])
            if digest(result.history[-1]) != digest(entry):
                raise ValueError("Checkpoint trajectory replay differs")
        if result.checkpoint() != saved:
            raise ValueError("Checkpoint final replay differs")
        return result
