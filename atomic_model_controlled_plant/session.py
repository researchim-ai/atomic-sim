"""Complete source-bound controlled-plant sessions and charging modes."""

import hashlib
import platform
from pathlib import Path
from threading import RLock

import numpy as np
import scipy

from atomic_model_plant_extension.initialization import solve
from atomic_model_plant_extension.session import Session as BaseSession
from atomic_model_plant_extension.session import identity as base_identity
from atomic_model_workspace.common import digest
from plants.ledger import HorizonLedger
from scripts import extended_natural_initialization as charging

from .model import ControlledPWR


def identity():
    root = Path(__file__).parent
    from atomic_model_hem import eos
    from atomic_model_workspace import common

    return dict(
        plant=base_identity(),
        runtime=dict(
            python=platform.python_version(),
            system=platform.system(),
            machine=platform.machine(),
            numpy=np.__version__,
            scipy=scipy.__version__,
        ),
        implementation={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))
        },
        shared_bootstrap={
            label: hashlib.sha256(
                Path(module.__file__).parent.joinpath("__init__.py").read_bytes()
            ).hexdigest()
            for label, module in [
                ("water_types", eos),
                ("canonical", common),
                ("charging", charging),
            ]
        },
        shared_leaves={
            label: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for label, module in [
                ("water_types", eos),
                ("canonical", common),
                ("charging", charging),
            ]
        },
    )


class Session(BaseSession):
    def __init__(
        self,
        *,
        height_m=10.0,
        maximum_internal_step_s=2.0,
        initialization="nominal",
        implementation_sha256=None,
    ):
        actual = digest(identity())
        if implementation_sha256 is not None and implementation_sha256 != actual:
            raise ValueError("Controlled source/runtime changed")
        if (
            type(maximum_internal_step_s) not in (int, float)
            or not np.isfinite(maximum_internal_step_s)
            or not 0.01 <= maximum_internal_step_s <= 2.0
        ):
            raise ValueError("Internal step domain.01..2s")
        if initialization not in ("nominal", "natural") or (
            initialization == "natural" and height_m != 10.0
        ):
            raise ValueError("Unsupported charging configuration")
        self.config = dict(
            height_m=height_m,
            maximum_internal_step_s=maximum_internal_step_s,
            initialization=initialization,
            implementation_sha256=actual,
        )
        self.plant = ControlledPWR(height_m)
        p = self.plant
        p.maximum_internal_step_s = maximum_internal_step_s
        if initialization == "nominal":
            self.initialization = solve(p)
        else:
            original = p.drive_torque
            rows = []
            for k in range(101):
                fraction = k / 100.0
                power = 1 - 0.95 * fraction
                p.throttle = power
                p.drive_torque = original * (1 - fraction)
                report = charging.charge(p, power)
                rows.append(
                    dict(fraction=fraction, power=power, scaled_defect=report["scaled_defect"])
                )
            if abs(p.omega) > 1e-8:
                raise ValueError("Natural shaft root not zero")
            state = p.pack()
            state[11] = 0.0
            p.drive_torque = original
            p.motor_on = False
            p.refresh_algebraic(state)
            rate = p.derivatives(0.0, state)
            if (
                abs(rate[11]) > 1e-12
                or max(abs(rate[14 + 3 * i]) for i in range(len(p.network.volumes))) > 1.0
            ):
                raise ValueError("Natural charging dimensional residual failed")
            self.initialization = dict(
                final=report,
                homotopy=rows,
                warmup_steps=0,
                exact_shaft_speed_rad_s=0.0,
                initial_fill="core pressure16MPa determines mass; no nominal-fill transient is asserted",
            )
        self.ledger = HorizonLedger()
        self.ledger.attach(p)
        self.history = []
        self.revision = 0
        self.lock = RLock()

    @classmethod
    def restore(cls, saved, *, max_replay_steps=10000):
        if (
            not isinstance(saved, dict)
            or set(saved) != {"payload", "sha256"}
            or digest(saved["payload"]) != saved["sha256"]
        ):
            raise ValueError("Invalid controlled checkpoint envelope")
        payload = saved["payload"]
        if (
            payload["source"] != base_identity()
            or payload["model_id"] != "pwr-reduced-flow-controlled"
            or payload["schema_version"] != 1
        ):
            raise ValueError("Controlled checkpoint source/model changed")
        if (
            not isinstance(payload["history"], list)
            or len(payload["history"]) > max_replay_steps
            or type(payload["revision"]) is not int
            or payload["revision"] != len(payload["history"])
        ):
            raise ValueError("Controlled checkpoint replay budget/revision mismatch")
        result = cls(**payload["config"])
        for entry in payload["history"]:
            if type(entry.get("reset_solver_before_step")) is not bool:
                raise ValueError("Missing solver-reset journal")
            if entry["reset_solver_before_step"]:
                result.plant._hybrid = None
                result.plant._hybrid_signature = None
                result.plant._continuation = None
                result.plant._continuation_signature = None
            result.advance(entry["duration_s"], commands=entry["commands"], **entry["options"])
            if digest(result.history[-1]) != digest(entry):
                raise ValueError("Controlled trajectory replay differs")
        if result.checkpoint() != saved:
            raise ValueError("Controlled final replay differs")
        return result
