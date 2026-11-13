"""Freeze wall assimilation after alarm and limit requested-power transitions."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
from atomic_model_multiphysics.provenance import seal, unseal
from atomic_model_thermal_observer.monitor import predict


def signature():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


@dataclass(frozen=True)
class RecoveryConfig:
    dt_s: float = .00025
    max_power_w: float = 100000.
    capacity_j_k: float = 300.
    conductance_w_k: float = 8000.
    ambient_ua_w_k: float = 20.
    gain_w_k: float = 40000.
    max_command_step: float = .1

    def __post_init__(self):
        bounds = dict(dt_s=(1e-6, .001), max_power_w=(1., 1e6),
                      capacity_j_k=(1., 1e7), conductance_w_k=(1., 1e5),
                      ambient_ua_w_k=(0., 1e5), gain_w_k=(0., 1e6),
                      max_command_step=(1e-6, 1.))
        for key, (low, high) in bounds.items():
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('Recovery parameter domain: ' + key)


class ThermalRecovery:
    """No truth/fault input. Latching recovery; other sensor channels are assumed valid.

    Power at observation i is the applied power over the previous interval.
    Temperature channels follow the benchmark convention: kelvin minus 440.
    This point predictor is neither a state enclosure nor a safety controller.
    """
    def __init__(self, config=None):
        self.config = config if config is not None else RecoveryConfig()
        if type(self.config) is not RecoveryConfig:
            raise ValueError('RecoveryConfig required')
        self._history = []
        self._previous = None
        self._estimate = None
        self._command = None
        self._alarm = False

    def act(self, observation, alarm, normal):
        o = np.asarray(observation)
        if o.shape != (6,) or o.dtype.kind not in 'if' or not np.isfinite(o).all():
            raise ValueError('Finite observation6 required')
        if not np.all((-200 <= o[:4]) & (o[:4] <= 600)) or not np.all((0 <= o[4:]) & (o[4:] <= 1)):
            raise ValueError('Recovery observation domain')
        if type(alarm) is not bool or type(normal) not in (int, float) or not math.isfinite(normal) or not 0 <= normal <= 1:
            raise ValueError('Boolean alarm and finite normal action0..1 required')
        if len(self._history) >= 256:
            raise ValueError('Recovery sample budget256')
        # All validation precedes state mutation.
        cfg = self.config
        latched = self._alarm or alarm
        estimate = 440. + float(o[0]) if self._estimate is None else self._estimate
        if self._previous is not None:
            estimate = predict(estimate, 440. + self._previous[1], 440. + float(o[3]),
                               float(o[4]) * cfg.max_power_w, cfg.dt_s,
                               cfg.capacity_j_k, cfg.conductance_w_k, cfg.ambient_ua_w_k)
        if not latched:
            estimate = 440. + float(o[0])
            command = float(normal)
        else:
            target = 440. + float(o[2])
            feedforward = cfg.conductance_w_k * (target - (440. + float(o[1])))
            feedforward += cfg.ambient_ua_w_k * (target - (440. + float(o[3])))
            desired = float(np.clip((feedforward + cfg.gain_w_k * (target - estimate)) / cfg.max_power_w, 0., 1.))
            previous = float(o[4]) if self._command is None else self._command
            command = float(np.clip(desired, max(0., previous - cfg.max_command_step),
                                    min(1., previous + cfg.max_command_step)))
        self._estimate = estimate
        self._command = command
        self._alarm = latched
        self._previous = list(map(float, o))
        self._history.append(dict(observation=self._previous.copy(), alarm=alarm, normal=float(normal)))
        return dict(action=command, estimated_wall_k=estimate, alarm=latched)

    def checkpoint(self):
        return seal(deepcopy(dict(config=asdict(self.config), source=signature(),
                                  history=self._history, estimate=self._estimate,
                                  command=self._command, alarm=self._alarm)))

    @classmethod
    def restore(cls, saved):
        p = unseal(deepcopy(saved))
        if p['source'] != signature() or not isinstance(p['history'], list) or len(p['history']) > 256:
            raise ValueError('Recovery source/budget mismatch')
        controller = cls(RecoveryConfig(**p['config']))
        for row in p['history']:
            controller.act(row['observation'], row['alarm'], row['normal'])
        if controller.checkpoint() != saved:
            raise ValueError('Recovery replay differs')
        return controller
