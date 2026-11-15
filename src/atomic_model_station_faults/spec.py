"""Tick-aligned, privileged fault specifications; no stochastic scheduling."""
from dataclasses import dataclass
import math
from types import MappingProxyType

CATALOG = MappingProxyType({
    'pump_off': ('equipment', 'Loss of drive; existing fluid momentum remains'),
    'drive_limit': ('equipment', 'Absolute upper bound on requested pump speed fraction'),
    'bypass_stuck': ('equipment', 'Capture actual bypass position; immobilize stem'),
    'cooling_loss': ('equipment', 'Multiply external water supply; finite loop retained'),
    'bias': ('sensor', 'Add constant offset at sample acquisition'),
    'drift': ('sensor', 'Add SI-unit drift per second since fault onset'),
    'stuck': ('sensor', 'Hold first acquired faulty sample, including its noise'),
    'dropout': ('sensor', 'Suppress sample delivery; old sample ages naturally'),
    'failure': ('sensor', 'Deliver unavailable quality through normal transport delay'),
})


def finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class Fault:
    name: str
    kind: str
    target: str
    start_tick: int
    end_tick: int | None = None
    value: float = 0.

    def __post_init__(self):
        if type(self.name) is not str or not self.name.isidentifier() or len(self.name) > 64:
            raise ValueError('Fault name')
        if type(self.kind) is not str or self.kind not in CATALOG:
            raise ValueError('Fault kind')
        if type(self.target) is not str or not self.target.isidentifier():
            raise ValueError('Fault target')
        if type(self.start_tick) is not int or not 0 <= self.start_tick <= 1000000:
            raise ValueError('Fault start tick')
        if self.end_tick is not None and (type(self.end_tick) is not int or not self.start_tick < self.end_tick <= 1000001):
            raise ValueError('Fault half-open interval')
        if not finite(self.value) or abs(self.value) > 1e12:
            raise ValueError('Fault finite magnitude')
        if self.kind in ('pump_off', 'drive_limit') and self.target not in ('feed', 'condensate', 'cooling'):
            raise ValueError('Supported pump target')
        if self.kind == 'bypass_stuck' and self.target != 'bypass':
            raise ValueError('Supported stuck valve')
        if self.kind == 'cooling_loss' and self.target != 'external_water':
            raise ValueError('Supported heat-rejection fault')
        if self.kind in ('drive_limit', 'cooling_loss'):
            if not 0 <= self.value <= 1:
                raise ValueError('Fault fraction')
        elif self.kind not in ('bias', 'drift') and self.value != 0:
            raise ValueError('Unused fault magnitude must be zero')

    @property
    def domain(self):
        return CATALOG[self.kind][0]

    def active(self, tick):
        return self.start_tick <= tick and (self.end_tick is None or tick < self.end_tick)


def canonical(faults, *, domain=None):
    if type(faults) not in (list, tuple) or len(faults) > 64 or any(type(f) is not Fault for f in faults):
        raise ValueError('Fault list budget/type')
    result = tuple(sorted(faults, key=lambda f: (f.start_tick, f.name)))
    if len({f.name for f in result}) != len(result):
        raise ValueError('Duplicate fault name')
    if domain is not None and any(f.domain != domain for f in result):
        raise ValueError('Wrong fault domain')
    for i, a in enumerate(result):
        for b in result[i + 1:]:
            if a.domain == b.domain and a.target == b.target and b.start_tick < (a.end_tick if a.end_tick is not None else 1000002):
                raise ValueError('Overlapping faults on same target')
    return result
