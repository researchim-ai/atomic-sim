"""Declared hot-station actuator laws; no chemical volume-control subsystem."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RodDrive:
    """Position is inserted fraction; motion is exact between command boundaries."""
    operating_speed_s: float = .01
    scram_speed_s: float = .2

    def __post_init__(self):
        for name in ('operating_speed_s', 'scram_speed_s'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= .5:
                raise ValueError('Finite positive rod speed within declared stroke/s range')
        if self.scram_speed_s < self.operating_speed_s:
            raise ValueError('SCRAM drive must not be slower than operating drive')

    def trajectory(self, position, requested, latched, origin):
        for value in (position, requested):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Rod stroke domain')
        if type(latched) is not bool or not math.isfinite(origin):
            raise ValueError('Rod mode/time domain')
        target = 1. if latched else requested
        speed = self.scram_speed_s if latched else self.operating_speed_s
        direction = 1. if target > position else -1. if target < position else 0.
        arrival = origin + abs(target - position) / speed
        def at(time):
            if time < origin:
                raise ValueError('Rod trajectory before command boundary')
            if time >= arrival:
                return target
            return position + direction * speed * (time - origin)
        return at, dict(initial=position, requested=requested, target=target,
                        speed_stroke_s=speed, direction=direction, arrival_s=arrival,
                        scram_latched=latched)


def boron_capability():
    return dict(available=False, reason='No finite chemical-volume-control supply/letdown tanks or mass-transfer commands in this configuration',
                command=None, direct_inventory_assignment_allowed=False)
