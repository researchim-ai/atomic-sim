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


def rod_arrival_event(motion,position,time_s):
    """Report an exact-mode arrival, including a rounded target at the endpoint.

    Position is never snapped. An early clock can qualify only when position
    equals the target exactly and the timing discrepancy is bounded by ULPs.
    """
    required={'direction','arrival_s','target','speed_stroke_s','scram_latched'}
    if type(motion) is not dict or not required<=set(motion):raise ValueError('Rod motion record required')
    for value in (position,time_s,motion['arrival_s'],motion['target'],motion['speed_stroke_s']):
        if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Finite rod arrival data required')
    if (not 0<=position<=1 or not 0<=motion['target']<=1 or time_s<0 or motion['arrival_s']<0 or
            not 0<motion['speed_stroke_s']<=.5 or type(motion['direction']) not in (int,float) or motion['direction'] not in (-1,0,1)
            or type(motion['scram_latched']) is not bool):raise ValueError('Rod arrival domain')
    if not motion['direction']:return None
    arrival=motion['arrival_s']
    if time_s<arrival:
        if position!=motion['target']:return None
        clock_bound=8*(math.ulp(time_s)+math.ulp(arrival)+math.ulp(motion['target'])/motion['speed_stroke_s'])
        if arrival-time_s>clock_bound:raise ValueError('Rod target reached before analytic arrival beyond roundoff')
    elif position!=motion['target']:raise ValueError('Analytic rod arrival lacks the exact target state')
    return dict(time_s=min(arrival,time_s),kind='rod_arrived',position=motion['target'],
        reason='scram_stop' if motion['scram_latched'] else 'requested_position')
