"""Observable-only detection and explicitly ambiguous diagnostic hypotheses."""
from copy import deepcopy
from dataclasses import asdict, dataclass
from .spec import finite


@dataclass(frozen=True)
class MonitorConfig:
    sensor: str = 'pressure'
    reference: str = 'pressure_reference'
    unit: str = 'Pa'
    disagreement: float = 100000.
    stuck_samples: int = 3
    moving_reference: float = 10.

    def __post_init__(self):
        if any(type(n) is not str or not n.isidentifier() for n in (self.sensor, self.reference)) or self.sensor == self.reference:
            raise ValueError('Independent monitor channel names')
        if type(self.unit) is not str or not self.unit or len(self.unit) > 32:
            raise ValueError('Monitor unit')
        if not finite(self.disagreement) or not 0 < self.disagreement <= 1e12:
            raise ValueError('Monitor disagreement threshold')
        if not finite(self.moving_reference) or not 0 < self.moving_reference <= self.disagreement:
            raise ValueError('Monitor reference-motion threshold')
        if type(self.stuck_samples) is not int or not 2 <= self.stuck_samples <= 1000:
            raise ValueError('Monitor repeated-sample budget')


class Monitor:
    def __init__(self, config=None):
        self.config = config if config is not None else MonitorConfig()
        if type(self.config) is not MonitorConfig:
            raise ValueError('MonitorConfig required')
        self.last_time = None
        self.last_sample = None
        self.last_value = None
        self.reference_origin = None
        self.repeat_count = 0
        self.active = []
        self.events = []

    def advance(self, frame):
        if type(frame) is not dict or not finite(frame.get('time_s')) or frame['time_s'] < 0:
            raise ValueError('Monitor frame clock')
        if self.last_time is not None and frame['time_s'] <= self.last_time:
            raise ValueError('Monitor clock must advance')
        trial = deepcopy(self)
        result = trial._advance(frame)
        self.__dict__.update(trial.__dict__)
        return result

    def _reading(self, frame, name):
        try:
            r = frame['measurements'][name]
            if (r['unit'] != self.config.unit or r['quality'] != ['good'] or not finite(r['value'])
                    or not finite(r['sampled_at_s']) or not finite(r['available_at_s'])
                    or not 0 <= r['sampled_at_s'] <= r['available_at_s'] <= frame['time_s']
                    or not finite(r['age_s']) or abs(r['age_s'] - (frame['time_s'] - r['sampled_at_s'])) > 1e-8):
                return None
            return r
        except (KeyError, TypeError):
            return None

    def _advance(self, frame):
        c = self.config
        primary = self._reading(frame, c.sensor)
        reference = self._reading(frame, c.reference)
        reasons = []
        if primary is None or reference is None:
            reasons.append('measurement_unavailable')
            self.repeat_count = 0
            self.last_sample = None
            self.last_value = None
            self.reference_origin = None
        else:
            aligned = primary['sampled_at_s'] == reference['sampled_at_s']
            if aligned and abs(primary['value'] - reference['value']) > c.disagreement:
                reasons.append('paired_disagreement')
            # Re-reading an old transported sample must never count as a frozen sensor.
            if self.last_sample is None or primary['sampled_at_s'] > self.last_sample:
                if aligned and self.last_value == primary['value']:
                    self.repeat_count += 1
                else:
                    self.repeat_count = 1
                    self.reference_origin = reference['value'] if aligned else None
                self.last_sample = primary['sampled_at_s']
                self.last_value = primary['value']
            if (aligned and self.reference_origin is not None and self.repeat_count >= c.stuck_samples
                    and abs(reference['value'] - self.reference_origin) >= c.moving_reference):
                reasons.append('unchanging_channel_with_moving_reference')
        for reason in sorted(set(reasons) - set(self.active)):
            self.events.append(dict(time_s=frame['time_s'], kind='detected', reason=reason))
        for reason in sorted(set(self.active) - set(reasons)):
            self.events.append(dict(time_s=frame['time_s'], kind='cleared', reason=reason))
        self.last_time = float(frame['time_s'])
        self.active = sorted(reasons)
        return dict(time_s=self.last_time, alarms=deepcopy(self.active),
                    hypotheses=['instrument_or_reference_or_alignment_problem'] if reasons else [],
                    uniquely_diagnosed=False)

    def state(self):
        return dict(config=asdict(self.config), last_time=self.last_time, last_sample=self.last_sample,
                    last_value=self.last_value, reference_origin=self.reference_origin,
                    repeat_count=self.repeat_count, active=deepcopy(self.active), events=deepcopy(self.events))
