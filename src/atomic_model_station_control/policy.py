"""Deterministic sampled protection state; thresholds are research parameters."""
from copy import deepcopy
from dataclasses import dataclass
from atomic_model_station_signals.model import CATALOG,finite

@dataclass(frozen=True)
class ProtectionConfig:
    pressure_open_pa:float=6.8e6
    pressure_close_pa:float=6.6e6
    pressure_trip_pa:float=7.5e6
    overspeed_fraction:float=1.05
    power_trip_w:float=3.6e9
    startup_grace_ticks:int=0

    def __post_init__(self):
        for k in ('pressure_open_pa','pressure_close_pa','pressure_trip_pa','overspeed_fraction','power_trip_w'):
            if not finite(getattr(self,k)):raise ValueError('Finite protection parameter')
        if not 5.2e6<=self.pressure_close_pa<self.pressure_open_pa<self.pressure_trip_pa<=7.9e6:raise ValueError('Pressure protection hysteresis')
        if not .95<=self.overspeed_fraction<=1.09 or not 1e6<=self.power_trip_w<=1e11:raise ValueError('Research protection thresholds')
        if type(self.startup_grace_ticks) is not int or not 0<=self.startup_grace_ticks<=600:raise ValueError('Protection startup grace')


def valid_reading(frame,name,quantity):
    try:
        reading=frame['measurements'][name]
        if reading['unit']!=CATALOG[quantity][1] or reading['quality']!=['good'] or not finite(reading['value']):return None
        if not finite(reading['sampled_at_s']) or not finite(reading['available_at_s']) or not reading['sampled_at_s']<=reading['available_at_s']<=frame['time_s']:return None
        if not finite(reading['age_s']) or reading['age_s']<0 or abs(reading['age_s']-(frame['time_s']-reading['sampled_at_s']))>1e-8:return None
        return float(reading['value'])
    except (KeyError,TypeError):return None


class Supervision:
    def __init__(self,config=None):
        self.config=config if config is not None else ProtectionConfig()
        if type(self.config) is not ProtectionConfig:raise ValueError('ProtectionConfig required')
        self.pressure_active=False;self.trip_latched=False;self.trip_reasons=[];self.events=[]
        self.last_time_s=None;self.last_tick=-1

    def advance(self,frame,requested,*,tick,nominal_speed,backend_scram=False,backend_turbine_trip=False,additional_required=None):
        if type(frame) is not dict or not finite(frame.get('time_s')) or not 0<=frame['time_s']<=60000000:raise ValueError('Protection clock')
        if type(tick) is not int or tick!=self.last_tick+1 or tick>1000000:raise ValueError('Consecutive protection tick required')
        if self.last_time_s is not None and frame['time_s']<=self.last_time_s:raise ValueError('Duplicate protection epoch')
        if not finite(nominal_speed) or not 50<=nominal_speed<=400:raise ValueError('Nominal protection speed')
        if any(type(v) is not bool for v in (backend_scram,backend_turbine_trip)):raise ValueError('Backend latch flags')
        if type(requested) is not dict or any(type(requested.get(k)) is not bool for k in ('grid_available','scram','turbine_trip')):raise ValueError('Protection request flags')
        if additional_required is not None and (type(additional_required) is not dict or any(type(n) is not str or not n.isidentifier() or type(q) is not str or q not in CATALOG or n in ('power','pressure','speed') for n,q in additional_required.items())):raise ValueError('Additional protection channels')
        trial=deepcopy(self);trial.last_time_s=float(frame['time_s']);trial.last_tick=tick
        result=trial._advance(frame,requested,nominal_speed,backend_scram,backend_turbine_trip,additional_required or {})
        self.__dict__.update(trial.__dict__);return result

    def _event(self,kind,**kw):self.events.append(dict(time_s=self.last_time_s,kind=kind,**kw))

    def _advance(self,frame,requested,nominal_speed,backend_scram,backend_turbine_trip,additional_required):
        c=self.config;reasons=[]
        read={n:valid_reading(frame,n,q) for n,q in [('power','fission_power_w'),('pressure','steam_pressure_pa'),('speed','turbine_speed_rad_s')]}
        read.update({n:valid_reading(frame,n,q) for n,q in sorted(additional_required.items())})
        if self.last_tick>=c.startup_grace_ticks:reasons.extend('measurement_unavailable:'+n for n,value in read.items() if value is None)
        p=read['pressure'];active=self.pressure_active
        if p is not None:
            if not active and p>=c.pressure_open_pa:self.pressure_active=True
            elif active and p<=c.pressure_close_pa:self.pressure_active=False
            if p>=c.pressure_trip_pa:reasons.append('measured_pressure_high')
        if active!=self.pressure_active:self._event('pressure_interlock_entered' if self.pressure_active else 'pressure_interlock_released')
        if read['power'] is not None and read['power']>=c.power_trip_w:reasons.append('measured_power_high')
        if read['speed'] is not None and read['speed']>=c.overspeed_fraction*nominal_speed:reasons.append('measured_overspeed')
        if not requested['grid_available']:reasons.append('grid_unavailable')
        if requested['scram']:reasons.append('operator_scram')
        if requested['turbine_trip']:reasons.append('operator_turbine_trip')
        if backend_scram or backend_turbine_trip:reasons.append('backend_trip_observed')
        new=sorted(set(reasons)-set(self.trip_reasons))
        if reasons and not self.trip_latched:
            self.trip_latched=True;self._event('protection_latched',reasons=sorted(reasons))
        for reason in new:self.trip_reasons.append(reason);self._event('trip_reason_recorded',reason=reason)
        self.trip_reasons.sort();return read

    def request_reset(self):
        if self.last_time_s is None:raise ValueError('Protection reset before first observation')
        reason='new_station_session_required' if self.trip_latched else 'pressure_hysteresis_active' if self.pressure_active else 'no_latch'
        self._event('reset_denied' if self.trip_latched or self.pressure_active else 'reset_not_required',reason=reason)
        return dict(reset_in_place_available=False,reason=reason)
