"""Sampled PI with external reset feedback; actuator units are fractions."""
from dataclasses import dataclass
from atomic_model_station_signals.model import finite


@dataclass(frozen=True)
class PIConfig:
    sensor:str
    output:str
    target:float
    scale:float
    kp:float=.05
    ki:float=.01
    direction:int=1
    bias:float=0.
    lower:float=0.
    upper:float=1.
    tracking_tau_s:float=2.
    rate_per_s:float=1.
    feedforward_sensor:str|None=None
    feedforward_reference:float=0.
    feedforward_scale:float=1.
    inventory_deadband_m:float=0.

    def __post_init__(self):
        if type(self.sensor) is not str or not self.sensor.isidentifier():raise ValueError('PI sensor')
        if self.output not in ('rod_target','steam_admission','steam_bypass','feed_flow_fraction','condensate_flow_fraction','makeup_valve','feed_inventory_fraction'):raise ValueError('Qualified PI output')
        if self.feedforward_sensor is not None and (type(self.feedforward_sensor) is not str or not self.feedforward_sensor.isidentifier()):raise ValueError('PI feed-forward sensor')
        if not finite(self.feedforward_reference) or abs(self.feedforward_reference)>1e12 or not finite(self.feedforward_scale) or not 1e-6<=self.feedforward_scale<=1e12:raise ValueError('PI feed-forward reference/scale')
        for key in ('target','scale','kp','ki','bias','lower','upper','tracking_tau_s','rate_per_s'):
            if not finite(getattr(self,key)):raise ValueError('Finite PI parameter:'+key)
        if not 0<=self.target<=1e12 or not 1e-6<=self.scale<=1e12:raise ValueError('PI target/scale')
        if not 0<=self.kp<=1e3 or not 0<=self.ki<=1e3:raise ValueError('PI gains')
        if type(self.direction) is not int or self.direction not in (-1,1):raise ValueError('PI direction')
        if not 0<=self.lower<self.upper<=1 or not self.lower<=self.bias<=self.upper:raise ValueError('PI bounds/bias')
        if not .01<=self.tracking_tau_s<=1e6 or not 0<self.rate_per_s<=100:raise ValueError('PI tracking/rate')
        if not finite(self.inventory_deadband_m) or not 0<=self.inventory_deadband_m<=.001:
            raise ValueError('Inventory neutral band: 0..1 mm')
        if self.inventory_deadband_m and (self.output!='feed_inventory_fraction' or self.bias!=.5
            or not self.lower<=.5<=self.upper or self.feedforward_sensor is not None):
            raise ValueError('Neutral band requires unbiased split-range inventory loop')


class PI:
    def __init__(self,config):
        if type(config) is not PIConfig:raise ValueError('PIConfig required')
        self.config=config;self.integral=0.;self.valid_last=False;self.last_error=0.

    def propose(self,value,target,feedback,dt,*,entering=False,feedforward=0.):
        c=self.config
        if type(entering) is not bool or any(not finite(x) for x in (value,target,feedback,dt,feedforward)) or not 0<=feedback<=1 or not 0<dt<=c.tracking_tau_s:raise ValueError('PI sample/clock')
        error=c.direction*(target-value)/c.scale
        if abs(error)>1e6 or abs(feedforward)>1e6:raise ValueError('PI measurement scale domain')
        base=feedback-c.bias-c.kp*error-feedforward if entering or not self.valid_last else self.integral
        if c.inventory_deadband_m>0 and abs(target-value)<=c.inventory_deadband_m:
            # Neutral is a real actuator request, not a projected tank state.
            # Discard stored demand in-band; commit still tracks delivered feedback.
            error=0.;base=.5-c.bias;feedforward=0.
        raw=c.bias+c.kp*error+base+feedforward
        limited=min(c.upper,max(c.lower,raw))
        command=min(feedback+c.rate_per_s*dt,max(feedback-c.rate_per_s*dt,limited))
        command=min(c.upper,max(c.lower,command))
        return dict(command=float(command),raw=float(raw),error=float(error),base_integral=float(base),feedforward=float(feedforward),saturated=bool(command!=raw))

    def commit(self,proposal,feedback,dt,*,automatic):
        c=self.config
        if type(automatic) is not bool or not finite(feedback) or not 0<=feedback<=1 or not finite(dt) or not 0<dt<=c.tracking_tau_s:raise ValueError('PI external reset feedback')
        if proposal is not None:
            keys={'command','raw','error','base_integral','feedforward','saturated'}
            if (type(proposal) is not dict or set(proposal)!=keys or type(proposal['saturated']) is not bool
                or any(not finite(proposal[k]) for k in keys-{'saturated'})
                or not c.lower<=proposal['command']<=c.upper or abs(proposal['error'])>1e6):raise ValueError('PI proposal')
        if proposal is None:
            if automatic:raise ValueError('Automatic PI requires a valid proposal')
            integral=feedback-c.bias;error=0.
        elif not automatic:
            integral=feedback-c.bias-c.kp*proposal['error']-proposal['feedforward'];error=proposal['error']
        else:
            integral=proposal['base_integral']+c.ki*proposal['error']*dt+(feedback-proposal['raw'])*dt/c.tracking_tau_s
            error=proposal['error']
        if not finite(integral):raise ValueError('PI integral overflow')
        self.integral=float(integral);self.valid_last=automatic;self.last_error=float(error)

    def state(self):return dict(integral=float(self.integral),valid_last=self.valid_last,last_error=float(self.last_error))
