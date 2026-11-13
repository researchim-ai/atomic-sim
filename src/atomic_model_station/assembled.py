"""R0 graph composed from owned inventories and conservative power blocks."""
from .assembly import Inventory,Boundary,PowerBlock,AffineAssembly
from .model import StationResidual,STATE_NAMES
from .session import StationSession


def build_r0(config,power,connected):
    # Reuse command validation only; physical matrices are assembled below.
    if type(power) not in (int,float) or not 0<=power<=2e6 or type(connected) is not bool:
        raise ValueError('Source/load command domain')
    import math
    if not math.isfinite(power):raise ValueError('Finite source power required')
    c=config;p,w,s,cold,k=STATE_NAMES
    stores=[Inventory('primary',p,'thermal',c.primary_capacity_j_k,('source','wall')),
        Inventory('wall',w,'thermal',c.wall_capacity_j_k,('primary','secondary')),
        Inventory('secondary',s,'thermal',c.secondary_capacity_j_k,('wall','engine')),
        Inventory('cold',cold,'thermal',c.cold_capacity_j_k,('engine','bearing','generator','cooling')),
        Inventory('rotor',k,'mechanical',1.,('engine','bearing','generator'))]
    boundaries=[Boundary('heat_source','thermal'),Boundary('ambient','thermal'),Boundary('grid','electrical_power')]
    blocks=[];connections=[]
    def add(owner,values,links):
        blocks.append(PowerBlock(owner,values))
        connections.extend([((owner,port),target) for port,target in links.items()])
    add('source',{'in':('thermal',{},-power),'out':('thermal',{},power)},
        {'in':('heat_source','port'),'out':('primary','source')})
    for label,a,b,ua,out_port,in_port in [('primary_wall','primary','wall',c.primary_ua_w_k,'wall','primary'),
                                         ('wall_secondary','wall','secondary',c.secondary_ua_w_k,'secondary','wall')]:
        state={x.owner:x.state for x in stores}
        add(label,{'hot':('thermal',{state[a]:-ua,state[b]:ua},0.),'cold':('thermal',{state[a]:ua,state[b]:-ua},0.)},
            {'hot':(a,out_port),'cold':(b,in_port)})
    u=c.engine_ua_w_k;e=c.engine_efficiency
    add('engine',{'hot':('thermal',{s:-u,cold:u},0.),'cold':('thermal',{s:(1-e)*u,cold:-(1-e)*u},0.),
                  'shaft':('mechanical',{s:e*u,cold:-e*u},0.)},
        {'hot':('secondary','engine'),'cold':('cold','engine'),'shaft':('rotor','engine')})
    b=2*c.bearing_drag_nm_s_rad/c.rotor_inertia_kg_m2
    add('bearing',{'shaft':('mechanical',{k:-b},0.),'heat':('thermal',{k:b},0.)},
        {'shaft':('rotor','bearing'),'heat':('cold','bearing')})
    g=2*(c.generator_drag_nm_s_rad if connected else 0.)/c.rotor_inertia_kg_m2;eta=c.generator_efficiency
    add('generator',{'shaft':('mechanical',{k:-g},0.),'loss':('thermal',{k:(1-eta)*g},0.),'export':('electrical_power',{k:eta*g},0.)},
        {'shaft':('rotor','generator'),'loss':('cold','generator'),'export':('grid','port')})
    u=c.cooling_ua_w_k;offset=u*c.ambient_temperature_k
    add('cooling',{'cold':('thermal',{cold:-u},offset),'ambient':('thermal',{cold:u},-offset)},
        {'cold':('cold','cooling'),'ambient':('ambient','port')})
    return AffineAssembly(stores,boundaries,blocks,connections)


class AssembledStationResidual(StationResidual):
    def operators(self,power,connected):return build_r0(self.config,power,connected).operators()
    def steady(self):
        c=self.config;p=c.nominal_source_power_w
        scales=[max(p/cap,1.) for cap in (c.primary_capacity_j_k,c.wall_capacity_j_k,c.secondary_capacity_j_k,c.cold_capacity_j_k)]+[p]
        y,residual=build_r0(c,p,True).steady(scales);self.validate(y)
        return y,residual


class AssembledStationSession(StationSession):
    def __init__(self,config=None):
        super().__init__(config)
        self.model=AssembledStationResidual(self.config)
        self._state,self.initialization_residual=self.model.steady()
    def snapshot(self):
        result=super().snapshot()
        result['passport']['model_id']='npp-r0-affine-assembled'
        result['passport']['scope']='Port-connected affine power-block R0 assembly; no fluid/steam solver'
        result['assembly']=build_r0(self.config,self._power,self._connected).manifest()
        transitions=[];time=0.;power=self.config.nominal_source_power_w;connected=True
        for row in self._history:
            commands={k:row[k] for k,previous in [('source_power_w',power),('generator_connected',connected)] if row[k]!=previous}
            if commands:transitions.append(dict(time_s=time,commands=commands))
            time+=row['duration_s'];power=row['source_power_w'];connected=row['generator_connected']
        result['command_transitions']=transitions
        return result
    def advance_schedule(self,duration_s,events):
        from copy import deepcopy
        import math
        if type(duration_s) not in (int,float) or not math.isfinite(duration_s) or not 0<duration_s<=60:
            raise ValueError('Schedule duration0..60s')
        if type(events) is not list or len(events)>32:raise ValueError('Schedule event budget0..32')
        normalized=[];last=-1.;power=self._power;connected=self._connected
        for event in events:
            if type(event) is not dict or set(event)!={'at_s','commands'}:raise ValueError('Event schema')
            time=event['at_s'];commands=event['commands']
            if type(time) not in (int,float) or not math.isfinite(time) or not last<time<duration_s or time<0:
                raise ValueError('Strictly ordered unique event times in [0,duration)')
            if type(commands) is not dict or not commands or not set(commands)<={'source_power_w','generator_connected'}:
                raise ValueError('Declared nonempty command patch')
            power=commands.get('source_power_w',power);connected=commands.get('generator_connected',connected)
            self.model.operators(power,connected)
            normalized.append(dict(at_s=float(time),commands=dict(commands)));last=time
        trial=type(self).restore(self.checkpoint());cursor=0.;patch={};segments=[]
        for event in normalized:
            if event['at_s']>cursor:
                segments.append(trial.advance(event['at_s']-cursor,**patch));patch={}
            patch.update(event['commands']);cursor=event['at_s']
        segments.append(trial.advance(duration_s-cursor,**patch))
        self.__dict__.update(deepcopy(trial.__dict__))
        return dict(final=self.snapshot(),segments=segments,planned_events=normalized)
