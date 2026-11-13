"""Registered nonlinear R0: real heat-wall adapter, nonlinear ideal-engine law."""
import numpy as np
from components.heat_exchanger import HeatExchangerWall
from .mixed import Variable,ResidualComponent,MixedSystem
from .model import StationConfig,StationResidual

X=('primary','wall','secondary','cold','rotor')
Z=('q_primary','q_secondary','q_engine','p_shaft','p_export','q_cooling')


def commands(u):
    if type(u) is not dict or set(u)!={'source_power_w','generator_connected','load_fraction'}:raise ValueError('Complete nonlinear R0 commands required')
    if type(u['generator_connected']) is not bool:raise ValueError('Boolean connection state')
    for key,low,high in [('source_power_w',0.,2e6),('load_fraction',.25,2.)]:
        v=u[key]
        if type(v) not in (float,int) or not np.isfinite(v) or not low<=v<=high:raise ValueError('Nonlinear command domain')
    return dict(u)


def build(config):
    if type(config) is not StationConfig:raise ValueError('StationConfig required')
    c=config
    xs=[Variable('stock',n,'J' if n=='rotor' else 'K',True,6e7 if n=='rotor' else 600.,1. if n=='rotor' else 280.,8e8 if n=='rotor' else 900.) for n in X]
    zs=[Variable('flux',n,'W',False,1e6,0.,1e8) for n in Z]
    def energy(t,y,d,u):
        u=commands(u);q={n:y['flux.'+n] for n in Z};k=y['stock.rotor']
        gen=(c.generator_drag_nm_s_rad*u['load_fraction'] if u['generator_connected'] else 0.)*2*k/c.rotor_inertia_kg_m2
        bearing=c.bearing_drag_nm_s_rad*2*k/c.rotor_inertia_kg_m2
        powers=[u['source_power_w']-q['q_primary'],q['q_primary']-q['q_secondary'],q['q_secondary']-q['q_engine'],
                q['q_engine']-q['p_shaft']+bearing+gen-q['p_export']-q['q_cooling'],q['p_shaft']-gen-bearing]
        caps=[c.primary_capacity_j_k,c.wall_capacity_j_k,c.secondary_capacity_j_k,c.cold_capacity_j_k,1.]
        return [cap*d['stock.'+n]-p for n,cap,p in zip(X,caps,powers,strict=True)]
    def closure(t,y,d,u):
        u=commands(u);tp,tw,ts,tc,k=[y['stock.'+n] for n in X]
        wall=HeatExchangerWall(c.wall_capacity_j_k,c.primary_ua_w_k,c.secondary_ua_w_k,tw)
        q1,q2=wall.heat_flows_w(tp,ts);heat=c.engine_ua_w_k*(ts-tc)
        # Efficiency is bounded by Carnot via a specified fractional factor.
        work=c.engine_efficiency*(1-tc/ts)*heat
        gen=(c.generator_drag_nm_s_rad*u['load_fraction'] if u['generator_connected'] else 0.)*2*k/c.rotor_inertia_kg_m2
        expected=[q1,q2,heat,work,c.generator_efficiency*gen,c.cooling_ua_w_k*(tc-c.ambient_temperature_k)]
        return [y['flux.'+n]-v for n,v in zip(Z,expected,strict=True)]
    def domain(y):
        if y[2]<y[3] or not 1<=np.sqrt(2*y[4]/c.rotor_inertia_kg_m2)<=400:raise ValueError('Nonlinear R0 engine/rotor domain')
    a=ResidualComponent('stock',xs,{v.key:v.unit for v in zs},[1e6]*5,energy)
    b=ResidualComponent('flux',zs,{v.key:v.unit for v in xs},[1e6]*6,closure)
    return MixedSystem([a,b],domain)


def guess(config):
    x,_=StationResidual(config).steady();return np.r_[x,np.ones(6)*1e5]


def continuation(config,loads=(.8,1.,1.2)):
    system=build(config);y=guess(config);rows=[]
    for load in loads:
        u=dict(source_power_w=config.nominal_source_power_w,generator_connected=True,load_fraction=load)
        y,r=system.steady(y,u);rows.append(dict(load_fraction=load,state=y.tolist(),receipt=r))
    return rows
