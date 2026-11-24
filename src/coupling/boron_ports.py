"""Sampled IF97 liquid-port admissibility, without altering fluxes or states."""
from functools import lru_cache
import math

from physics.materials.iapws import properties_ph,saturation_temperature_k


@lru_cache(maxsize=2048)
def _water(pressure_pa,enthalpy_j_kg):
    # ThermoState is frozen. Exact keys, bounded cache, no rounded state keys.
    return properties_ph(pressure_pa,enthalpy_j_kg)


def assess_liquid_ports(*,flow_kg_s,primary_pressure_pa,supply_pressure_pa,receiver_pressure_pa,
        supply_enthalpy_j_kg,pump_work_j_kg,donor_enthalpy_j_kg,recovered_heat_w,margin_k=1.):
    values=(flow_kg_s,primary_pressure_pa,supply_pressure_pa,receiver_pressure_pa,
        supply_enthalpy_j_kg,pump_work_j_kg,donor_enthalpy_j_kg,recovered_heat_w,margin_k)
    if any(type(v) not in (float,int) or not math.isfinite(v) for v in values):
        raise ValueError('Finite real liquid-port inputs required')
    if not (0<=flow_kg_s<=1e6 and 0<=pump_work_j_kg<=1e6 and 0<margin_k<=20
            and all(1e4<=p<=16e6 for p in (primary_pressure_pa,supply_pressure_pa,receiver_pressure_pa))
            and primary_pressure_pa>max(supply_pressure_pa,receiver_pressure_pa)):
        raise ValueError('Liquid-port pressure/work/margin domain')
    if flow_kg_s==0:
        if recovered_heat_w!=0:raise ValueError('Zero flow cannot carry recovered heat')
        return dict(enabled=True,active=False,checked=False,passed=True,states={},gates={},margin_k=margin_k)
    heat=recovered_heat_w/flow_kg_s;hp=supply_enthalpy_j_kg+pump_work_j_kg
    inputs=dict(supply=(supply_pressure_pa,supply_enthalpy_j_kg),
        pump_outlet=(primary_pressure_pa,hp),primary_return=(primary_pressure_pa,hp+heat),
        primary_donor=(primary_pressure_pa,donor_enthalpy_j_kg),
        letdown_inlet=(primary_pressure_pa,donor_enthalpy_j_kg-heat),
        letdown_outlet=(receiver_pressure_pa,donor_enthalpy_j_kg-heat))
    states={};water={}
    for name,(pressure,enthalpy) in inputs.items():
        state=_water(pressure,enthalpy);water[name]=state
        saturation=saturation_temperature_k(pressure*1e-6)
        states[name]=dict(pressure_pa=pressure,requested_enthalpy_j_kg=enthalpy,
            enthalpy_j_kg=float(state.h_j_kg),temperature_k=float(state.T_k),region=int(state.region),
            quality=float(state.quality),entropy_j_kg_k=float(state.s_j_kg_k),
            saturation_temperature_k=float(saturation),subcooling_k=float(saturation-state.T_k),
            enthalpy_residual_j_kg=float(abs(state.h_j_kg-enthalpy)))
    pump_entropy=flow_kg_s*(water['pump_outlet'].s_j_kg_k-water['supply'].s_j_kg_k)
    exchanger_entropy=flow_kg_s*(water['primary_return'].s_j_kg_k+water['letdown_inlet'].s_j_kg_k-water['pump_outlet'].s_j_kg_k-water['primary_donor'].s_j_kg_k)
    valve_entropy=flow_kg_s*(water['letdown_outlet'].s_j_kg_k-water['letdown_inlet'].s_j_kg_k)
    approaches=dict(hot_end_k=float(water['primary_donor'].T_k-water['primary_return'].T_k),
        cold_end_k=float(water['letdown_inlet'].T_k-water['pump_outlet'].T_k))
    gates=dict(liquid=all(x['region']==1 and x['subcooling_k']>=margin_k for x in states.values()),
        enthalpy=max(x['enthalpy_residual_j_kg'] for x in states.values())<=.01,
        endpoint_pinches=min(approaches.values())>=-1e-7,
        pump_entropy=pump_entropy>=-1e-6,exchanger_entropy=exchanger_entropy>=-1e-6,valve_entropy=valve_entropy>=-1e-6)
    return dict(enabled=True,active=True,checked=True,passed=all(gates.values()),states=states,gates={k:bool(v) for k,v in gates.items()},
        margin_k=margin_k,endpoint_approaches=approaches,pump_entropy_w_k=float(pump_entropy),
        exchanger_entropy_w_k=float(exchanger_entropy),valve_entropy_w_k=float(valve_entropy),
        scope='Sampled water IF97 port states and endpoint pinches, not internal exchanger profiles, valve dynamics or tank entropy')
