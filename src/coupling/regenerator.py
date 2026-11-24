"""Steady counterflow, constant inlet heat capacities, no stored heat.

Effectiveness-NTU relation: EnergyPlus Engineering Reference, heat exchangers.
Outlet temperatures and entropy describe this capacity model, not an EOS flash.
"""
import math


def counterflow(*,flow_kg_s,ua_w_k,hot_k,cold_k,hot_cp_j_kg_k,cold_cp_j_kg_k):
    values=(flow_kg_s,ua_w_k,hot_k,cold_k,hot_cp_j_kg_k,cold_cp_j_kg_k)
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
        raise ValueError('Finite real regenerator inputs required')
    if not (0<=flow_kg_s<=1e6 and 0<=ua_w_k<=1e8 and 1<=cold_k<=2000 and 1<=hot_k<=2000
            and 1<=hot_cp_j_kg_k<=1e5 and 1<=cold_cp_j_kg_k<=1e5):
        raise ValueError('Regenerator capacity/temperature domain')
    hot_capacity=flow_kg_s*hot_cp_j_kg_k;cold_capacity=flow_kg_s*cold_cp_j_kg_k
    epsilon=heat=entropy=0.;hot_out=hot_k;cold_out=cold_k
    if flow_kg_s>0 and ua_w_k>0:
        minimum=min(hot_capacity,cold_capacity);ratio=minimum/max(hot_capacity,cold_capacity)
        difference=1.-ratio
        if difference==0.:
            epsilon=ua_w_k/(ua_w_k+minimum)
        else:
            numerator=-math.expm1(-(ua_w_k/minimum)*difference)
            epsilon=numerator/(difference+ratio*numerator)
        heat=epsilon*minimum*(hot_k-cold_k)
        hot_out=hot_k-heat/hot_capacity;cold_out=cold_k+heat/cold_capacity
        entropy=hot_capacity*math.log1p((hot_out-hot_k)/hot_k)+cold_capacity*math.log1p((cold_out-cold_k)/cold_k)
    return dict(heat_w=heat,effectiveness=epsilon,hot_outlet_capacity_temperature_k=hot_out,
        cold_outlet_capacity_temperature_k=cold_out,capacity_entropy_generation_w_k=entropy,
        hot_capacity_w_k=hot_capacity,cold_capacity_w_k=cold_capacity,
        hot_cp_j_kg_k=hot_cp_j_kg_k,cold_cp_j_kg_k=cold_cp_j_kg_k,
        hot_inlet_temperature_k=hot_k,cold_inlet_temperature_k=cold_k,ua_w_k=ua_w_k,
        closure='Steady counterflow; capacities frozen at inlet; no wall/fluid storage or pressure-drop model')
