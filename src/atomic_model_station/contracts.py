"""SI connection contracts; no EOS, choking or hydraulic-flow closure implied."""
from dataclasses import dataclass
import math


def scalar(x):
    if type(x) not in (int,float) or not math.isfinite(x):raise ValueError('Finite SI scalar required')
    return float(x)


@dataclass(frozen=True)
class FluidState:
    pressure_pa:float
    enthalpy_j_kg:float
    outward_mass_flow_kg_s:float
    composition:tuple=()
    phase:str='liquid'
    def __post_init__(self):
        if not 1<=scalar(self.pressure_pa)<=1e8 or abs(scalar(self.enthalpy_j_kg))>1e8 or abs(scalar(self.outward_mass_flow_kg_s))>1e7:raise ValueError('Fluid contract domain')
        if self.phase not in ('liquid','vapor','mixture'):raise ValueError('Explicit fluid phase required')
        if type(self.composition) is not tuple or len(self.composition)>16:raise ValueError('Bounded immutable composition required')
        seen=set();total=0.
        for item in self.composition:
            if type(item) is not tuple or len(item)!=2:raise ValueError('Species/fraction pair')
            species,fraction=item
            if type(species) is not str or not species or len(species)>80 or species in seen or not 0<=scalar(fraction)<=1:raise ValueError('Composition admission')
            seen.add(species);total+=fraction
        if total>1+1e-14:raise ValueError('Composition sum exceeds unity')


def fluid_connection(a,b,tolerance=1e-10):
    if type(a) is not FluidState or type(b) is not FluidState or scalar(tolerance)<=0:raise ValueError('Fluid states/tolerance required')
    residual=[a.pressure_pa-b.pressure_pa,a.outward_mass_flow_kg_s+b.outward_mass_flow_kg_s]
    if abs(residual[1])>tolerance:raise ValueError('Connection mass flow is not conservative')
    q=a.outward_mass_flow_kg_s;up=a if q>=0 else b
    species=set(dict(a.composition))|set(dict(b.composition));fractions=dict(up.composition)
    # At zero flow no enthalpy/composition mixing or artificial transport occurs.
    return dict(residual_pressure_pa=residual[0],residual_mass_kg_s=residual[1],
        upstream='a' if q>0 else 'b' if q<0 else None,
        mass_a_to_b_kg_s=q,enthalpy_a_to_b_w=q*up.enthalpy_j_kg,
        species_a_to_b_kg_s={s:q*fractions.get(s,0.) for s in sorted(species)},
        phase_admitted=up.phase if q else None)


def thermal_connection(temperature_a_k,heat_a_w,temperature_b_k,heat_b_w):
    ta,qa,tb,qb=map(scalar,(temperature_a_k,heat_a_w,temperature_b_k,heat_b_w))
    if min(ta,tb)<=0:raise ValueError('Absolute temperatures required')
    return dict(effort_residual_k=ta-tb,flow_residual_w=qa+qb)


def mechanical_connection(speed_a_rad_s,torque_a_nm,speed_b_rad_s,torque_b_nm):
    wa,ta,wb,tb=map(scalar,(speed_a_rad_s,torque_a_nm,speed_b_rad_s,torque_b_nm))
    return dict(speed_residual_rad_s=wa-wb,torque_residual_nm=ta+tb,power_residual_w=wa*ta+wb*tb)


def electrical_connection(power_a_w,connected_a,power_b_w,connected_b):
    pa,pb=map(scalar,(power_a_w,power_b_w))
    if type(connected_a) is not bool or type(connected_b) is not bool or connected_a!=connected_b:raise ValueError('Compatible explicit electrical topology required')
    if not connected_a and (pa!=0 or pb!=0):raise ValueError('Disconnected power must be zero')
    return dict(power_residual_w=pa+pb,connected=connected_a)
