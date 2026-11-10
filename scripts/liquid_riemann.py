"""Independent nonlinear liquid Riemann oracle (IF97 region 1 only).

Shock Hugoniot and isentropic quadrature, not an HLLC implementation.
Reference: Menikoff & Plohr, Rev. Mod. Phys. 61, 75 (1989),
https://doi.org/10.1103/RevModPhys.61.75 . Numerical, not symbolic exactness.
No fallback, metastable continuation or two-phase wave construction.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
from scipy.integrate import quad
from scipy.optimize import brentq
from atomic_model_hem_qualified import eos

SOURCE = 'https://doi.org/10.1103/RevModPhys.61.75'


def liquid(state):
    if (state.region != 1 or state.phase != 'liquid' or
        not all(math.isfinite(x) for x in (state.pressure_pa,state.temperature_k,
            state.density_kg_m3,state.internal_energy_j_kg,state.entropy_j_kg_k))):
        raise ValueError('Oracle requires finite IF97 liquid states')
    # Reject inconsistent hand-made states, not merely their phase label.
    if eos.pt(state.pressure_pa,state.temperature_k) != state:
        raise ValueError('Oracle requires consistent forward EOS states')
    eos.characteristic_speeds(state)
    return state


def temperature_root(p,residual):
    if not eos.P_MIN+100 < p < eos.P_MAX-100:
        raise ValueError('Oracle pressure leaves derivative domain')
    upper=min(623.15,eos.saturation(p)[0].temperature_k-1e-6)
    def f(t):return residual(eos.pt(p,t))
    if f(eos.T_MIN)*f(upper)>0:
        raise ValueError('No admissible liquid temperature root')
    t=brentq(f,eos.T_MIN,upper,xtol=2e-11,rtol=1e-14)
    return liquid(eos.pt(p,t))


class WaveCurve:
    def __init__(self,state):self.initial=liquid(state)

    @lru_cache(maxsize=512)
    def isentrope(self,p):
        initial=self.initial
        if p==initial.pressure_pa:return initial
        return temperature_root(p,lambda s:s.entropy_j_kg_k-initial.entropy_j_kg_k)

    @lru_cache(maxsize=512)
    def at(self,p):
        initial=self.initial;p0=initial.pressure_pa;v0=1/initial.density_kg_m3
        if p==p0:return 0.,initial,'contact',0.
        if p>p0:
            def hugoniot(s):
                return s.internal_energy_j_kg-initial.internal_energy_j_kg+.5*(p+p0)*(1/s.density_kg_m3-v0)
            state=temperature_root(p,hugoniot)
            compression=v0-1/state.density_kg_m3
            if compression<=0:raise ValueError('Non-compressive shock')
            if state.entropy_j_kg_k<initial.entropy_j_kg_k-2e-8:
                raise ValueError('Shock violates entropy condition')
            return math.sqrt((p-p0)*compression),state,'shock',abs(hugoniot(state))
        def integrand(q):
            state=self.isentrope(float(q));c=eos.characteristic_speeds(state)[1]
            return 1/(state.density_kg_m3*c)
        value,error=quad(integrand,p0,p,epsabs=2e-10,epsrel=2e-10,limit=80)
        if error>2e-8:raise ValueError('Rarefaction quadrature not converged')
        return value,self.isentrope(p),'rarefaction',error


@dataclass(frozen=True)
class RiemannSolution:
    pressure_pa: float
    velocity_m_s: float
    left: object
    right: object
    left_wave: str
    right_wave: str
    velocity_residual_m_s: float
    left_residual: float
    right_residual: float
    left_outer_speed_m_s: float
    left_inner_speed_m_s: float
    right_inner_speed_m_s: float
    right_outer_speed_m_s: float

    def interface_flux(self):
        """Godunov flux only when x/t=0 lies between acoustic waves.

        A fan-interior or supersonic sample requires a different sampler and
        is refused explicitly; star-state flux must not stand in for it.
        """
        if not self.left_inner_speed_m_s<=0<=self.right_inner_speed_m_s:
            raise ValueError('Interface not between acoustic waves')
        state=self.left if self.velocity_m_s>=0 else self.right
        rho=state.density_kg_m3;v=self.velocity_m_s
        energy=rho*(state.internal_energy_j_kg+.5*v*v)
        return (rho*v,rho*v*v+self.pressure_pa,v*(energy+self.pressure_pa))


def solve(left,right,velocity_left=0.,velocity_right=0.,*,pressure_bracket):
    """Match velocity at the material contact using a caller-owned bracket.

    No hidden expansion across saturation or outside the qualified EOS chart.
    A caller must supply bounds containing a liquid, entropy-admissible root.
    """
    vl=float(velocity_left);vr=float(velocity_right)
    lo,hi=map(float,pressure_bracket)
    if not all(math.isfinite(x) for x in (vl,vr,lo,hi)) or not eos.P_MIN+100<lo<hi<eos.P_MAX-100:
        raise ValueError('Finite velocities and interior pressure bracket required')
    l=WaveCurve(left);r=WaveCurve(right)
    def residual(p):return l.at(float(p))[0]+r.at(float(p))[0]+vr-vl
    if residual(lo)*residual(hi)>0:raise ValueError('Pressure bracket does not contain contact root')
    p=(left.pressure_pa if left.pressure_pa==right.pressure_pa and vl==vr and lo<=left.pressure_pa<=hi
       else brentq(residual,lo,hi,xtol=2e-5,rtol=1e-13))
    fl,ls,lk,le=l.at(float(p));fr,rs,rk,re=r.at(float(p))
    uleft=vl-fl;uright=vr+fr;v=.5*(uleft+uright)
    if abs(uleft-uright)>2e-8:raise ValueError('Contact velocity mismatch')
    def speeds(curve,state,kind,velocity,sign):
        initial=curve.initial
        if kind=='shock':
            mass_flux=math.sqrt((p-initial.pressure_pa)/(1/initial.density_kg_m3-1/state.density_kg_m3))
            speed=velocity+sign*mass_flux/initial.density_kg_m3
            upstream=velocity+sign*eos.characteristic_speeds(initial)[1]
            downstream=v+sign*eos.characteristic_speeds(state)[1]
            if sign*(speed-upstream)<-1e-7 or sign*(downstream-speed)<-1e-7:
                raise ValueError('Shock violates Lax wave ordering')
            return speed,speed
        return velocity+sign*eos.characteristic_speeds(initial)[1],v+sign*eos.characteristic_speeds(state)[1]
    louter,linner=speeds(l,ls,lk,vl,-1)
    router,rinner=speeds(r,rs,rk,vr,1)
    if louter>linner+1e-7 or rinner>router+1e-7:
        raise ValueError('Nonclassical wave ordering not supported')
    return RiemannSolution(p,v,ls,rs,lk,rk,abs(uleft-uright),le,re,louter,linner,rinner,router)
