"""Self-similar liquid wave sampler and conservative finite-volume averages.

Independent of the numerical graph/flux; uses the nonlinear wave oracle.
No centre-point substitution at shocks or inside rarefaction fans.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
from scipy.optimize import brentq
from atomic_model_hem_qualified import eos
from scripts.liquid_riemann import solve,WaveCurve


@dataclass(frozen=True)
class Sample:
    state: object
    velocity_m_s: float
    zone: str

    def conserved(self,energy_reference=0.):
        if not math.isfinite(energy_reference):raise ValueError('Finite energy reference required')
        rho=self.state.density_kg_m3;v=self.velocity_m_s
        return np.array([rho,rho*v,rho*(self.state.internal_energy_j_kg-energy_reference+.5*v*v)])


class Profile:
    def __init__(self,left,right,velocity_left=0.,velocity_right=0.,*,pressure_bracket):
        self.solution=solve(left,right,velocity_left,velocity_right,pressure_bracket=pressure_bracket)
        self.left=WaveCurve(left);self.right=WaveCurve(right)
        self.vl=float(velocity_left);self.vr=float(velocity_right)

    @lru_cache(maxsize=4096)
    def sample(self,xi):
        xi=float(xi)
        if not math.isfinite(xi):raise ValueError('Finite similarity coordinate required')
        s=self.solution
        if xi<=s.velocity_m_s:
            if xi<=s.left_outer_speed_m_s:return Sample(self.left.initial,self.vl,'left_initial')
            if xi>=s.left_inner_speed_m_s:return Sample(s.left,s.velocity_m_s,'left_star')
            curve=self.left;velocity=self.vl;sign=-1;kind=s.left_wave
        else:
            if xi>=s.right_outer_speed_m_s:return Sample(self.right.initial,self.vr,'right_initial')
            if xi<=s.right_inner_speed_m_s:return Sample(s.right,s.velocity_m_s,'right_star')
            curve=self.right;velocity=self.vr;sign=1;kind=s.right_wave
        if kind!='rarefaction':raise ValueError('Unsupported wave interior')
        def evaluate(p):
            f,state,_,_=curve.at(float(p));v=velocity+sign*f
            return v+sign*eos.characteristic_speeds(state)[1]-xi,state,v
        lo,hi=sorted((s.pressure_pa,curve.initial.pressure_pa))
        p=brentq(lambda p:evaluate(p)[0],lo,hi,xtol=2e-5,rtol=1e-13)
        residual,state,v=evaluate(p)
        if abs(residual)>2e-7:raise ValueError('Fan characteristic residual exceeded')
        return Sample(state,v,'left_fan' if sign<0 else 'right_fan')

    def cell_average(self,a,b,time,*,origin=0.,energy_reference=0.):
        """Average conserved densities, splitting all waves and the contact.

        Two Gauss rules control the quadrature on smooth fan pieces. Constant
        pieces are integrated exactly. An initial-time discontinuity is split
        explicitly, without evaluating x/t.
        """
        if not all(math.isfinite(x) for x in (a,b,time,origin,energy_reference)) or a>=b or time<0:
            raise ValueError('Finite ordered cell and nonnegative time required')
        if time==0:
            left=max(0.,min(b,origin)-a);left=min(left,b-a)
            return (left*Sample(self.left.initial,self.vl,'left_initial').conserved(energy_reference)+
                    (b-a-left)*Sample(self.right.initial,self.vr,'right_initial').conserved(energy_reference))/(b-a)
        s=self.solution
        speeds=(s.left_outer_speed_m_s,s.left_inner_speed_m_s,s.velocity_m_s,s.right_inner_speed_m_s,s.right_outer_speed_m_s)
        cuts=sorted(set([a,b]+[origin+time*v for v in speeds if a<origin+time*v<b]))
        total=np.zeros(3)
        for lo,hi in zip(cuts,cuts[1:]):
            middle=.5*(lo+hi);value=self.sample((middle-origin)/time)
            if not value.zone.endswith('_fan'):
                total+=(hi-lo)*value.conserved(energy_reference);continue
            def integrate(order):
                points,weights=np.polynomial.legendre.leggauss(order)
                return .5*(hi-lo)*sum(w*self.sample((middle+.5*(hi-lo)*x-origin)/time).conserved(energy_reference)
                                       for x,w in zip(points,weights,strict=True))
            coarse,fine=integrate(8),integrate(16)
            if np.any(abs(fine-coarse)>1e-10*np.maximum(abs(fine),1.)+1e-9):
                raise ValueError('Fan cell quadrature did not converge')
            total+=fine
        return total/(b-a)
