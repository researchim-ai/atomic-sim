"""Exact liquid-first Newton branch with one raw Gibbs evaluation per iterate.

Only the original adapter's first liquid chart is accelerated. Other chart
orders, saturation, endpoints and failed attempts use the unchanged adapter.
No new seed, tolerance, projection, root selection or EOS coefficient is used.
"""
import numpy as np
from atomic_model_hem_qualified import eos
from physics.materials import iapws


def liquid_first_rho_u(density,internal_energy,pressure_seed_pa=1e6,temperature_seed_k=450.):
    rho=eos.number(density,'density');u=eos.number(internal_energy,'energy')
    p=eos.number(pressure_seed_pa,'pressure seed');t=eos.number(temperature_seed_k,'temperature seed')
    if rho<=0:raise ValueError('Nonpositive density')
    eos._domain(p,t)
    target=np.array([1/rho,u]);scale=np.array([1/rho,max(abs(u),1e5)])
    f,g=eos.saturation(p)
    x=(target[0]-1/f.density_kg_m3)/(1/g.density_kg_m3-1/f.density_kg_m3)
    if x<0:
        y=np.array([p,min(t,623.15)])
        for _ in range(32):
            pressure,second=map(float,y)
            try:
                raw=eos._raw(pressure,second,1)
                value=np.array([raw['v'],raw['u']*1000])
                # Same liquid Jacobian arithmetic as single_jacobian; reuse raw.
                vt=raw['dv_dt'];vp=raw['dv_dp_mpa']/1e6
                j=np.array([[vp,vt],[-second*vt-pressure*vp,raw['cp']*1000-pressure*vt]])
                residual=value-target;norm=float(np.max(abs(residual/scale)))
                if abs(residual[0])<=5e-14*target[0] and abs(residual[1])<=1e-7:
                    ts=eos.saturation(pressure)[0].temperature_k
                    # Leave all saturation/endpoint decisions to the old adapter.
                    if abs(second-ts)>=1e-8 and second<=ts:
                        candidate=eos._state(pressure,second,1)
                        if iapws.classify_region(second,pressure*1e-6)==1 and eos._matches(candidate,rho,u):
                            return candidate
                    break
                step=np.linalg.solve(j,-residual)
                factor=min(1.,.4*pressure/max(abs(step[0]),1e-30))
                factor=min(factor,30/max(abs(step[1]),1e-30))
                accepted=False
                for _ in range(18):
                    trial=y+factor*step;pp,ss=map(float,trial)
                    if eos.P_MIN<=pp<=eos.P_MAX and eos.T_MIN<=ss<=623.15:
                        raw=eos._raw(pp,ss,1);probe=np.array([raw['v'],raw['u']*1000])
                        if np.isfinite(probe).all() and np.max(abs((probe-target)/scale))<norm:
                            y=trial;accepted=True;break
                    factor*=.5
                if not accepted:break
            except (ValueError,FloatingPointError,np.linalg.LinAlgError):break
    return eos.rho_u(rho,u,p,t)
