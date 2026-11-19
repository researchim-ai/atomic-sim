"""Scalar arithmetic and IF97 derivative consistency, not plant validation."""
import argparse
from decimal import Decimal,localcontext
from pathlib import Path
from atomic_model_station.film_correlations import nusselt_gnielinski
import atomic_model_station.film_correlations as implementation
from physics.materials import iapws
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json


def run(output):
    out=Path(output)
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    rows=[]
    with localcontext() as ctx:
        ctx.prec=60
        for re,denominator in [(10000,'5.64'),(100000,'7.46'),(1000000,'9.28')]:
            # Pr=8 gives Pr^(2/3)=4 exactly. No logarithm/fractional exponent in this oracle.
            d=Decimal(denominator)
            expected=float(Decimal(re-1000)/(d*d)/(1+Decimal('38.1')/(d*Decimal(8).sqrt())))
            actual=nusselt_gnielinski(re,8.).value;relative=abs(actual/expected-1)
            rows.append(dict(reynolds=re,prandtl=8.,reference_nusselt=expected,actual_nusselt=actual,relative=relative))
    raw=iapws._region1_raw(500.,10.);beta=float(raw['dv_dt']/raw['v']);eps=.001
    def volume(t):return 1/iapws.properties_pt(10e6,t).rho_kg_m3
    finite=(volume(500+eps)-volume(500-eps))/(2*eps*volume(500.));error=abs(beta/finite-1)
    report=dict(passed=all(r['relative']<3e-15 for r in rows) and error<1e-7,
        implementation_sha256=sha(implementation.__file__),driver_sha256=sha(__file__),
        oracle='60-digit decimal reference at Pr=8; no logarithm or fractional exponent in oracle',nusselt=rows,
        beta=dict(analytic_per_k=beta,finite_difference_per_k=finite,relative=error,pressure_pa=10e6,temperature_k=500.,difference_step_k=eps),
        scope='Arithmetic oracle and internal EOS derivative consistency; not an independent EOS or experimental validation',validated=False)
    atomic_json(out,seal(report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=run(a.output);print('scalar references passed:',r['passed']);return 0 if r['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
