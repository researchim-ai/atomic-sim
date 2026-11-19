"""Bounded bulk-property smooth-tube Gnielinski variant, not a rod-bundle law.

Equation: ANL/RERTR/TM-11-38 Rev.2, pp.127-128 (PDF135-136),
https://publications.anl.gov/anlpubs/2018/04/143592.pdf
Gnielinski (1976), Int. Chemical Engineering16:359-368.
The selected domain is deliberately narrower than the cited Re envelope.
"""
from math import isfinite,log10,sqrt
from numbers import Real
from core.status import Applicability,DomainError,ModelPassport,ModelStatus
from physics.hydraulics.closures import CorrelationResult

MODELS=('dittus_boelter','gnielinski')
MAX_RICHARDSON_D=.01  # Engineering screening policy, not a universal error bound.
PASSPORT_GNIELINSKI=ModelPassport(model_id='gnielinski-bulk-smooth-tube',version='1.0.0',
    status=ModelStatus.VERIFIED,phenomenon='Single-phase bulk-property heat transfer in smooth round tubes',
    approximations=('Bulk fluid properties; no wall-property correction or entrance correction',
                    'Smooth circular equivalent tube, heating only; not a rod bundle or mixed-convection closure'),
    applicability='Selected envelope: 3000<=Re<=1e6,0.6<=Pr<=200,L/D>10; station adds Ri_D<=0.01 screening',
    data_sources=('https://publications.anl.gov/anlpubs/2018/04/143592.pdf#page=136',
                  'Gnielinski1976, Int.Chem.Eng.16:359-368; Darcy friction factor per ANL p.127, round-tube numerator1'))


def validate_model(name):
    if type(name) is not str or name not in MODELS:raise ValueError('Explicit film model required')
    return name


def nusselt_gnielinski(reynolds,prandtl):
    for value in (reynolds,prandtl):
        if not isinstance(value,Real) or isinstance(value,bool) or not isfinite(value):raise DomainError('Finite Gnielinski inputs required')
    if not 3000<=reynolds<=1e6 or not .6<=prandtl<=200:raise DomainError('Gnielinski outside selected Re3000..1e6,Pr0.6..200 envelope')
    # Darcy, not Fanning. log10 form follows the cited ANL equation.
    factor=(1.82*log10(reynolds)-1.64)**-2
    nu=(factor/8)*(reynolds-1000)*prandtl/(1+12.7*sqrt(factor/8)*(prandtl**(2/3)-1))
    return CorrelationResult(float(nu),Applicability.IN_DOMAIN,PASSPORT_GNIELINSKI.model_id)
