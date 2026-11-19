"""Explicit sampled quality gate and unmodified initial wall compatibility."""
import numpy as np
from atomic_model_station_regular.model import _finite_real
from .system import ResolvedThermalSystem
from .audit import audit_resolved


class ThermalQualityError(ValueError):
    def __init__(self, receipt):
        super().__init__('Resolved thermal candidate failed heat or sampled retained wall gates')
        self.receipt = receipt


def initial_wall_compatibility(system, *, tolerance=1e-8):
    if type(system) is not ResolvedThermalSystem:
        raise ValueError('ResolvedThermalSystem required')
    _finite_real(tolerance,'wall compatibility tolerance',1e-14,1e-3)
    rows = {}
    for model,q in system.temperature:
        key=(model.k,model.m)
        _,_,lap=model.jets(q,[1.])
        # u=0 and theta=0 on the fixed wall: theta_t=0 requires kappa*lap(theta)+Q=0.
        rows[str(key)]=float(abs(system.diffusivity*lap[0]+system.source(key,[1.])[0]))
    maximum=max(rows.values())
    return dict(maximum_temperature_rate_mismatch=maximum,rows=rows,
        compatible=maximum<=tolerance,tolerance=tolerance,state_modified=False,
        scope='First thermal time compatibility at fixed isothermal wall; no momentum compatibility claim')


def accept_resolved(system, result, *, heat_tolerance=1e-10, wall_tolerance=1e-8):
    """Raise with evidence on failure. Neither the system nor the candidate is changed."""
    _finite_real(heat_tolerance,'physical heat tolerance',1e-14,1e-3)
    _finite_real(wall_tolerance,'retained wall tolerance',1e-14,1e-3)
    audit=audit_resolved(system,result,radius=np.linspace(.005,1.,129))
    heat=result.report.get('maximum_physical_heat_budget_residual')
    if type(heat) not in (int,float) or not np.isfinite(heat) or heat<0:
        raise ValueError('Finite physical heat budget evidence required')
    heat_passed=bool(heat<=heat_tolerance)
    wall_passed=bool(max(audit['maximum_wall_momentum_residual'],audit['maximum_wall_thermal_residual'])<=wall_tolerance)
    receipt=dict(time=result.report['time'],passed=heat_passed and wall_passed,physical_heat_passed=heat_passed,
        retained_wall_passed=wall_passed,heat_tolerance=heat_tolerance,wall_tolerance=wall_tolerance,
        maximum_physical_heat_budget_residual=heat,audit=audit,
        sampled_radii=129,state_modified=False,retained_modes_only=True,
        spatial_refinement_checked=False,general_strong_pde_qualified=False,
        station_usable=False,validated=False)
    if not receipt['passed']:
        raise ThermalQualityError(receipt)
    return receipt
