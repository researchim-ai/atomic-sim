"""First-order reactivity response; diffusion, scattering and fission held fixed."""
import hashlib
from pathlib import Path

import numpy as np
from atomic_model_spatial import Model, eigenvalue
from atomic_model_spatial.solvers import factor, relative_residual
from atomic_model_spatial.workflow import identity as spatial_identity


def signature():
    return dict(files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(Path(__file__).parent.glob('*.py'))}, spatial=spatial_identity())


def absorption_temperature_response(specification, derivative_per_m_k, *, derivative_provenance, max_iterations=10000):
    """Return d(1-1/k)/dT with exact cell-wise adjoint absorption weights.

    Supplied derivatives must use the model's group/cell order and SI units.
    This computes sensitivity, not temperature-dependent nuclear data.
    """
    if not isinstance(derivative_provenance,str) or not derivative_provenance.strip():
        raise ValueError('Derivative provenance required')
    if type(max_iterations) is not int or not 1<=max_iterations<=100000:
        raise ValueError('Adjoint iteration budget required')
    model=Model(specification)
    derivative=np.asarray(derivative_per_m_k)
    if derivative.shape!=(model.g,model.n) or derivative.dtype.kind not in 'if' or not np.isfinite(derivative).all():
        raise ValueError('Finite signed group/cell derivative required, no broadcasting')
    derivative=derivative.astype(float,copy=True)
    forward=eigenvalue(model,tolerance=1e-12)
    phi=np.asarray(forward['flux']).ravel()
    loss=model.loss.T.tocsc();fission=model.fission.T.tocsc()
    solve=factor(loss).solve
    psi=np.ones(model.g*model.n);psi/=psi.sum()
    for iteration in range(1,max_iterations+1):
        raw=solve(fission@psi);k=float(raw.sum())
        if not np.isfinite(raw).all() or np.any(raw<0) or not np.isfinite(k) or k<=0:
            raise ValueError('Invalid adjoint eigenpair')
        new=raw/k
        residual=relative_residual(loss,new,fission@new/k)
        change=float(np.max(abs(new-psi))/max(np.max(abs(new)),1e-300))
        psi=new
        if residual<=1e-12 and change<=1e-12:break
    else:raise ValueError('Adjoint iteration budget exhausted')
    if abs(k-forward['k'])/forward['k']>1e-9:
        raise ValueError('Forward/adjoint eigenvalue mismatch')
    denominator=float(psi@(model.fission@phi))
    if not np.isfinite(denominator) or denominator<=0:raise ValueError('Degenerate adjoint normalization')
    gradient=(-psi*phi/denominator).reshape(model.g,model.n)*model.width
    coefficient=float(np.sum(gradient*derivative))
    if not np.isfinite(gradient).all() or not np.isfinite(coefficient):raise ValueError('Response overflow')
    return dict(k=forward['k'],reactivity=1.-1./forward['k'],
                temperature_coefficient_per_k=coefficient,temperature_coefficient_pcm_per_k=coefficient*1e5,
                absorption_gradient_m=gradient.tolist(),forward_flux=forward['flux'],adjoint_flux=psi.reshape(model.g,model.n).tolist(),
                forward_relative_residual=forward['relative_residual'],adjoint_relative_residual=residual,
                adjoint_iterations=iteration,derivative_provenance=derivative_provenance,
                scope='Absorption derivative only; all other coefficients and geometry fixed',
                source=signature(),validated=False,plant_referenced=False)
