"""Separate high-order scalar factory; legacy fingerprints are preserved."""
from pathlib import Path
import hashlib
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import cholesky, solve_triangular
from atomic_model_multiphysics.provenance import sha
from atomic_model_station_regular.model import _finite_real
from atomic_model_station_thermal import scalar as prior
from atomic_model_station_thermal.scalar import ScalarMode, scalar_basis


def resolved_scalar_mode(k, m, alpha, *, order=8, quadrature=96):
    if type(k) is not int or abs(k) > 5 or type(m) is not int or not 0 <= m <= 5:
        raise ValueError('Scalar Fourier domain |k|<=5, m=0..5 required')
    if (type(order) is not int or not 2 <= order <= 48 or type(quadrature) is not int
            or not max(64, 2*order+m+4) <= quadrature <= 256):
        raise ValueError('Resolved scalar order and quadrature required')
    _finite_real(alpha, 'scalar alpha', .01, 10.)
    if abs(k)<=2 and m<=2 and order<=16:
        return prior.scalar_mode(k,m,alpha,order=order,quadrature=quadrature)
    x, w = leggauss(quadrature)
    r, w = np.sqrt((x+1)/2), w/4
    raw, dr, _ = scalar_basis(r, m, k, alpha, order)
    gram = raw.conj().T @ (w[:, None]*raw)
    condition = np.linalg.cond(gram)
    if not np.isfinite(condition) or condition > 1e10:
        raise ValueError('Ill-conditioned scalar space')
    lower = cholesky(gram, lower=True)
    transform = solve_triangular(lower.conj().T, np.eye(order), lower=False)
    basis, radial = raw @ transform, dr @ transform
    mass = basis.conj().T @ (w[:, None]*basis)
    stiffness = (radial.conj().T @ (w[:, None]*radial)
                 + basis.conj().T @ ((w*(m*m/r**2+(alpha*k)**2))[:, None]*basis))
    mass = (mass+mass.conj().T)/2
    stiffness = (stiffness+stiffness.conj().T)/2
    if np.min(np.linalg.eigvalsh(stiffness)) <= 0:
        raise ValueError('Scalar diffusion is not coercive')
    digest = hashlib.sha256((sha(Path(__file__))+sha(prior.__file__)).encode())
    digest.update(repr((k, m, float(alpha), order, quadrature)).encode())
    for a in (r, w, transform, basis, radial, mass, stiffness):
        digest.update(a.tobytes())
        a.setflags(write=False)
    return ScalarMode(k, m, float(alpha), order, quadrature, r, w, transform,
                      basis, radial, mass, stiffness, digest.hexdigest())
