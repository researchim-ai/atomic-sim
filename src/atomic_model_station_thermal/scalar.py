"""Regular scalar Fourier spaces with homogeneous isothermal wall data."""
from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
from numpy.polynomial import Legendre
from numpy.polynomial.legendre import leggauss
from scipy.linalg import cholesky, solve_triangular

from atomic_model_multiphysics.provenance import sha
from atomic_model_station_regular.model import _finite_real


def scalar_basis(radius, m, k, alpha, order):
    r = np.asarray(radius, dtype=float)
    if (r.ndim != 1 or not 1 <= len(r) <= 4096 or not np.all(np.isfinite(r))
            or np.any(r < 0) or np.any(r > 1)):
        raise ValueError('Finite scalar radii in [0,1] required')
    values, radial, laplacian = [], [], []
    wall = Legendre([.5, -.5], domain=[0, 1])
    for j in range(order):
        p = wall * Legendre.basis(j, domain=[0, 1])
        s = r*r
        values.append(r**m * p(s))
        radial.append((m*r**(m-1)*p(s) if m else np.zeros_like(r))
                      + 2*r**(m+1)*p.deriv()(s))
        laplacian.append(r**m * (4*(m+1)*p.deriv()(s)
                         + 4*s*p.deriv(2)(s) - (alpha*k)**2*p(s)))
    return tuple(np.asarray(a, dtype=complex).T for a in (values, radial, laplacian))


@dataclass(frozen=True)
class ScalarMode:
    k: int
    m: int
    alpha: float
    order: int
    quadrature: int
    radius: np.ndarray
    weights: np.ndarray
    transform: np.ndarray
    basis: np.ndarray
    radial: np.ndarray
    mass: np.ndarray
    stiffness: np.ndarray
    fingerprint: str

    @property
    def dimension(self):
        return self.order

    def coefficients(self, value):
        q = np.asarray(value, dtype=complex)
        if q.shape != (self.order,) or not np.all(np.isfinite(q)):
            raise ValueError('Complete finite temperature coefficients required')
        return q

    def jets(self, value, radius):
        q = self.transform @ self.coefficients(value)
        return tuple(a @ q for a in scalar_basis(radius, self.m, self.k, self.alpha, self.order))

    def evaluate(self, value, radius):
        return self.jets(value, radius)[0]

    def fit(self, values):
        a = np.asarray(values, dtype=complex)
        if a.shape != self.radius.shape or not np.all(np.isfinite(a)):
            raise ValueError('Finite scalar values on model quadrature required')
        return np.linalg.solve(self.mass, self.basis.conj().T @ (self.weights*a))


def scalar_mode(k, m, alpha, *, order=8, quadrature=96):
    if type(k) is not int or abs(k) > 2 or type(m) is not int or not 0 <= m <= 2:
        raise ValueError('Scalar Fourier domain |k|<=2, m=0..2 required')
    if (type(order) is not int or not 2 <= order <= 16 or type(quadrature) is not int
            or not max(64, 2*order+m+4) <= quadrature <= 256):
        raise ValueError('Resolved scalar order and quadrature required')
    _finite_real(alpha, 'scalar alpha', .01, 10.)
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
    digest = hashlib.sha256(sha(Path(__file__)).encode())
    digest.update(repr((k, m, float(alpha), order, quadrature)).encode())
    for a in (r, w, transform, basis, radial, mass, stiffness):
        digest.update(a.tobytes())
        a.setflags(write=False)
    return ScalarMode(k, m, float(alpha), order, quadrature, r, w, transform,
                      basis, radial, mass, stiffness, digest.hexdigest())
