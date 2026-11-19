"""Scalar transport: independently paired convolution and padded Fourier FFT."""
import numpy as np
from atomic_model_station_advection.kernel import _radius, _key


def _validate(radius, alpha, velocity, temperature, radial):
    r = _radius(radius, alpha)
    maps = (velocity, temperature, radial)
    if any(type(a) is not dict for a in maps) or not 1 <= len(velocity) <= 25:
        raise ValueError('Bounded scalar transport dictionaries required')
    if set(velocity) != set(temperature) or set(temperature) != set(radial):
        raise ValueError('Matching signed velocity and temperature modes required')
    for key in velocity:
        _key(key, 2)
        opposite = (-key[0], -key[1])
        if opposite not in velocity:
            raise ValueError('Hermitian scalar transport spectrum required')
        for mapping, shape in zip(maps, ((3, len(r)), (len(r),), (len(r),))):
            a = np.asarray(mapping[key])
            if a.shape != shape or not np.iscomplexobj(a) or not np.all(np.isfinite(a)):
                raise ValueError('Finite complex scalar transport arrays required')
            if not np.allclose(a, np.asarray(mapping[opposite]).conj(), rtol=1e-11, atol=1e-13):
                raise ValueError('Scalar transport conjugate symmetry violated')
    return r


def scalar_products(radius, alpha, velocity, temperature, radial, *, method='fft'):
    r = _validate(radius, alpha, velocity, temperature, radial)
    if method == 'direct':
        output = {}
        for p, u in velocity.items():
            for q, theta in temperature.items():
                key = (p[0]+q[0], p[1]+q[1])
                value = u[0]*radial[q] + 1j*q[1]*u[1]/r*theta + 1j*alpha*q[0]*u[2]*theta
                output[key] = output.get(key, np.zeros(len(r), complex)) + value
        result = {key: output.get(key, np.zeros(len(r), complex)) for key in sorted(velocity)}
        if any(not np.all(np.isfinite(a)) for a in result.values()):
            raise ValueError('Nonfinite direct scalar transport')
        return result
    if method != 'fft':
        raise ValueError('Scalar transport method must be fft or direct')
    nz = 3*max(abs(k) for k, m in velocity)+1
    nt = 3*max(abs(m) for k, m in velocity)+1
    def physical(mapping, vector=False, multiplier=lambda k, m: 1):
        a = np.zeros(((3,) if vector else ()) + (len(r), nz, nt), complex)
        for (k, m), value in mapping.items():
            a[..., k % nz, m % nt] = value*multiplier(k, m)
        return np.fft.ifft2(a, axes=(-2, -1))*(nz*nt)
    u = physical(velocity, vector=True)
    adv = (u[0]*physical(radial) + u[1]/r[:, None, None]*physical(temperature, multiplier=lambda k,m: 1j*m)
           + u[2]*physical(temperature, multiplier=lambda k,m: 1j*alpha*k))
    transformed = np.fft.fft2(adv, axes=(-2, -1))/(nz*nt)
    result = {key: transformed[:, key[0] % nz, key[1] % nt].copy() for key in sorted(velocity)}
    if any(not np.all(np.isfinite(a)) for a in result.values()):
        raise ValueError('Nonfinite scalar transport')
    return result
