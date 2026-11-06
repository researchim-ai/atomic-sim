"""Fixed outer-wave HLLC and exact local forward-mode derivatives.

Outer bounds +/-a are checked by the caller's liquid/Mach guard. The selected
star state is checked with the actual IF97 EOS. No automatic Rusanov fallback.
Contact-branch derivatives at S*=0 are one deterministic generalized choice.
"""

import numpy as np

from core.status import DomainError


class Dual:
    def __init__(self, value, gradient):
        self.value = float(value)
        self.gradient = np.asarray(gradient, dtype=float)

    def pair(self, other):
        return other if isinstance(other, Dual) else Dual(other, np.zeros_like(self.gradient))

    def __add__(self, other):
        other = self.pair(other)
        return Dual(self.value + other.value, self.gradient + other.gradient)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.value, -self.gradient)

    def __sub__(self, other):
        return self + -self.pair(other)

    def __rsub__(self, other):
        return self.pair(other) + -self

    def __mul__(self, other):
        other = self.pair(other)
        return Dual(
            self.value * other.value, self.gradient * other.value + other.gradient * self.value
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.pair(other)
        if other.value == 0:
            raise DomainError("Singular HLLC wave denominator")
        return Dual(
            self.value / other.value,
            (self.gradient - self.value / other.value * other.gradient) / other.value,
        )

    def __rtruediv__(self, other):
        return self.pair(other) / self


def interface(model, left, right, with_jacobian=False):
    n = 6 if with_jacobian else 0

    def state(q, offset):
        thermo, u, _, _ = model.cell(q)
        raw, jac, _ = model.flux(q, True) if with_jacobian else (model.flux(q), None, None)
        variables = []
        for i, value in enumerate(q):
            gradient = np.zeros(n)
            if n:
                gradient[offset + i] = 1
            variables.append(Dual(value, gradient))
        mass, momentum, energy = variables
        pressure_gradient = np.zeros(n)
        if n:
            pressure_gradient[offset : offset + 3] = (
                jac[1] - np.array([-u * u / model.dx, 2 * u / model.dx, 0.0])
            ) / model.config.area_m2
        pressure = Dual(thermo.p_pa, pressure_gradient)
        flux = []
        for i, value in enumerate(raw):
            gradient = np.zeros(n)
            if n:
                gradient[offset : offset + 3] = jac[i]
            flux.append(Dual(value, gradient))
        return mass, momentum, energy, momentum / mass, pressure, flux

    ml, jl, el, ul, pl, fl = state(left, 0)
    mr, jr, er, ur, pr, fr = state(right, 3)
    sl = -model.config.wave_speed_bound_m_s
    sr = model.config.wave_speed_bound_m_s
    # Density factors 1/V cancel in S*, except pressure, which needs p*V.
    contact = ((pr - pl) * model.volume + ml * ul * (sl - ul) - mr * ur * (sr - ur)) / (
        ml * (sl - ul) - mr * (sr - ur)
    )
    if not sl < contact.value < sr:
        raise DomainError("HLLC contact escaped outer waves")
    if contact.value >= 0:
        mass, momentum, energy, velocity, pressure, flux, speed = ml, jl, el, ul, pl, fl, sl
    else:
        mass, momentum, energy, velocity, pressure, flux, speed = mr, jr, er, ur, pr, fr, sr
    factor = (speed - velocity) / (speed - contact)
    star = [
        factor * mass,
        factor * mass * contact,
        factor
        * (
            energy
            + (contact - velocity) * (mass * contact + pressure * model.volume / (speed - velocity))
        ),
    ]
    model.cell(np.array([v.value for v in star]))
    selected = [
        flux[i] + speed / model.dx * (star[i] - original)
        for i, original in enumerate((mass, momentum, energy))
    ]
    pressure_star = pressure + mass / model.volume * (speed - velocity) * (contact - velocity)
    values = np.array([v.value for v in selected])
    gradient = np.array([v.gradient for v in selected])
    if not np.isfinite(values).all() or not np.isfinite(gradient).all():
        raise DomainError("Nonfinite HLLC flux")
    return dict(
        flux=values,
        left_gradient=gradient[:, :3] if n else None,
        right_gradient=gradient[:, 3:] if n else None,
        contact_velocity_m_s=contact.value,
        pressure_star_pa=pressure_star.value,
    )
