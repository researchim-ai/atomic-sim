"""Compatible staggered kinetic-energy accounting for a uniform liquid pipe.

This closes the reduced lumped momentum model's discrete total energy. It does
not supply the missing momentum convection/acoustic/entropy qualifications of
full compressible Navier-Stokes. Mass-based donor density makes the chain rule
independent of inverse-EOS residuals.
"""

import numpy as np

from core.status import DomainError


def allocation(cells):
    if type(cells) is not int or cells < 1:
        raise ValueError("Positive integer cell count required")
    w = np.zeros((cells, cells + 1))
    w[0, 0] = 1.0
    w[-1, -1] = 1.0
    for j in range(1, cells):
        w[j - 1, j] = w[j, j] = 0.5
    return w


def kinematics(model, state):
    c = model.config
    n = c.cells
    f = np.asarray(state[2 * n : 3 * n + 1])
    rho = np.empty(n + 1)
    donors = []
    for j, flow in enumerate(f):
        donor = j - 1 if flow >= 0 else j
        donors.append(donor)
        if 0 <= donor < n:
            rho[j] = state[2 * donor] / (c.total_volume_m3 / n)
        else:
            rho[j] = model.network.boundaries["left" if donor < 0 else "right"].density_kg_m3
    if not np.isfinite(rho).all() or np.any(rho <= 0):
        raise DomainError("Nonpositive mechanical donor density")
    junctions = list(model.network.junctions.values())
    a = np.array([j.area_m2 for j in junctions])
    length = np.array([j.length_m for j in junctions])
    loss = np.array([j.loss_coefficient for j in junctions])
    velocity = f / (rho * a)
    alpha = length / (a * rho)
    kinetic = 0.5 * alpha * f**2
    return dict(
        flows=f,
        rho=rho,
        donors=donors,
        area=a,
        length=length,
        loss=loss,
        alpha=alpha,
        velocity=velocity,
        kinetic=kinetic,
        kinetic_flux=0.5 * f * velocity**2,
    )


def mechanical_rates(model, state):
    n = model.config.cells
    k = kinematics(model, state)
    f = k["flows"]
    rho = k["rho"]
    a = k["area"]
    pressures = np.r_[
        model.network.boundaries["left"].pressure_pa,
        [v.pressure_pa for v in model.network.volumes.values()],
        model.network.boundaries["right"].pressure_pa,
    ]
    dp = pressures[:-1] - pressures[1:]
    mass_rate = f[:-1] - f[1:]
    volume = model.config.total_volume_m3 / n
    rho_rate = np.array([mass_rate[d] / volume if 0 <= d < n else 0.0 for d in k["donors"]])
    force_loss = k["loss"] * f * abs(f) / (2 * rho * a**2)
    acceleration = a / k["length"] * (dp - force_loss)
    pressure_work = dp * f / rho
    dissipation = k["loss"] * abs(f) ** 3 / (2 * rho**2 * a**2)
    density_work = -k["kinetic"] * rho_rate / rho
    kinetic_rate = k["alpha"] * f * acceleration + density_work
    phi = k["kinetic_flux"]
    w = allocation(n)
    correction = phi[:-1] - phi[1:] - w @ kinetic_rate
    return dict(
        k,
        acceleration=acceleration,
        pressure_difference=dp,
        rho_rate=rho_rate,
        pressure_work=pressure_work,
        dissipation=dissipation,
        density_work=density_work,
        kinetic_rate=kinetic_rate,
        internal_correction=correction,
        identity_residual_w=float(sum(kinetic_rate - pressure_work + dissipation - density_work)),
    )


def mechanical_jacobian_correction(model, state, jac, therm):
    """Add the exact local chain rule to the legacy thermal sparse Jacobian."""
    n = model.config.cells
    size = len(state)
    r = mechanical_rates(model, state)
    f = r["flows"]
    rho = r["rho"]
    gk = np.zeros((n + 1, size))
    gp = gk.copy()
    gd = gk.copy()
    gw = gk.copy()
    gphi = gk.copy()
    volume = model.config.total_volume_m3 / n
    for j, donor in enumerate(r["donors"]):
        ef = np.zeros(size)
        ef[2 * n + j] = 1.0
        gr = np.zeros(size)
        gb = np.zeros(size)
        gdp = np.zeros(size)
        if j > 0:
            gdp[2 * (j - 1) : 2 * j] += therm[j - 1].pressure
        if j < n:
            gdp[2 * j : 2 * j + 2] -= therm[j].pressure
        if 0 <= donor < n:
            gr[2 * donor] = 1 / volume
            gb[2 * n + donor] = 1 / volume
            gb[2 * n + donor + 1] = -1 / volume
        k = r["kinetic"][j]
        alpha = r["alpha"][j]
        p = r["pressure_difference"][j]
        a = r["area"][j]
        b = r["rho_rate"][j]
        gk[j] = alpha * f[j] * ef - k / rho[j] * gr
        gp[j] = f[j] / rho[j] * gdp + p / rho[j] * ef - p * f[j] / rho[j] ** 2 * gr
        gd[j] = (
            3 * r["loss"][j] * f[j] * abs(f[j]) / (2 * rho[j] ** 2 * a**2) * ef
            - 2 * r["dissipation"][j] / rho[j] * gr
        )
        gw[j] = -b / rho[j] * gk[j] - k / rho[j] * gb + k * b / rho[j] ** 2 * gr
        gphi[j] = (
            3 * f[j] ** 2 / (2 * rho[j] ** 2 * a**2) * ef - f[j] ** 3 / (rho[j] ** 3 * a**2) * gr
        )
    gdot = gp - gd + gw
    correction = gphi[:-1] - gphi[1:] - allocation(n) @ gdot
    for i in range(n):
        jac[2 * i + 1, :] += correction[i]
    jac[4 * n + 1, :] = sum(gp)
    jac[4 * n + 2, :] = sum(gd)
    jac[4 * n + 3, :] = sum(gw)
    jac[-1, :] += gphi[0] - gphi[-1]
    return jac


def kinetic_gradient_hessian(model, state):
    """Inventory derivatives for independent differential balance tests."""
    r = kinematics(model, state)
    n = model.config.cells
    size = len(state)
    gradient = np.zeros(size)
    hessian = np.zeros((size, size))
    for j, donor in enumerate(r["donors"]):
        col = 2 * n + j
        gradient[col] += r["alpha"][j] * r["flows"][j]
        hessian[col, col] += r["alpha"][j]
        if 0 <= donor < n:
            mcol = 2 * donor
            mass = state[mcol]
            k = r["kinetic"][j]
            gradient[mcol] -= k / mass
            hessian[mcol, mcol] += 2 * k / mass**2
            cross = -r["alpha"][j] * r["flows"][j] / mass
            hessian[col, mcol] += cross
            hessian[mcol, col] += cross
    return gradient, hessian
