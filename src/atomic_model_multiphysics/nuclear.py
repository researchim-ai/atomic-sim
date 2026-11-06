"""Evaluated MGXS admission into the immutable slab model, no group reversal guess."""

import numpy as np

from atomic_model_spatial import Model

from .provenance import unseal


def specification(seed=41, temperature_k=294.0, density_kg_m3=3000.0):
    return dict(
        seed=seed,
        temperature_k=temperature_k,
        density_kg_m3=density_kg_m3,
        nuclides={"U235": 0.0004, "U238": 0.0096, "O16": 0.34, "H1": 0.65},
        width_m=1.0,
        energy_bounds_ev_ascending=[1e-5, 0.625, 2e7],
        particles=4000,
        batches=45,
        inactive=10,
    )


def model_from_mgxs(envelope, cells=16):
    p = unseal(envelope)
    if type(cells) is not int or not 1 <= cells <= 512:
        raise ValueError("MGXS model cell budget")
    if p.get("source_family") != "evaluated":
        raise ValueError("Numerical coefficients cannot become evaluated nuclear data")
    expected_scores = {
        "absorption",
        "nu-fission",
        "fission",
        "scatter matrix",
        "chi",
        "inverse-velocity",
        "diffusion-coefficient",
        "kappa-fission",
    }
    if set(p["means"]) != expected_scores or set(p["standard_errors"]) != expected_scores:
        raise ValueError("Incomplete MGXS score/uncertainty inventory")
    for key, values in p["standard_errors"].items():
        a = np.asarray(values, float)
        if (
            not np.isfinite(a).all()
            or np.any(a < 0)
            or a.shape != np.asarray(p["means"][key]).shape
        ):
            raise ValueError("Invalid MGXS statistical uncertainty")
    if (
        p["nuclear_data"]["library"] != "ENDF/B-VIII.0"
        or type(p["k_std"]) not in (int, float)
        or not np.isfinite(p["k_std"])
        or type(p["k_mean"]) not in (int, float)
        or not np.isfinite(p["k_mean"])
        or p["k_mean"] <= 0
        or p["k_std"] <= 0
        or p["k_std"] > 0.01
    ):
        raise ValueError("Unqualified evaluated library or Monte Carlo uncertainty")
    axes = p["axes"]
    if (
        axes["macro_unit"] != "1/cm"
        or axes.get("diffusion_unit") != "cm"
        or axes.get("inverse_velocity_unit") != "s/cm"
        or axes.get("kappa_fission_unit") != "eV/cm"
        or axes["scatter"] != "incoming,outgoing"
        or axes["groups"] != "fast to thermal, OpenMC group indices increasing"
    ):
        raise ValueError("MGXS axis/unit mismatch")
    m = p["means"]
    g = len(m["absorption"])

    def tiled(key, scale=1):
        a = np.asarray(m[key], float)
        if a.shape != (g,) or not np.isfinite(a).all():
            raise ValueError("Invalid MGXS " + key)
        return np.repeat((a * scale)[:, None], cells, axis=1).tolist()

    scattering = np.asarray(m["scatter matrix"], float).T * 100
    if scattering.shape != (g, g) or not np.isfinite(scattering).all() or np.any(scattering < 0):
        raise ValueError("Scatter matrix axis mismatch")
    np.fill_diagonal(scattering, 0)
    inverse = np.asarray(m["inverse-velocity"], float) * 100
    if np.any(inverse <= 0):
        raise ValueError("Nonpositive inverse velocity")
    chi = np.asarray(m["chi"], float)
    if abs(chi.sum() - 1) > 1e-8:
        raise ValueError("MGXS spectrum normalization differs")
    config = dict(
        faces_m=np.linspace(0, p["specification"]["width_m"], cells + 1).tolist(),
        diffusion_m=tiled("diffusion-coefficient", 0.01),
        absorption_per_m=tiled("absorption", 100),
        nu_fission_per_m=tiled("nu-fission", 100),
        scatter_per_m=np.repeat(scattering[:, :, None], cells, axis=2).tolist(),
        speed_m_s=np.repeat((1 / inverse)[:, None], cells, axis=1).tolist(),
        chi_prompt=np.repeat(chi[:, None], cells, axis=1).tolist(),
        chi_delayed=np.repeat(chi[:, None], cells, axis=1).tolist(),
        beta=[[0.0] * cells],
        decay_per_s=[1.0],
        boundaries=["reflecting", "reflecting"],
        provenance="ENDF/B-VIII.0 homogeneous research mixture; exact source envelope "
        + envelope["sha256"]
        + "; beta=0 static comparison, not a plant kinetics data set",
    )
    return Model(config)
