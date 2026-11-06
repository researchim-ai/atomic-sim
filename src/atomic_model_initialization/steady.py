"""Solve dimensional node/edge/thermal balances without transient warm-up."""

import hashlib
import platform
from copy import deepcopy
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares

from atomic_model_workspace.common import digest
from scripts.buoyant_inertial_graph import BuoyantInertialGraph, BuoyantSystem, signature

CRITERIA = dict(
    mass_kg_s=1e-10,
    momentum_pa=1e-5,
    thermal_w=1e-5,
    relative_jacobian_rank_tolerance=1e-9,
    max_evaluations=2000,
)


def identity():
    return dict(
        parent=signature(),
        sources={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
    )


class Equations:
    def __init__(self, request):
        self.request = deepcopy(request)
        self.system = s = BuoyantSystem(self.request)
        self.m, self.n = len(s.edges), len(s.names)
        if len(s.fixed) != 1:
            raise ValueError("Exactly one explicit pressure gauge required")
        self.free = np.array(s.free, dtype=int)
        gauge = next(iter(s.fixed))
        self.pbase = s.initial_pressures[s.names[gauge]] - s.rho * s.gravity * (
            s.elevation - s.elevation[gauge]
        )
        self.origin = np.r_[
            np.zeros(self.m), self.pbase[self.free], np.full(self.n, s.buoyancy_tref)
        ]
        self.scale = np.r_[np.full(self.m, 0.1), np.full(self.n - 1, 1000.0), np.full(self.n, 10.0)]
        self.tolerances = np.r_[
            np.full(self.n - 1, CRITERIA["mass_kg_s"]),
            np.full(self.m, CRITERIA["momentum_pa"]),
            np.full(self.n, CRITERIA["thermal_w"]),
        ]
        self.units = ["kg/s"] * (self.n - 1) + ["Pa"] * self.m + ["W"] * self.n
        self.names = (
            ["mass/" + s.names[i] for i in self.free]
            + ["momentum/" + e for e in s.edges]
            + ["thermal/" + n for n in s.names]
        )
        contrast = min(0.1 * s.buoyancy_tref, s.density_limit / s.beta if s.beta else np.inf)
        low = max(s.tmin, s.buoyancy_tref - contrast)
        high = min(s.tmax, s.buoyancy_tref + contrast)
        self.lower = (
            np.r_[np.full(self.m, -s.qmax), np.full(self.n - 1, s.pmin), np.full(self.n, low)]
            - self.origin
        ) / self.scale
        self.upper = (
            np.r_[np.full(self.m, s.qmax), np.full(self.n - 1, s.pmax), np.full(self.n, high)]
            - self.origin
        ) / self.scale
        self.calls = 0

    def unpack(self, scaled):
        x = self.origin + self.scale * np.asarray(scaled)
        q = x[: self.m]
        p = self.pbase.copy()
        p[self.free] = x[self.m : self.m + self.n - 1]
        T = x[-self.n :]
        return q, p, T

    def raw(self, scaled):
        s = self.system
        q, p, T = self.unpack(scaled)
        tr, _, loss, _ = s.thermal_terms(q, T, p)
        force, _ = s.buoyancy(T)
        mass = s.A @ q
        momentum = s.A.T @ (p + s.rho * s.gravity * s.elevation) + loss - force
        thermal = s.capacity * tr
        return np.r_[mass[self.free], momentum, thermal]

    def fun(self, scaled):
        self.calls += 1
        if self.calls > 200000:
            raise ValueError("Initialization residual work budget exhausted")
        residual = self.raw(scaled)
        if not np.isfinite(residual).all():
            raise ValueError("Nonfinite initialization residual")
        return residual / self.tolerances

    def jacobian(self, scaled):
        # Independent final certificate: central differences when admissible,
        # otherwise an explicit one-sided derivative; no projection of x.
        cols = []
        for i in range(len(scaled)):
            h = 1e-5 * max(1.0, abs(scaled[i]))
            a, b = scaled.copy(), scaled.copy()
            a[i] = max(self.lower[i], scaled[i] - h)
            b[i] = min(self.upper[i], scaled[i] + h)
            if b[i] <= a[i]:
                raise ValueError("No derivative interval")
            cols.append((self.raw(b) - self.raw(a)) / (b[i] - a[i]) / self.tolerances)
        return np.array(cols).T


def initialize(request, *, flow_guess_kg_s=0.13, temperature_guess_k=310.0):
    if not np.isfinite([flow_guess_kg_s, temperature_guess_k]).all():
        raise ValueError("Finite explicit initial guesses required")
    eq = Equations(request)
    x0 = (
        np.r_[np.full(eq.m, flow_guess_kg_s), eq.pbase[eq.free], np.full(eq.n, temperature_guess_k)]
        - eq.origin
    ) / eq.scale
    if np.any(x0 < eq.lower) or np.any(x0 > eq.upper):
        raise ValueError("Initial guess outside declared bounds; no clipping")
    result = least_squares(
        eq.fun,
        x0,
        bounds=(eq.lower, eq.upper),
        jac="3-point",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-10,
        max_nfev=CRITERIA["max_evaluations"],
    )
    residual = eq.raw(result.x)
    jac = eq.jacobian(result.x)
    singular = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * CRITERIA["relative_jacobian_rank_tolerance"]))
    if not result.success or np.max(abs(residual) / eq.tolerances) > 1:
        raise ValueError("Steady balances not converged within dimensional tolerances")
    if rank != len(result.x):
        raise ValueError(
            "Initialization not locally unique; explicit gauge/thermal constraint missing"
        )
    q, p, T = eq.unpack(result.x)
    # Continuity at the gauge is redundant only for this closed graph; verify it too.
    if np.max(abs(eq.system.A @ q)) > CRITERIA["mass_kg_s"]:
        raise ValueError("Gauge-node mass balance failed")
    seeded = deepcopy(request)
    seeded["base"]["hydraulic"]["initial_flows_kg_s"] = dict(zip(eq.system.edges, map(float, q)))
    for name, temp in zip(eq.system.names, T):
        seeded["base"]["thermal"]["nodes"][name]["initial_temperature_k"] = float(temp)
    runtime = BuoyantInertialGraph(seeded)
    point = runtime.snapshot()["state"]
    report = dict(
        schema_version=1,
        source=identity(),
        request=deepcopy(request),
        initialized_request=seeded,
        criteria=deepcopy(CRITERIA),
        solver_success=bool(result.success),
        evaluations=eq.calls,
        jacobian_rank=rank,
        unknown_count=len(result.x),
        scaled_jacobian_singular_values=singular.tolist(),
        local_uniqueness_only=True,
        multiple_stationary_branches_not_excluded=True,
        pressure_gauge=dict(eq.system.initial_pressures),
        residuals=[
            dict(name=name, unit=unit, value=float(value), tolerance=float(tol))
            for name, unit, value, tol in zip(eq.names, eq.units, residual, eq.tolerances)
        ],
        max_mass_residual_kg_s=float(np.max(abs(eq.system.A @ q))),
        max_momentum_residual_pa=float(np.max(abs(residual[eq.n - 1 : eq.n - 1 + eq.m]))),
        max_thermal_residual_w=float(np.max(abs(residual[-eq.n :]))),
        state=point,
        initialization_time_s=0.0,
        hidden_warmup_steps=0,
        validated=False,
        plant_referenced=False,
    )
    return dict(payload=report, sha256=digest(report))


def audit_certificate(saved):
    if set(saved) != {"payload", "sha256"} or digest(saved["payload"]) != saved["sha256"]:
        raise ValueError("Invalid initialization envelope")
    p = saved["payload"]
    if (
        p["source"] != identity()
        or p["criteria"] != CRITERIA
        or p["validated"]
        or p["plant_referenced"]
    ):
        raise ValueError("Initialization source/acceptance mismatch")
    eq = Equations(p["request"])
    seeded = p["initialized_request"]
    q = np.array([seeded["base"]["hydraulic"]["initial_flows_kg_s"][e] for e in eq.system.edges])
    T = np.array(
        [
            seeded["base"]["thermal"]["nodes"][name]["initial_temperature_k"]
            for name in eq.system.names
        ]
    )
    runtime = BuoyantInertialGraph(seeded)
    point = runtime.snapshot()["state"]
    pressure = np.array([point["pressures_pa"][name] for name in eq.system.names])
    x = (np.r_[q, pressure[eq.free], T] - eq.origin) / eq.scale
    if np.any(x < eq.lower) or np.any(x > eq.upper):
        raise ValueError("Certified state outside original domain")
    raw = eq.raw(x)
    if np.max(abs(raw) / eq.tolerances) > 1:
        raise ValueError("Certificate fails original physical residual")
    singular = np.linalg.svd(eq.jacobian(x), compute_uv=False)
    rank = int(np.sum(singular > singular[0] * CRITERIA["relative_jacobian_rank_tolerance"]))
    if rank != len(x) or p["jacobian_rank"] != rank or p["unknown_count"] != len(x):
        raise ValueError("Certificate fails rank check")
    # Ensure only initial stocks changed; no substituted heat/law/gauge/topology.
    expected = deepcopy(p["request"])
    expected["base"]["hydraulic"]["initial_flows_kg_s"] = deepcopy(
        seeded["base"]["hydraulic"]["initial_flows_kg_s"]
    )
    for name in eq.system.names:
        expected["base"]["thermal"]["nodes"][name]["initial_temperature_k"] = seeded["base"][
            "thermal"
        ]["nodes"][name]["initial_temperature_k"]
    if digest(expected) != digest(seeded) or digest(point) != digest(p["state"]):
        raise ValueError("Certificate changed model coefficients or recorded state")
    if p["hidden_warmup_steps"] != 0 or p["initialization_time_s"] != 0:
        raise ValueError("Unreported initialization history")
    expected_names = eq.names
    if len(p["residuals"]) != len(raw):
        raise ValueError("Certificate residual inventory mismatch")
    for index, row in enumerate(p["residuals"]):
        if (
            set(row) != {"name", "unit", "value", "tolerance"}
            or row["name"] != expected_names[index]
            or row["unit"] != eq.units[index]
            or row["tolerance"] != eq.tolerances[index]
            or not np.isfinite(row["value"])
            or abs(row["value"] - raw[index]) > 0.01 * eq.tolerances[index]
        ):
            raise ValueError("Certificate dimensional residual report changed")
    for key, value, tolerance in [
        ("max_mass_residual_kg_s", np.max(abs(eq.system.A @ q)), CRITERIA["mass_kg_s"]),
        (
            "max_momentum_residual_pa",
            np.max(abs(raw[eq.n - 1 : eq.n - 1 + eq.m])),
            CRITERIA["momentum_pa"],
        ),
        ("max_thermal_residual_w", np.max(abs(raw[-eq.n :])), CRITERIA["thermal_w"]),
    ]:
        if not np.isfinite(p[key]) or p[key] < 0 or abs(p[key] - value) > 0.01 * tolerance:
            raise ValueError("Certificate maxima changed")
    if (
        p["solver_success"] is not True
        or p["local_uniqueness_only"] is not True
        or p["multiple_stationary_branches_not_excluded"] is not True
        or p["pressure_gauge"] != eq.system.initial_pressures
        or type(p["evaluations"]) is not int
        or not 1 <= p["evaluations"] <= 200000
        or np.shape(p["scaled_jacobian_singular_values"]) != np.shape(singular)
        or not np.allclose(p["scaled_jacobian_singular_values"], singular, rtol=1e-3, atol=1e-5)
    ):
        raise ValueError("Certificate solver/rank/gauge metadata changed")
    return runtime
