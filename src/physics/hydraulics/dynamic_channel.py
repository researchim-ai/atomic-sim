"""Single-phase pressure-driven IF97 channel with lumped wall and signed flows.

Low-Mach internal-energy formulation of FlowNetwork: kinetic energy storage,
acoustic accuracy and dissipation heating are not qualified by this component.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from core.continuation import ImplicitContinuation
from core.solvers import integrate
from core.status import DomainError, ModelPassport, ModelStatus
from physics.hydraulics.isobaric import _finite
from physics.hydraulics.network import BoundaryState, FlowNetwork, Junction, Volume
from physics.hydraulics.reconstruction import face_values
from physics.materials.iapws import (
    P_MIN_MPA,
    P_REGION13_MPA,
    T_MIN,
    T_REGION1_MAX,
    _region1_raw,
    properties_pt,
    properties_rho_u,
    saturation_temperature_k,
)


@lru_cache(maxsize=4096)
def _liquid_state(density, energy, pressure_seed, temperature_seed):
    # Fixed seeds and exact keys: identical RHS states cannot depend on the
    # order of Newton/Jacobian probes. No rounded thermo-state lookup.
    state = properties_rho_u(density, energy, pressure_seed, temperature_seed)
    if state.region != 1:
        raise DomainError("Boiling/steam is outside pressure-driven channel passport")
    return state


@lru_cache(maxsize=4096)
def _sound_speed(pressure, temperature):
    raw = _region1_raw(temperature, pressure / 1e6)
    # ds=cp/T dT - (dv/dT)_p dp; dp in Pa, cp in J/(kg K).
    dv_dp_s = raw["dv_dp_mpa"] / 1e6 + temperature * raw["dv_dt"] ** 2 / (1000 * raw["cp"])
    if not np.isfinite(dv_dp_s) or dv_dp_s >= 0:
        raise DomainError("Nonpositive isentropic compressibility")
    return float(np.sqrt(-(raw["v"] ** 2) / dv_dp_s))


@lru_cache(maxsize=4096)
def _face_liquid_bounds(pressure):
    p = pressure / 1e6
    if not np.isfinite(p) or not P_MIN_MPA < p <= 100.0:
        raise DomainError("Face pressure outside IF97 liquid range")
    subcritical = p <= P_REGION13_MPA
    upper_t = saturation_temperature_k(p) if subcritical else T_REGION1_MAX
    # dh/dT|p=cp>0 in the admitted liquid domain. Testing the exact
    # enthalpy interval needs no inverse EOS or face temperature solve.
    return (
        _region1_raw(T_MIN, p)["h"] * 1000.0,
        _region1_raw(upper_t, p)["h"] * 1000.0,
        subcritical,
    )


@lru_cache(maxsize=4096)
def _check_face_liquid(pressure, enthalpy):
    lower, upper, subcritical = _face_liquid_bounds(pressure)
    upper_ok = enthalpy < upper if subcritical else enthalpy <= upper
    if not np.isfinite(enthalpy) or enthalpy < lower or not upper_ok:
        raise DomainError("Reconstructed enthalpy left liquid IF97 region 1")


PASSPORT = ModelPassport(
    model_id="pressure-driven-wall-channel",
    version="0.1.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Однофазный канал с динамикой давления, импульсом соединений и стенкой",
    approximations=(
        "смешиваемые неподвижные объёмы IF97, донорная энтальпия",
        "жидкость без кипения, горизонтальные соединения",
        "сосредоточенный импульс, заданные местные сопротивления",
        "внутренняя энергия и энтальпийные потоки без запаса кинетической энергии",
        "постоянные C/UA; без насоса, CHF и нейтроники",
    ),
    applicability="Жидкость IF97 region 1; прямые и обратные потоки; low-Mach приближение",
    data_sources=(
        "physics.hydraulics.network.FlowNetwork",
        "https://iapws.org/technical-guidance/release/IF97-Rev",
    ),
)

COMPATIBLE_PASSPORT = ModelPassport(
    model_id="compatible-total-energy-liquid-channel",
    version="0.1.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Согласованный полный энергетический баланс редуцированного жидкого канала",
    approximations=(
        "однофазная жидкость IF97 region 1, равномерная горизонтальная труба",
        "сосредоточенный импульс без конвективного члена; не полный Euler/Navier-Stokes",
        "кинетическая энергия на двойственных объёмах, распределение к соседним ячейкам",
        "полная энтальпия на границах; диссипация и работа плотности учитываются однократно",
        "постоянные заданные C/UA/K; без насоса, CHF, энтропийной и акустической квалификации",
    ),
    applicability="IF97 region 1, Mach<=0.1, прямые и обратные потоки; экспериментальная валидация отсутствует",
    data_sources=(
        "Дискретная цепная производная K=sum L F²/(2 A rho_donor)",
        "https://iapws.org/technical-guidance/release/IF97-Rev",
    ),
)

_OPTIONAL_VECTORS = (
    "junction_lengths_m",
    "junction_loss_coefficients",
    "initial_pressure_profile_pa",
    "initial_temperature_profile_k",
    "initial_wall_temperature_profile_k",
)


DEFAULT_ABSOLUTE_TOLERANCES = dict(
    mass_kg=1e-10,
    energy_j=1e-3,
    flow_kg_s=1e-8,
    wall_temperature_k=1e-8,
    boundary_mass_kg=1e-10,
    boundary_energy_j=1e-3,
)


def validate_absolute_tolerances(options=None, *, include_mechanical=False):
    defaults = dict(DEFAULT_ABSOLUTE_TOLERANCES)
    if include_mechanical:
        defaults["mechanical_work_j"] = 1e-6
    if options is None:
        return defaults
    if not isinstance(options, dict) or set(options) - set(defaults):
        raise ValueError("Unknown absolute tolerance fields")
    return {
        key: _finite(options.get(key, value), key, positive=True) for key, value in defaults.items()
    }


@dataclass(frozen=True)
class DynamicConfig:
    cells: int
    total_volume_m3: float
    left_pressure_pa: float
    right_pressure_pa: float
    left_temperature_k: float
    right_temperature_k: float
    initial_pressure_pa: float
    initial_temperature_k: float
    area_m2: float
    junction_length_m: float
    loss_coefficient: float
    total_heat_capacity_j_k: float
    total_ua_w_k: float
    heat_w: float
    initial_flow_kg_s: float = 0.0
    initial_wall_temperature_k: float | None = None
    junction_lengths_m: tuple | None = None
    junction_loss_coefficients: tuple | None = None
    initial_pressure_profile_pa: tuple | None = None
    initial_temperature_profile_k: tuple | None = None
    initial_wall_temperature_profile_k: tuple | None = None
    enthalpy_transport: str = "donor"
    energy_formulation: str = "internal"

    def __post_init__(self):
        if type(self.cells) is not int or not 1 <= self.cells <= 32:
            raise ValueError("cells must be an integer from 1 to 32")
        for name in self.__dataclass_fields__:
            if (
                name
                in (
                    "cells",
                    "initial_wall_temperature_k",
                    "enthalpy_transport",
                    "energy_formulation",
                )
                + _OPTIONAL_VECTORS
            ):
                continue
            value = _finite(getattr(self, name), name)
            if name not in ("initial_flow_kg_s", "heat_w", "loss_coefficient") and value <= 0:
                raise ValueError(name + " must be positive")
        if self.heat_w < 0 or self.loss_coefficient < 0:
            raise ValueError("heat and loss coefficient must be nonnegative")
        wall = self.initial_wall_temperature_k
        if wall is None:
            wall = self.initial_temperature_k
        object.__setattr__(
            self, "initial_wall_temperature_k", _finite(wall, "wall T", positive=True)
        )
        for name, positive in (("junction_lengths_m", True), ("junction_loss_coefficients", False)):
            values = getattr(self, name)
            if values is None:
                continue
            if not isinstance(values, (tuple, list)) or len(values) != self.cells + 1:
                raise ValueError(name + " requires cells + 1 entries")
            values = tuple(_finite(value, name, positive=positive) for value in values)
            if any(value < 0 for value in values):
                raise ValueError(name + " must be nonnegative")
            object.__setattr__(self, name, values)
        for name in _OPTIONAL_VECTORS[2:]:
            values = getattr(self, name)
            if values is None:
                continue
            if not isinstance(values, (tuple, list)) or len(values) != self.cells:
                raise ValueError(name + " requires cells entries")
            object.__setattr__(self, name, tuple(_finite(v, name, positive=True) for v in values))
        for p, t in (
            (self.left_pressure_pa, self.left_temperature_k),
            (self.right_pressure_pa, self.right_temperature_k),
            (self.initial_pressure_pa, self.initial_temperature_k),
        ):
            state = properties_pt(p, t)
            if state.region != 1:
                raise DomainError("Pressure-driven channel requires liquid IF97 region 1")
        for i in range(self.cells):
            p = (
                self.initial_pressure_pa
                if self.initial_pressure_profile_pa is None
                else self.initial_pressure_profile_pa[i]
            )
            t = (
                self.initial_temperature_k
                if self.initial_temperature_profile_k is None
                else self.initial_temperature_profile_k[i]
            )
            if properties_pt(p, t).region != 1:
                raise DomainError("Initial profile requires liquid IF97 region 1")
        if self.enthalpy_transport not in ("donor", "muscl_mc"):
            raise ValueError("Unknown enthalpy transport scheme")
        if self.energy_formulation not in ("internal", "compatible_total"):
            raise ValueError("Unknown energy formulation")
        if self.energy_formulation == "compatible_total":
            if self.junction_lengths_m is None:
                raise ValueError("Total energy requires explicit pipe geometry")
            length = sum(self.junction_lengths_m)
            weights = np.r_[0.5, np.ones(self.cells - 1), 0.5]
            if not (
                np.allclose(
                    self.junction_lengths_m, length / self.cells * weights, rtol=1e-12, atol=1e-14
                )
                and np.isclose(self.total_volume_m3, self.area_m2 * length, rtol=1e-12, atol=1e-14)
            ):
                raise ValueError("Total energy requires a uniform dual-volume pipe")
        if self.enthalpy_transport == "muscl_mc":
            if self.cells < 3 or self.junction_lengths_m is None:
                raise ValueError("MUSCL requires >=3 uniform pipe cells and explicit lengths")
            length = sum(self.junction_lengths_m)
            weights = np.r_[0.5, np.ones(self.cells - 1), 0.5]
            if not (
                np.allclose(
                    self.junction_lengths_m, length / self.cells * weights, rtol=1e-12, atol=1e-14
                )
                and np.isclose(self.total_volume_m3, self.area_m2 * length, rtol=1e-12, atol=1e-14)
            ):
                raise ValueError("MUSCL requires uniform pipe volume/length geometry")

    def as_dict(self):
        data = asdict(self)
        for name in _OPTIONAL_VECTORS:
            if data[name] is None:
                data.pop(name)  # Preserve legacy checkpoint/config representation exactly.
            else:
                data[name] = list(data[name])
        if self.enthalpy_transport == "donor":
            data.pop("enthalpy_transport")
        if self.energy_formulation == "internal":
            data.pop("energy_formulation")
        return data

    @classmethod
    def from_pipe(cls, *, cells, total_length_m, total_loss_coefficient, **parameters):
        """Uniform cells, half-length boundary connections, fixed physical pipe."""
        if type(cells) is not int or not 1 <= cells <= 32:
            raise ValueError("cells must be an integer from 1 to 32")
        length = _finite(total_length_m, "total_length_m", positive=True)
        loss = _finite(total_loss_coefficient, "total_loss_coefficient")
        if loss < 0:
            raise ValueError("total loss must be nonnegative")
        forbidden = {
            "total_volume_m3",
            "junction_length_m",
            "loss_coefficient",
            "junction_lengths_m",
            "junction_loss_coefficients",
        }
        if set(parameters) & forbidden:
            raise ValueError("Pipe geometry cannot override derived volumes/connections")
        area = _finite(parameters.get("area_m2"), "area_m2", positive=True)
        weights = (0.5,) + (1.0,) * (cells - 1) + (0.5,)
        return cls(
            cells=cells,
            total_volume_m3=area * length,
            junction_length_m=length / cells,
            loss_coefficient=loss / cells,
            junction_lengths_m=tuple(length / cells * w for w in weights),
            junction_loss_coefficients=tuple(loss / cells * w for w in weights),
            **parameters,
        )


class DynamicWallChannel:
    passport = PASSPORT

    def __init__(self, config: DynamicConfig):
        self.config, self.time_s = config, 0.0
        self.passport = (
            COMPATIBLE_PASSPORT if config.energy_formulation == "compatible_total" else PASSPORT
        )
        c, n = config, config.cells
        volumes = {}
        for i in range(n):
            v = Volume(str(i), c.total_volume_m3 / n, 0.0)
            p = (
                c.initial_pressure_pa
                if c.initial_pressure_profile_pa is None
                else c.initial_pressure_profile_pa[i]
            )
            t = (
                c.initial_temperature_k
                if c.initial_temperature_profile_k is None
                else c.initial_temperature_profile_k[i]
            )
            v.set_liquid(p, t)
            volumes[str(i)] = v
        boundaries = {}
        for name, p, t in (
            ("left", c.left_pressure_pa, c.left_temperature_k),
            ("right", c.right_pressure_pa, c.right_temperature_k),
        ):
            s = properties_pt(p, t)
            boundaries[name] = BoundaryState(p, s.h_j_kg, s.rho_kg_m3, 0.0)
        nodes = ["left"] + list(volumes) + ["right"]
        junctions = {
            str(i): Junction(
                str(i),
                nodes[i],
                nodes[i + 1],
                c.area_m2,
                c.junction_length_m if c.junction_lengths_m is None else c.junction_lengths_m[i],
                c.loss_coefficient
                if c.junction_loss_coefficients is None
                else c.junction_loss_coefficients[i],
                mass_flow_kg_s=c.initial_flow_kg_s,
            )
            for i in range(n + 1)
        }
        self.network = FlowNetwork(volumes, junctions, boundaries)
        # [M,U] per volume, signed junction flows, Tw, boundary mass/energy integrals.
        self._state = np.r_[
            [x for v in volumes.values() for x in (v.mass_kg, v.internal_energy_j)],
            np.full(n + 1, c.initial_flow_kg_s),
            np.full(n, c.initial_wall_temperature_k)
            if c.initial_wall_temperature_profile_k is None
            else c.initial_wall_temperature_profile_k,
            0.0,
            0.0,
        ].astype(float)
        if c.energy_formulation == "compatible_total":
            self._state = np.r_[self._state[:-2], np.zeros(3), self._state[-2:]]
        self._continuation = self._policy = self._horizon = None
        self._solver_totals = dict(
            accepted_rhs_evaluations=0,
            accepted_jacobian_evaluations=0,
            accepted_lu_decompositions=0,
            accepted_solver_steps=0,
            accepted_output_intervals=0,
            failed_output_intervals=0,
            failed_rhs_evaluations=0,
        )
        self._total_rhs_calls = 0
        self._analytic_jacobian_calls = 0
        self._analytic_jacobian_successes = 0
        self._last_jacobian_nnz = 0
        self.initial_mass_kg, self.initial_energy_j = self.inventories(self._state)
        self.initial_kinetic_energy_j = self.kinetic_energy(self._state)
        self.rates(0.0, self._state)

    def _load(self, state):
        n = self.config.cells
        y = np.asarray(state)
        size = 4 * n + (6 if self.config.energy_formulation == "compatible_total" else 3)
        if y.shape != (size,) or not np.all(np.isfinite(y)):
            raise DomainError("Invalid pressure-driven channel state")
        if np.any(y[: 2 * n : 2] <= 0) or np.any(y[3 * n + 1 : 4 * n + 1] <= 0):
            raise DomainError("Nonpositive mass or wall temperature")
        full = np.r_[
            [x for i in range(n) for x in (y[2 * i], y[2 * i + 1], 0.0)], y[2 * n : 3 * n + 1]
        ]
        self.network.unpack(full)
        for volume in self.network.volumes.values():
            density = volume.mass_kg / volume.volume_m3
            energy = volume.internal_energy_j / volume.mass_kg
            thermo = _liquid_state(
                density, energy, self.config.initial_pressure_pa, self.config.initial_temperature_k
            )
            volume.thermo = thermo
            volume.pressure_pa, volume.temperature_k = thermo.p_pa, thermo.T_k
            volume._thermo_key = (density, energy)
        return full

    def inventories(self, state):
        n, c = self.config.cells, self.config
        mass = float(sum(state[: 2 * n : 2]))
        energy = float(
            sum(state[1 : 2 * n : 2])
            + c.total_heat_capacity_j_k / n * sum(state[3 * n + 1 : 4 * n + 1])
        )
        return mass, energy + self.kinetic_energy(state)

    def kinetic_energy(self, state):
        if self.config.energy_formulation != "compatible_total":
            return 0.0
        from physics.hydraulics.mechanical_energy import kinematics

        return float(sum(kinematics(self, state)["kinetic"]))

    def rates(self, t, state):
        del t
        c, n = self.config, self.config.cells
        full = self._load(state)
        tw = state[3 * n + 1 : 4 * n + 1]
        tf = np.array([v.temperature_k for v in self.network.volumes.values()])
        q = c.total_ua_w_k / n * (tw - tf)
        for v, heat in zip(self.network.volumes.values(), q):
            v.heat_w = float(heat)
        raw = self.network.derivatives(full)
        fluid = np.array([raw[3 * i + j] for i in range(n) for j in (0, 1)])
        wall = (c.heat_w / n - q) / (c.total_heat_capacity_j_k / n)
        flows = state[2 * n : 3 * n + 1]
        left, right = self.network.boundaries["left"], self.network.boundaries["right"]
        first, last = self.network.volumes["0"], self.network.volumes[str(n - 1)]
        h_left = left.enthalpy_j_kg if flows[0] >= 0 else first.thermo.h_j_kg
        h_right = last.thermo.h_j_kg if flows[-1] >= 0 else right.enthalpy_j_kg
        reconstructed = None
        if c.enthalpy_transport != "donor":
            enthalpies = np.array([v.thermo.h_j_kg for v in self.network.volumes.values()])
            old_faces = face_values(enthalpies, flows, left.enthalpy_j_kg, right.enthalpy_j_kg)
            reconstructed = face_values(
                enthalpies,
                flows,
                left.enthalpy_j_kg,
                right.enthalpy_j_kg,
                scheme=c.enthalpy_transport,
            )
            correction = flows * (reconstructed - old_faces)
            fluid[1::2] += correction[:-1] - correction[1:]
            h_left, h_right = reconstructed[0], reconstructed[-1]
        mach = []
        nodes = [left] + list(self.network.volumes.values()) + [right]
        for i, flow in enumerate(flows):
            donor = nodes[i] if flow >= 0 else nodes[i + 1]
            if isinstance(donor, BoundaryState):
                # Boundary temperatures are immutable configuration values.
                temperature = c.left_temperature_k if donor is left else c.right_temperature_k
                density, pressure = donor.density_kg_m3, donor.pressure_pa
            else:
                temperature, density, pressure = (
                    donor.temperature_k,
                    donor.thermo.rho_kg_m3,
                    donor.pressure_pa,
                )
            mach.append(abs(flow) / (density * c.area_m2 * _sound_speed(pressure, temperature)))
            if reconstructed is not None:
                _check_face_liquid(float(pressure), float(reconstructed[i]))
        if max(mach) > 0.1:
            raise DomainError("Mach > 0.1 exceeds the low-Mach component passport")
        dm = float(flows[0] - flows[-1])
        de = float(flows[0] * h_left - flows[-1] * h_right + c.heat_w)
        mechanical = None
        momentum = raw[3 * n :]
        extra = []
        if c.energy_formulation == "compatible_total":
            from physics.hydraulics.mechanical_energy import mechanical_rates

            mechanical = mechanical_rates(self, state)
            momentum = mechanical["acceleration"]
            fluid[1::2] += mechanical["internal_correction"]
            de += float(mechanical["kinetic_flux"][0] - mechanical["kinetic_flux"][-1])
            extra = [
                float(sum(mechanical[name]))
                for name in ("pressure_work", "dissipation", "density_work")
            ]
        return dict(
            derivative=np.r_[fluid, momentum, wall, extra, dm, de],
            boundary_mass_kg_s=dm,
            boundary_energy_w=de,
            wall_to_fluid_w=float(sum(q)),
            wall_energy_rate_w=float(sum(wall) * c.total_heat_capacity_j_k / n),
            max_mach=float(max(mach)),
            mechanical=mechanical,
        )

    def derivatives(self, t, state):
        return self.rates(t, state)["derivative"]

    def _counted_derivatives(self, t, state):
        self._total_rhs_calls += 1
        return self.derivatives(t, state)

    def analytic_jacobian(self, t, state):
        from physics.hydraulics.channel_jacobian import channel_jacobian

        self._analytic_jacobian_calls += 1
        result = channel_jacobian(self, t, state)
        self._analytic_jacobian_successes += 1
        self._last_jacobian_nnz = result.nnz
        return result

    def enable_continuation(self, horizon):
        _finite(horizon, "horizon", positive=True)
        if horizon <= self.time_s:
            raise ValueError("horizon must exceed current time")
        self._horizon, self._continuation = float(horizon), None

    def advance(
        self,
        dt_s,
        *,
        rtol=1e-7,
        method="Radau",
        max_step_s=0.1,
        jacobian_relative_step=None,
        absolute_tolerances=None,
        jacobian_mode="finite_difference",
    ):
        _finite(dt_s, "dt_s", positive=True)
        _finite(rtol, "rtol", positive=True)
        _finite(max_step_s, "max_step_s", positive=True)
        if jacobian_relative_step is not None:
            _finite(jacobian_relative_step, "jacobian_relative_step", positive=True)
        if jacobian_mode not in ("finite_difference", "analytic") or (
            jacobian_mode == "analytic" and jacobian_relative_step is not None
        ):
            raise ValueError("Analytic Jacobian excludes a finite-difference step")
        callback = self.analytic_jacobian if jacobian_mode == "analytic" else None
        saved, t0 = self._state.copy(), self.time_s
        target = t0 + dt_s
        n = self.config.cells
        compatible = self.config.energy_formulation == "compatible_total"
        tolerances = validate_absolute_tolerances(
            absolute_tolerances, include_mechanical=compatible
        )
        atol = np.r_[
            np.tile([tolerances["mass_kg"], tolerances["energy_j"]], n),
            np.full(n + 1, tolerances["flow_kg_s"]),
            np.full(n, tolerances["wall_temperature_k"]),
            np.full(3, tolerances["mechanical_work_j"]) if compatible else [],
            tolerances["boundary_mass_kg"],
            tolerances["boundary_energy_j"],
        ]
        policy = (
            rtol,
            method,
            max_step_s,
            jacobian_relative_step,
            tuple(tolerances.items()),
            jacobian_mode,
        )
        rhs_start = self._total_rhs_calls
        try:
            if self._horizon is None:
                result = integrate(
                    self._counted_derivatives,
                    saved,
                    (t0, target),
                    rtol=rtol,
                    atol=atol,
                    max_step=max_step_s,
                    method=method,
                    jacobian_relative_step=jacobian_relative_step,
                    jacobian=callback,
                )
            else:
                if self._continuation is None or self._policy != policy:
                    self._continuation = ImplicitContinuation(
                        self._counted_derivatives,
                        saved,
                        t0,
                        self._horizon,
                        rtol=rtol,
                        atol=atol,
                        max_step=max_step_s,
                        method=method,
                        initial_target=target,
                        jacobian_relative_step=jacobian_relative_step,
                        jacobian=callback,
                    )
                    self._policy = policy
                result = self._continuation.advance_to(target)
            self._load(result.final)
            self._state, self.time_s = result.final.copy(), target
            observation = self.observation()
            self._solver_totals["accepted_rhs_evaluations"] += self._total_rhs_calls - rhs_start
            self._solver_totals["accepted_jacobian_evaluations"] += result.n_jacobian_evaluations
            self._solver_totals["accepted_lu_decompositions"] += result.n_lu_decompositions
            self._solver_totals["accepted_solver_steps"] += result.n_accepted_steps or 0
            self._solver_totals["accepted_output_intervals"] += 1
            return observation
        except BaseException:
            self._state, self.time_s, self._continuation = saved, t0, None
            self._load(saved)
            self._solver_totals["failed_output_intervals"] += 1
            self._solver_totals["failed_rhs_evaluations"] += self._total_rhs_calls - rhs_start
            raise

    def solver_diagnostics(self):
        return dict(
            self._solver_totals,
            analytic_jacobian_calls=self._analytic_jacobian_calls,
            analytic_jacobian_successes=self._analytic_jacobian_successes,
            last_analytic_jacobian_nnz=self._last_jacobian_nnz,
            scope="Process-local solver RHS callbacks, including numerical probes, split by accepted/failed intervals; analytic-Jacobian preparation is separate; counters reset on restore; preflight/observation calls excluded",
        )

    def observation(self):
        n = self.config.cells
        rates = self.rates(self.time_s, self._state)
        mass, energy = self.inventories(self._state)
        pressures = [v.pressure_pa for v in self.network.volumes.values()]
        temperatures = [v.temperature_k for v in self.network.volumes.values()]
        result = dict(
            time_s=self.time_s,
            inlet_mass_flow_kg_s=float(self._state[2 * n]),
            outlet_mass_flow_kg_s=float(self._state[3 * n]),
            min_pressure_pa=float(min(pressures)),
            max_pressure_pa=float(max(pressures)),
            min_temperature_k=float(min(temperatures)),
            max_temperature_k=float(max(temperatures)),
            right_cell_temperature_k=float(temperatures[-1]),
            midpoint_pressure_pa=float(np.interp(0.5, (np.arange(n) + 0.5) / n, pressures)),
            mean_wall_temperature_k=float(np.mean(self._state[3 * n + 1 : 4 * n + 1])),
            min_wall_temperature_k=float(min(self._state[3 * n + 1 : 4 * n + 1])),
            max_wall_temperature_k=float(max(self._state[3 * n + 1 : 4 * n + 1])),
            heat_w=self.config.heat_w,
            wall_to_fluid_w=rates["wall_to_fluid_w"],
            max_mach=rates["max_mach"],
            stored_mass_kg=mass,
            stored_energy_j=energy,
            mass_residual_kg=float(mass - self.initial_mass_kg - self._state[-2]),
            energy_residual_j=float(energy - self.initial_energy_j - self._state[-1]),
        )
        if self.config.energy_formulation == "compatible_total":
            r = rates["mechanical"]
            kinetic = float(sum(r["kinetic"]))
            work = self._state[4 * n + 1 : 4 * n + 4]
            result.update(
                kinetic_energy_j=kinetic,
                thermal_energy_j=energy - kinetic,
                pressure_work_w=float(sum(r["pressure_work"])),
                dissipation_w=float(sum(r["dissipation"])),
                density_work_w=float(sum(r["density_work"])),
                kinetic_rate_w=float(sum(r["kinetic_rate"])),
                kinetic_boundary_power_w=float(r["kinetic_flux"][0] - r["kinetic_flux"][-1]),
                pressure_work_integral_j=float(work[0]),
                dissipation_integral_j=float(work[1]),
                density_work_integral_j=float(work[2]),
                mechanical_residual_j=float(
                    kinetic - self.initial_kinetic_energy_j - work[0] + work[1] - work[2]
                ),
                mechanical_identity_residual_w=r["identity_residual_w"],
            )
        return result

    def snapshot(self):
        return dict(
            schema_version=2 if self.config.energy_formulation == "compatible_total" else 1,
            model_id=self.passport.model_id,
            model_version=self.passport.version,
            config=self.config.as_dict(),
            time_s=self.time_s,
            state=self._state.tolist(),
        )

    @classmethod
    def from_snapshot(cls, payload):
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {"schema_version", "model_id", "model_version", "config", "time_s", "state"}
            or type(payload.get("schema_version")) is not int
        ):
            raise DomainError("Incompatible dynamic channel checkpoint")
        try:
            c = DynamicConfig(**payload["config"])
            t = _finite(payload["time_s"], "time_s")
            state = np.asarray(payload["state"], dtype=float)
            if t < 0:
                raise ValueError("Negative checkpoint time")
        except (ValueError, TypeError, KeyError) as exc:
            raise DomainError("Invalid dynamic channel checkpoint") from exc
        expected = COMPATIBLE_PASSPORT if c.energy_formulation == "compatible_total" else PASSPORT
        schema = 2 if c.energy_formulation == "compatible_total" else 1
        if (
            payload["schema_version"] != schema
            or payload["model_id"] != expected.model_id
            or payload["model_version"] != expected.version
        ):
            raise DomainError("Checkpoint energy formulation/header mismatch")
        model = cls(c)
        model._load(state)
        model.time_s, model._state = t, state.copy()
        return model
