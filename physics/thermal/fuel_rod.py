"""Радиальная теплопроводность твэла. Зазор — проводимость и излучение, без механики контакта."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.solvers import integrate
from core.status import DomainError, ModelPassport, ModelStatus

PASSPORT = ModelPassport(
    model_id="fuel-rod-radial-fv",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Радиальная теплопроводность топлива и оболочки",
    approximations=(
        "осевая теплопроводность не учитывается",
        "зазор не закрывается и не зависит от давления газов",
        "поток на оси равен нулю",
    ),
    applicability="Температуры в диапазоне корреляций k(T); повреждение топлива не моделируется",
    data_sources=("Геометрия и свойства передаются вызывающей стороной",),
    notes=("Температурный порог сам по себе не является моделью разрушения.",),
)

STEFAN_BOLTZMANN = 5.670374419e-8


def gap_conductance(
    k_gas_w_m_k: float, gap_m: float, emissivity: float, t_fuel_k: float, t_clad_k: float
) -> float:
    """h = k_gas/gap + излучение серых поверхностей. Контакт не добавляется."""
    if gap_m <= 0.0:
        raise DomainError(
            "Закрытый зазор требует термомеханической модели",
        )
    conduction = k_gas_w_m_k / gap_m
    radiation = emissivity * STEFAN_BOLTZMANN * (t_fuel_k + t_clad_k) * (t_fuel_k**2 + t_clad_k**2)
    return conduction + radiation


def analytical_cylinder_temperature(
    radius_m: float,
    outer_radius_m: float,
    volumetric_power_w_m3: float,
    conductivity: float,
    surface_temperature_k: float,
) -> float:
    """Стационар цилиндра с постоянным k и q''': T(r) = Ts + q'''/(4k) * (R^2 - r^2)."""
    return surface_temperature_k + volumetric_power_w_m3 / (4.0 * conductivity) * (
        outer_radius_m**2 - radius_m**2
    )


def cell_average_analytical(
    r_inner: float,
    r_outer: float,
    outer_radius: float,
    volumetric_power: float,
    conductivity: float,
    surface_temperature: float,
) -> float:
    """Среднее по площади кольца аналитического профиля."""
    r2 = outer_radius**2
    mean_r2 = 0.5 * (r_inner**2 + r_outer**2)
    return surface_temperature + volumetric_power / (4.0 * conductivity) * (r2 - mean_r2)


@dataclass
class FuelRod:
    r_fuel_m: float
    r_clad_inner_m: float
    r_clad_outer_m: float
    n_fuel: int
    n_clad: int
    length_m: float
    n_rods: int
    rho_fuel: float
    cp_fuel: float
    rho_clad: float
    cp_clad: float
    k_fuel: float
    k_clad: float
    h_gap: float
    h_coolant: float

    def __post_init__(self) -> None:
        self.fuel_faces = np.linspace(0.0, self.r_fuel_m, self.n_fuel + 1)
        self.clad_faces = np.linspace(self.r_clad_inner_m, self.r_clad_outer_m, self.n_clad + 1)
        self.n_nodes = self.n_fuel + self.n_clad

    def fuel_cell_bounds(self):
        return self.fuel_faces[:-1], self.fuel_faces[1:]

    def _ring_volume(self, r_in: float, r_out: float) -> float:
        return np.pi * (r_out**2 - r_in**2) * self.length_m * self.n_rods

    def volumes(self) -> np.ndarray:
        values = []
        for r_in, r_out in zip(self.fuel_faces[:-1], self.fuel_faces[1:]):
            values.append(self._ring_volume(r_in, r_out))
        for r_in, r_out in zip(self.clad_faces[:-1], self.clad_faces[1:]):
            values.append(self._ring_volume(r_in, r_out))
        return np.array(values)

    def capacitance(self) -> np.ndarray:
        vol = self.volumes()
        cap = np.empty(self.n_nodes)
        cap[: self.n_fuel] = self.rho_fuel * self.cp_fuel * vol[: self.n_fuel]
        cap[self.n_fuel :] = self.rho_clad * self.cp_clad * vol[self.n_fuel :]
        return cap

    def _centers(self, faces: np.ndarray) -> np.ndarray:
        r_in = faces[:-1]
        r_out = faces[1:]
        centers = np.sqrt(0.5 * (r_in**2 + r_out**2))
        centers[0] = max(centers[0], 0.5 * faces[1])
        return centers

    def conductance_matrix(self, coolant_temperature_k: float | None) -> np.ndarray:
        """Матрица K: Q_out_linear = K T, так что C dT/dt = q - K T.

        Если coolant_temperature_k задан, связь с теплоносителем переносится в источник
        отдельно; здесь K включает только проводимости между узлами и к границе,
        а вклад h*T_coolant возвращается вызывающим кодом.
        """
        k_matrix = np.zeros((self.n_nodes, self.n_nodes))
        fuel_c = self._centers(self.fuel_faces)
        clad_c = self._centers(self.clad_faces)
        for i in range(self.n_fuel - 1):
            g = (
                self.n_rods
                * 2.0
                * np.pi
                * self.k_fuel
                * self.length_m
                / np.log(fuel_c[i + 1] / fuel_c[i])
            )
            k_matrix[i, i] += g
            k_matrix[i, i + 1] -= g
            k_matrix[i + 1, i] -= g
            k_matrix[i + 1, i + 1] += g
        g_gap = self.h_gap * 2.0 * np.pi * self.r_fuel_m * self.length_m * self.n_rods
        last_fuel = self.n_fuel - 1
        first_clad = self.n_fuel
        # Дополнительное сопротивление топлива от центра последней топливной ячейки до поверхности
        # и оболочки от внутренней поверхности до центра первой ячейки оболочки входит в log-члены,
        # а зазор — отдельной проводимостью. Для тонкого зазора проводимость велика.
        g_fuel_skin = (
            self.n_rods
            * 2.0
            * np.pi
            * self.k_fuel
            * self.length_m
            / np.log(self.r_fuel_m / fuel_c[-1])
        )
        g_clad_skin = (
            self.n_rods
            * 2.0
            * np.pi
            * self.k_clad
            * self.length_m
            / np.log(clad_c[0] / self.r_clad_inner_m)
        )
        g_interface = 1.0 / (1.0 / g_fuel_skin + 1.0 / g_gap + 1.0 / g_clad_skin)
        k_matrix[last_fuel, last_fuel] += g_interface
        k_matrix[last_fuel, first_clad] -= g_interface
        k_matrix[first_clad, last_fuel] -= g_interface
        k_matrix[first_clad, first_clad] += g_interface
        for i in range(self.n_clad - 1):
            a = first_clad + i
            g = (
                self.n_rods
                * 2.0
                * np.pi
                * self.k_clad
                * self.length_m
                / np.log(clad_c[i + 1] / clad_c[i])
            )
            k_matrix[a, a] += g
            k_matrix[a, a + 1] -= g
            k_matrix[a + 1, a] -= g
            k_matrix[a + 1, a + 1] += g
        g_out = (
            self.n_rods
            * 2.0
            * np.pi
            * self.k_clad
            * self.length_m
            / np.log(self.r_clad_outer_m / clad_c[-1])
        )
        h_out = self.h_coolant * 2.0 * np.pi * self.r_clad_outer_m * self.length_m * self.n_rods
        g_coolant = 1.0 / (1.0 / g_out + 1.0 / h_out)
        k_matrix[-1, -1] += g_coolant
        self._coolant_conductance = g_coolant
        return k_matrix

    def generation(self, volumetric_power_w_m3: float) -> np.ndarray:
        q = np.zeros(self.n_nodes)
        for i, (r_in, r_out) in enumerate(zip(self.fuel_faces[:-1], self.fuel_faces[1:])):
            q[i] = volumetric_power_w_m3 * self._ring_volume(r_in, r_out)
        return q

    def fuel_only_operator(self) -> tuple:
        """Топливо с заданной температурой внешней поверхности топлива. Возвращает K и G_surface."""
        n = self.n_fuel
        matrix = np.zeros((n, n))
        centers = self._centers(self.fuel_faces)
        for i in range(n - 1):
            g = (
                self.n_rods
                * 2.0
                * np.pi
                * self.k_fuel
                * self.length_m
                / np.log(centers[i + 1] / centers[i])
            )
            matrix[i, i] += g
            matrix[i, i + 1] -= g
            matrix[i + 1, i] -= g
            matrix[i + 1, i + 1] += g
        g_surface = (
            self.n_rods
            * 2.0
            * np.pi
            * self.k_fuel
            * self.length_m
            / np.log(self.r_fuel_m / centers[-1])
        )
        matrix[-1, -1] += g_surface
        return matrix, g_surface

    def steady_fuel(self, volumetric_power_w_m3: float, surface_temperature_k: float) -> np.ndarray:
        matrix, g_surface = self.fuel_only_operator()
        rhs = self.generation(volumetric_power_w_m3)[: self.n_fuel]
        rhs[-1] += g_surface * surface_temperature_k
        return np.linalg.solve(matrix, rhs)

    def derivatives(
        self, temperature: np.ndarray, volumetric_power: float, coolant_temperature_k: float
    ) -> np.ndarray:
        matrix = self.conductance_matrix(coolant_temperature_k)
        source = self.generation(volumetric_power)
        source[-1] += self._coolant_conductance * coolant_temperature_k
        return (source - matrix @ temperature) / self.capacitance()

    def integrate(
        self,
        temperature: np.ndarray,
        volumetric_power: float,
        coolant_temperature_k: float,
        t_end: float,
    ) -> np.ndarray:
        def rhs(_time: float, y: np.ndarray) -> np.ndarray:
            return self.derivatives(y, volumetric_power, coolant_temperature_k)

        return integrate(rhs, temperature, (0.0, t_end), rtol=1e-6, atol=1e-6).final

    def boundary_heat_w(self, temperature: np.ndarray, coolant_temperature_k: float) -> float:
        self.conductance_matrix(coolant_temperature_k)
        return float(self._coolant_conductance * (temperature[-1] - coolant_temperature_k))
