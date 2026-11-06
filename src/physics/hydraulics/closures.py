"""Замыкания теплогидравлики с явной областью применимости."""

from __future__ import annotations

from dataclasses import dataclass

from core.status import Applicability, DomainError, ModelPassport, ModelStatus

PASSPORT_DITTUS = ModelPassport(
    model_id="dittus-boelter-heating",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Турбулентная однофазная теплоотдача в круглой трубе",
    approximations=("Nu = 0.023 Re^0.8 Pr^0.4", "нагрев жидкости"),
    applicability="Re >= 10000, 0.7 <= Pr <= 160, L/D > 10, однофазная жидкость",
    data_sources=(
        "Dittus F.W., Boelter L.M.K., Univ. California Publ. Eng. 2 (1930), форма, принятая в инженерных справочниках",
    ),
)

PASSPORT_LAMINAR = ModelPassport(
    model_id="laminar-nusselt-3.66",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Развитый ламинарный теплообмен при постоянной температуре стенки",
    approximations=("Nu = 3.66",),
    applicability="Re < 2300, круглая труба, развитый профиль",
    data_sources=("Классическое решение Гретца–Нуссельта для постоянной температуры стенки",),
)


@dataclass
class CorrelationResult:
    value: float
    applicability: Applicability
    model_id: str


def annulus_flow_area_m2(outer_diameter_m: float, inner_diameter_m: float) -> float:
    """Площадь кольцевого зазора. Гидравлический диаметр этого зазора равен Do − Di."""
    from math import pi

    if inner_diameter_m < 0.0 or outer_diameter_m <= inner_diameter_m:
        raise DomainError("Кольцевой зазор задан при неположительной ширине")
    return pi / 4.0 * (outer_diameter_m**2 - inner_diameter_m**2)


def annulus_hydraulic_diameter_m(outer_diameter_m: float, inner_diameter_m: float) -> float:
    if inner_diameter_m < 0.0 or outer_diameter_m <= inner_diameter_m:
        raise DomainError("Гидравлический диаметр кольца задан при неположительной ширине")
    return outer_diameter_m - inner_diameter_m


def area(diameter_m: float) -> float:
    from math import pi

    return pi * diameter_m**2 / 4.0


def reynolds_number(mass_flow_kg_s: float, diameter_m: float, viscosity_pa_s: float) -> float:
    """Re = |m| D / (mu A)."""
    if diameter_m <= 0.0 or viscosity_pa_s <= 0.0:
        raise DomainError("Диаметр и вязкость должны быть положительными")
    return abs(mass_flow_kg_s) * diameter_m / (viscosity_pa_s * area(diameter_m))


def prandtl(viscosity_pa_s: float, cp_j_kg_k: float, conductivity: float) -> float:
    return viscosity_pa_s * cp_j_kg_k / conductivity


def nusselt_dittus_boelter(re_number: float, pr_number: float) -> CorrelationResult:
    if re_number < 10000.0 or pr_number < 0.7 or pr_number > 160.0:
        raise DomainError("Dittus–Boelter вне Re>=10000 и 0.7<=Pr<=160")
    nu = 0.023 * re_number**0.8 * pr_number**0.4
    return CorrelationResult(nu, Applicability.IN_DOMAIN, PASSPORT_DITTUS.model_id)


def nusselt_laminar() -> CorrelationResult:
    return CorrelationResult(3.66, Applicability.IN_DOMAIN, PASSPORT_LAMINAR.model_id)


def thom_nucleate_flux_w_m2(
    wall_temperature_k: float, saturation_temperature_k: float, pressure_pa: float
) -> float:
    """Thom, Walker, Fallon, Reising, Proc. IMechE 180 (1965): ΔT = 22.65 q^0.5 exp(-p/87).

    q в МВт/м², p в бар, ΔT в K. Кризис теплоотдачи и post-CHF этой формулой не заменяются.
    """
    import math

    if wall_temperature_k <= saturation_temperature_k:
        raise DomainError("Кипение Тома требует перегрев стенки над насыщением")
    pressure_bar = pressure_pa / 1.0e5
    if pressure_bar < 1.0 or pressure_bar > 150.0:
        raise DomainError("Корреляция Тома вне 1–150 бар")
    delta = wall_temperature_k - saturation_temperature_k
    factor = 22.65 * math.exp(-pressure_bar / 87.0)
    return (delta / factor) ** 2 * 1.0e6


def heat_flux_w_m2(regime: str, **kwargs: float) -> float:
    """Диспетчер режима. Необъявленный режим — отказ, не соседняя корреляция."""
    if regime == "single_phase":
        return float(kwargs["coefficient_w_m2_k"]) * (
            float(kwargs["wall_k"]) - float(kwargs["bulk_k"])
        )
    if regime == "nucleate_thom":
        return thom_nucleate_flux_w_m2(
            float(kwargs["wall_k"]), float(kwargs["saturation_k"]), float(kwargs["pressure_pa"])
        )
    if regime in ("chf", "post_chf"):
        raise DomainError("Кризис и post-CHF не подставляются без собственной области и данных")
    raise DomainError("Режим теплоотдачи %s не заявлен отдельной корреляцией" % regime)


def heat_transfer_coefficient(
    mass_flow_kg_s: float, diameter_m: float, viscosity: float, cp: float, conductivity: float
) -> CorrelationResult:
    re_number = reynolds_number(mass_flow_kg_s, diameter_m, viscosity)
    pr_number = prandtl(viscosity, cp, conductivity)
    if re_number < 2300.0:
        nu = nusselt_laminar()
    elif re_number >= 10000.0:
        nu = nusselt_dittus_boelter(re_number, pr_number)
    else:
        raise DomainError(
            "Переходный режим 2300<=Re<10000 не имеет отдельной корреляции в этом реестре"
        )
    return CorrelationResult(nu.value * conductivity / diameter_m, nu.applicability, nu.model_id)


def poiseuille_drop_pa(
    mass_flow_kg_s: float,
    diameter_m: float,
    viscosity_pa_s: float,
    length_m: float,
    density_kg_m3: float,
) -> float:
    """Ламинарный перепад Пуазейля: 128 μ L |m| / (π D^4 ρ)."""
    from math import pi

    if min(diameter_m, viscosity_pa_s, length_m, density_kg_m3) <= 0.0:
        raise DomainError(
            "Диаметр, вязкость, длина и плотность Пуазейля должны быть положительными"
        )
    return (
        128.0
        * viscosity_pa_s
        * length_m
        * abs(mass_flow_kg_s)
        / (pi * diameter_m**4 * density_kg_m3)
    )


def haaland_friction_factor(re_number: float, roughness_m: float, diameter_m: float) -> float:
    """Явная формула Хааланда. Шероховатость увеличивает коэффициент трения."""
    import math

    if re_number <= 0.0 or diameter_m <= 0.0 or roughness_m < 0.0:
        raise DomainError("Хааланд задан при неположительном Re или диаметре")
    inner = (roughness_m / diameter_m) / 3.7
    return (1.0 / (-1.8 * math.log10(inner**1.11 + 6.9 / re_number))) ** 2


def poiseuille_flow_kg_s(
    pressure_drop_pa: float,
    diameter_m: float,
    viscosity_pa_s: float,
    length_m: float,
    density_kg_m3: float,
) -> float:
    """Расход из перепада Пуазейля. Обратный перепад даёт обратный расход."""
    from math import pi

    if min(diameter_m, viscosity_pa_s, length_m, density_kg_m3) <= 0.0:
        raise DomainError(
            "Диаметр, вязкость, длина и плотность Пуазейля должны быть положительными"
        )
    return (
        pressure_drop_pa * pi * diameter_m**4 * density_kg_m3 / (128.0 * viscosity_pa_s * length_m)
    )


def parallel_poiseuille_flow_kg_s(
    pressure_drop_pa: float,
    diameters_m,
    viscosity_pa_s: float,
    length_m: float,
    density_kg_m3: float,
) -> float:
    """Параллельные ламинарные трубы одного перепада и одной длины. Расходы складываются."""
    diameters = list(diameters_m)
    if not diameters:
        raise DomainError("Параллельная сборка труб пуста")
    return float(
        sum(
            poiseuille_flow_kg_s(
                pressure_drop_pa, diameter, viscosity_pa_s, length_m, density_kg_m3
            )
            for diameter in diameters
        )
    )


def series_poiseuille_drop_pa(
    mass_flow_kg_s: float,
    diameter_m: float,
    viscosity_pa_s: float,
    lengths_m,
    density_kg_m3: float,
) -> float:
    """Последовательные ламинарные участки одного расхода. Перепады складываются."""
    lengths = list(lengths_m)
    if not lengths:
        raise DomainError("Последовательная труба не содержит участков")
    return float(
        sum(
            poiseuille_drop_pa(mass_flow_kg_s, diameter_m, viscosity_pa_s, length, density_kg_m3)
            for length in lengths
        )
    )


def friction_darcy(
    mass_flow_kg_s: float,
    diameter_m: float,
    viscosity: float,
    roughness_m: float,
    length_m: float,
    density: float,
) -> float:
    """Потеря давления Дарси–Вейсбаха. Ламинарный f=64/Re, турбулентный — Хааланд.

    Знак не зашит: возвращается положительный модуль перепада по направлению модуля расхода.
    Обратный расход учитывается вызывающим кодом через m|m|.
    """
    re_number = reynolds_number(mass_flow_kg_s, diameter_m, viscosity)
    if re_number < 1e-8:
        return 0.0
    if re_number < 2300.0:
        return poiseuille_drop_pa(mass_flow_kg_s, diameter_m, viscosity, length_m, density)
    else:
        f_coef = haaland_friction_factor(re_number, roughness_m, diameter_m)
        model = "haaland"
    del model
    dynamic = mass_flow_kg_s**2 / (density * area(diameter_m) ** 2)
    return f_coef * (length_m / diameter_m) * dynamic / 2.0


def series_darcy_drop_pa(
    mass_flow_kg_s: float,
    diameter_m: float,
    viscosity: float,
    roughnesses_m,
    lengths_m,
    density: float,
) -> float:
    """Последовательные участки одного расхода. Перепады Дарси–Вейсбаха складываются."""
    roughnesses = list(roughnesses_m)
    lengths = list(lengths_m)
    if not lengths or len(roughnesses) != len(lengths):
        raise DomainError(
            "Последовательный Дарси задан пустым или несогласованным списком участков"
        )
    return float(
        sum(
            friction_darcy(mass_flow_kg_s, diameter_m, viscosity, rough, length, density)
            for rough, length in zip(roughnesses, lengths)
        )
    )
