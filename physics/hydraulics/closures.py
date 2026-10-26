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
        f_coef = 64.0 / re_number
        model = "laminar-64-re"
    else:
        if roughness_m < 0.0:
            raise DomainError("Шероховатость отрицательна")
        # Haaland explicit approximation of Colebrook, Re>=2300.
        import math

        inner = (roughness_m / diameter_m) / 3.7
        f_coef = (1.0 / (-1.8 * math.log10(inner**1.11 + 6.9 / re_number))) ** 2
        model = "haaland"
    del model
    dynamic = mass_flow_kg_s**2 / (density * area(diameter_m) ** 2)
    return f_coef * (length_m / diameter_m) * dynamic / 2.0
