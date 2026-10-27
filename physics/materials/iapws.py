"""IAPWS-IF97 для областей 1, 2, 3, 5 и линии насыщения 4.

Коэффициенты — из Revised Release IAPWS-IF97 (2007). Вязкость — IAPWS 2008
без критической добавки. Теплопроводность — IAPWS 2011 без критической добавки.
Насыщение выше 623.15 K по-прежнему вне области: это не подменяется проекцией.

Внутренний интерфейс модуля принимает СИ (Па, Дж/кг) и переводит в МПа и кДж/кг
только на границе формуляции.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.status import Applicability, DomainError

R_KJ = 0.461526  # кДж/(кг·К), удельная газовая постоянная воды в IF97
TC_K = 647.096
PC_MPA = 22.064
RHOC = 322.0
P_MIN_MPA = 611.212677e-6
P_REGION13_MPA = 16.5291642526
T_MIN = 273.15
T_REGION1_MAX = 623.15
T_REGION2_MAX = 1073.15

_R1_I = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        4,
        4,
        4,
        5,
        8,
        8,
        21,
        23,
        29,
        30,
        31,
        32,
    ],
    dtype=float,
)
_R1_J = np.array(
    [
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        -9,
        -7,
        -1,
        0,
        1,
        3,
        -3,
        0,
        1,
        3,
        17,
        -4,
        0,
        6,
        -5,
        -2,
        10,
        -8,
        -11,
        -6,
        -29,
        -31,
        -38,
        -39,
        -40,
        -41,
    ],
    dtype=float,
)
_R1_N = np.array(
    [
        0.14632971213167,
        -0.84548187169114,
        -3.756360367204,
        0.33855169168385e1,
        -0.95791963387872,
        0.15772038513228,
        -0.16616417199501e-1,
        0.81214629983568e-3,
        0.28319080123804e-3,
        -0.60706301565874e-3,
        -0.18990068218419e-1,
        -0.32529748770505e-1,
        -0.21841717175414e-1,
        -0.52838357969930e-4,
        -0.47184321073267e-3,
        -0.30001780793026e-3,
        0.47661393906987e-4,
        -0.44141845330846e-5,
        -0.72694996297594e-15,
        -0.31679644845054e-4,
        -0.28270797985312e-5,
        -0.85205128120103e-9,
        -0.22425281908000e-5,
        -0.65171222895601e-6,
        -0.14341729937924e-12,
        -0.40516996860117e-6,
        -0.12734301741641e-8,
        -0.17424871230634e-9,
        -0.68762131295531e-18,
        0.14478307828521e-19,
        0.26335781662795e-22,
        -0.11947622640071e-22,
        0.18228094581404e-23,
        -0.93537087292458e-25,
    ]
)

_R2_I = np.array(
    [
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        5,
        6,
        6,
        6,
        7,
        7,
        7,
        8,
        8,
        9,
        10,
        10,
        10,
        16,
        16,
        18,
        20,
        20,
        20,
        21,
        22,
        23,
        24,
        24,
        24,
    ],
    dtype=float,
)
_R2_J = np.array(
    [
        0,
        1,
        2,
        3,
        6,
        1,
        2,
        4,
        7,
        36,
        0,
        1,
        3,
        6,
        35,
        1,
        2,
        3,
        7,
        3,
        16,
        35,
        0,
        11,
        25,
        8,
        36,
        13,
        4,
        10,
        14,
        29,
        50,
        57,
        20,
        35,
        48,
        21,
        53,
        39,
        26,
        40,
        58,
    ],
    dtype=float,
)
_R2_N = np.array(
    [
        -0.17731742473213e-2,
        -0.17834862292358e-1,
        -0.45996013696365e-1,
        -0.57581259083432e-1,
        -0.50325278727930e-1,
        -0.33032641670203e-4,
        -0.18948987516315e-3,
        -0.39392777243355e-2,
        -0.43797295650573e-1,
        -0.26674547914087e-4,
        0.20481737692309e-7,
        0.43870667284435e-6,
        -0.32277677238570e-4,
        -0.15033924542148e-2,
        -0.40668253562649e-1,
        -0.78847309559367e-9,
        0.12790717852285e-7,
        0.48225372718507e-6,
        0.22922076337661e-5,
        -0.16714766451061e-10,
        -0.21171471012189e-2,
        -0.23895741934104e2,
        -0.59059564324270e-17,
        -0.12621808899101e-5,
        -0.38946842435739e-1,
        0.11256211360459e-10,
        -0.82311340897998e1,
        0.19809712802088e-7,
        0.10406965210174e-18,
        -0.10234747095929e-12,
        -0.10018179379511e-8,
        -0.80882908646985e-10,
        0.10693031879409,
        -0.33662250574171,
        0.89185845355421e-24,
        0.30629316876232e-12,
        -0.42002467698208e-5,
        -0.59056029685639e-25,
        0.37826947613457e-5,
        -0.12768608934681e-14,
        0.73087610595061e-28,
        0.55414715350778e-16,
        -0.94369707241210e-6,
    ]
)
_R2_J0 = np.array([0, 1, -5, -4, -3, -2, -1, 2, 3], dtype=float)
_R2_N0 = np.array(
    [
        -0.96927686500217e1,
        0.10086655968018e2,
        -0.56087911283020e-2,
        0.71452738081455e-1,
        -0.40710498223928,
        0.14240819171444e1,
        -0.43839511319450e1,
        -0.28408632460772,
        0.21268463753307e-1,
    ]
)

_SAT_N = np.array(
    [
        0.11670521452767e4,
        -0.72421316703206e6,
        -0.17073846940092e2,
        0.12020824702470e5,
        -0.32325550322333e7,
        0.14915108613530e2,
        -0.48232657361591e4,
        0.40511340542057e6,
        -0.23855557567849,
        0.65017534844798e3,
    ]
)

_MU_H0 = np.array([1.67752, 2.20462, 0.6366564, -0.241605])
_MU_I = np.array([0, 1, 2, 3, 0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 0, 1, 0, 3, 4, 3, 5])
_MU_J = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6])
_MU_H = np.array(
    [
        0.520094,
        0.850895e-1,
        -0.108374e1,
        -0.289555,
        0.222531,
        0.999115,
        0.188797e1,
        0.126613e1,
        0.120573,
        -0.281378,
        -0.906851,
        -0.772479,
        -0.489837,
        -0.257040,
        0.161913,
        0.257399,
        -0.325372e-1,
        0.698452e-1,
        0.872102e-2,
        -0.435673e-2,
        -0.593264e-3,
    ]
)
_K_N0 = np.array([2.443221e-3, 1.323095e-2, 6.770357e-3, -3.454586e-3, 4.096266e-4])
_K_I = np.array(
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
)
_K_J = np.array(
    [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5]
)
_K_N = np.array(
    [
        1.60397357,
        -0.646013523,
        0.111443906,
        0.102997357,
        -0.0504123634,
        0.00609859258,
        2.33771842,
        -2.78843778,
        1.53616167,
        -0.463045512,
        0.0832827019,
        -0.00719201245,
        2.19650529,
        -4.54580785,
        3.55777244,
        -1.40944978,
        0.275418278,
        -0.0205938816,
        -1.21051378,
        1.60812989,
        -0.621178141,
        0.0716373224,
        -2.7203370,
        4.57586331,
        -3.18369245,
        1.1168348,
        -0.19268305,
        0.012913842,
    ]
)


@dataclass(frozen=True)
class ThermoState:
    """Термодинамическое состояние воды. Давление и энергия — в СИ."""

    T_k: float
    p_pa: float
    rho_kg_m3: float
    v_m3_kg: float
    u_j_kg: float
    h_j_kg: float
    s_j_kg_k: float
    cp_j_kg_k: float
    cv_j_kg_k: float
    sound_speed_m_s: float
    viscosity_pa_s: float
    conductivity_w_m_k: float
    quality: float
    region: int
    applicability: Applicability = Applicability.IN_DOMAIN
    energy_mismatch_j_kg: float = 0.0
    volume_mismatch: float = 0.0

    @property
    def phase(self) -> str:
        if self.region == 4:
            return "two_phase"
        if self.region == 2 or self.region == 5:
            return "vapor"
        if self.region == 3:
            return "region3"
        return "liquid"


def saturation_pressure_mpa(temperature_k: float) -> float:
    if temperature_k < T_MIN or temperature_k > TC_K:
        raise DomainError("Давление насыщения IF97 запрошено вне 273.15–647.096 K")
    n = _SAT_N
    theta = temperature_k + n[8] / (temperature_k - n[9])
    a_coef = theta**2 + n[0] * theta + n[1]
    b_coef = n[2] * theta**2 + n[3] * theta + n[4]
    c_coef = n[5] * theta**2 + n[6] * theta + n[7]
    disc = b_coef**2 - 4.0 * a_coef * c_coef
    return (2.0 * c_coef / (-b_coef + np.sqrt(disc))) ** 4


def boundary_23_mpa(temperature_k: float) -> float:
    """Граница областей 2 и 3, уравнение B23. Давление в МПа."""
    if temperature_k < T_REGION1_MAX or temperature_k > 863.15:
        raise DomainError("Граница B23 определена на 623.15–863.15 K")
    theta = temperature_k
    return 0.34805185628969e3 - 0.11671859879975e1 * theta + 0.10192970039326e-2 * theta**2


def saturation_temperature_k(pressure_mpa: float) -> float:
    if pressure_mpa < P_MIN_MPA or pressure_mpa > PC_MPA:
        raise DomainError("Температура насыщения IF97 запрошена вне диапазона давления")
    n = _SAT_N
    beta = pressure_mpa**0.25
    e_coef = beta**2 + n[2] * beta + n[5]
    f_coef = n[0] * beta**2 + n[3] * beta + n[6]
    g_coef = n[1] * beta**2 + n[4] * beta + n[7]
    disc = f_coef**2 - 4.0 * e_coef * g_coef
    d_coef = 2.0 * g_coef / (-f_coef - np.sqrt(disc))
    return (n[9] + d_coef - np.sqrt((n[9] + d_coef) ** 2 - 4.0 * (n[8] + n[9] * d_coef))) / 2.0


def classify_region(temperature_k: float, pressure_mpa: float) -> int:
    """Номер области IF97 либо DomainError, если формуляция не реализована."""
    if temperature_k < T_MIN or pressure_mpa <= 0.0 or pressure_mpa > 100.0:
        raise DomainError("T или p вне промышленной формуляции IF97")
    if temperature_k <= T_REGION1_MAX:
        p_sat = saturation_pressure_mpa(temperature_k)
        if pressure_mpa >= p_sat:
            return 1
        return 2
    if temperature_k <= 863.15:
        if pressure_mpa <= boundary_23_mpa(temperature_k):
            return 2
        if pressure_mpa <= 100.0:
            return 3
        raise DomainError("Давление выше 100 МПа не входит в IF97")
    if temperature_k <= T_REGION2_MAX:
        if pressure_mpa <= 100.0:
            return 2
        raise DomainError("Давление выше 100 МПа не входит в IF97")
    if temperature_k <= 2273.15 and pressure_mpa <= 50.0:
        return 5
    raise DomainError("T или p вне областей 1–5 IF97")


def _region1_raw(temperature_k: float, pressure_mpa: float) -> dict:
    tau = 1386.0 / temperature_k
    pi = pressure_mpa / 16.53
    base_p = 7.1 - pi
    base_t = tau - 1.222
    if base_p <= 0.0 or base_t <= 0.0:
        raise DomainError("Аргумент области 1 IF97 покинул допустимый интервал")
    bp_i = base_p**_R1_I
    bt_j = base_t**_R1_J
    gamma = float(np.sum(_R1_N * bp_i * bt_j))
    i_pos = _R1_I > 0.0
    gamma_pi = float(
        -np.sum(
            _R1_N[i_pos] * _R1_I[i_pos] * base_p ** (_R1_I[i_pos] - 1.0) * base_t ** _R1_J[i_pos]
        )
    )
    i_second = _R1_I > 1.0
    gamma_pi_pi = float(
        np.sum(
            _R1_N[i_second]
            * _R1_I[i_second]
            * (_R1_I[i_second] - 1.0)
            * base_p ** (_R1_I[i_second] - 2.0)
            * base_t ** _R1_J[i_second]
        )
    )
    j_pos = _R1_J != 0.0
    gamma_tau = float(
        np.sum(_R1_N[j_pos] * bp_i[j_pos] * _R1_J[j_pos] * base_t ** (_R1_J[j_pos] - 1.0))
    )
    j_second = (_R1_J != 0.0) & (_R1_J != 1.0)
    gamma_tau_tau = float(
        np.sum(
            _R1_N[j_second]
            * bp_i[j_second]
            * _R1_J[j_second]
            * (_R1_J[j_second] - 1.0)
            * base_t ** (_R1_J[j_second] - 2.0)
        )
    )
    cross = i_pos & j_pos
    gamma_pi_tau = float(
        -np.sum(
            _R1_N[cross]
            * _R1_I[cross]
            * base_p ** (_R1_I[cross] - 1.0)
            * _R1_J[cross]
            * base_t ** (_R1_J[cross] - 1.0)
        )
    )
    v = pi * gamma_pi * R_KJ * temperature_k / pressure_mpa / 1000.0
    h = tau * gamma_tau * R_KJ * temperature_k
    s = R_KJ * (tau * gamma_tau - gamma)
    cp = -R_KJ * tau**2 * gamma_tau_tau
    cv = R_KJ * (-(tau**2) * gamma_tau_tau + (gamma_pi - tau * gamma_pi_tau) ** 2 / gamma_pi_pi)
    with np.errstate(invalid="ignore", divide="ignore"):
        sound_arg = (
            R_KJ
            * temperature_k
            * 1000.0
            * gamma_pi**2
            / ((gamma_pi - tau * gamma_pi_tau) ** 2 / (tau**2 * gamma_tau_tau) - gamma_pi_pi)
        )
        sound = np.sqrt(sound_arg) if sound_arg > 0.0 else np.nan
    return {
        "v": v,
        "h": h,
        "s": s,
        "cp": cp,
        "cv": cv,
        "w": float(sound),
        "u": h - pressure_mpa * v * 1000.0,
    }


def _region2_ideal(tau: float, pi: float) -> tuple:
    g0 = np.log(pi) + float(np.sum(_R2_N0 * tau**_R2_J0))
    g0_pi = 1.0 / pi
    g0_pi_pi = -1.0 / pi**2
    j_pos = _R2_J0 != 0.0
    g0_tau = float(np.sum(_R2_N0[j_pos] * _R2_J0[j_pos] * tau ** (_R2_J0[j_pos] - 1.0)))
    j_second = (_R2_J0 != 0.0) & (_R2_J0 != 1.0)
    g0_tau_tau = float(
        np.sum(
            _R2_N0[j_second]
            * _R2_J0[j_second]
            * (_R2_J0[j_second] - 1.0)
            * tau ** (_R2_J0[j_second] - 2.0)
        )
    )
    return g0, g0_pi, g0_pi_pi, g0_tau, g0_tau_tau


def _region2_raw(temperature_k: float, pressure_mpa: float) -> dict:
    tau = 540.0 / temperature_k
    pi = pressure_mpa
    base_t = tau - 0.5
    if pi <= 0.0 or base_t <= 0.0:
        raise DomainError("Аргумент области 2 IF97 покинул допустимый интервал")
    g0, g0_pi, g0_pi_pi, g0_tau, g0_tau_tau = _region2_ideal(tau, pi)
    gr = float(np.sum(_R2_N * pi**_R2_I * base_t**_R2_J))
    i_pos = _R2_I > 0.0
    gr_pi = float(
        np.sum(_R2_N[i_pos] * _R2_I[i_pos] * pi ** (_R2_I[i_pos] - 1.0) * base_t ** _R2_J[i_pos])
    )
    i_second = _R2_I > 1.0
    gr_pi_pi = float(
        np.sum(
            _R2_N[i_second]
            * _R2_I[i_second]
            * (_R2_I[i_second] - 1.0)
            * pi ** (_R2_I[i_second] - 2.0)
            * base_t ** _R2_J[i_second]
        )
    )
    j_pos = _R2_J != 0.0
    gr_tau = float(
        np.sum(_R2_N[j_pos] * pi ** _R2_I[j_pos] * _R2_J[j_pos] * base_t ** (_R2_J[j_pos] - 1.0))
    )
    j_second = (_R2_J != 0.0) & (_R2_J != 1.0)
    gr_tau_tau = float(
        np.sum(
            _R2_N[j_second]
            * pi ** _R2_I[j_second]
            * _R2_J[j_second]
            * (_R2_J[j_second] - 1.0)
            * base_t ** (_R2_J[j_second] - 2.0)
        )
    )
    cross = i_pos & j_pos
    gr_pi_tau = float(
        np.sum(
            _R2_N[cross]
            * _R2_I[cross]
            * pi ** (_R2_I[cross] - 1.0)
            * _R2_J[cross]
            * base_t ** (_R2_J[cross] - 1.0)
        )
    )
    gamma_pi = g0_pi + gr_pi
    v = pi * gamma_pi * R_KJ * temperature_k / pressure_mpa / 1000.0
    h = tau * (g0_tau + gr_tau) * R_KJ * temperature_k
    s = R_KJ * (tau * (g0_tau + gr_tau) - (g0 + gr))
    cp = -R_KJ * tau**2 * (g0_tau_tau + gr_tau_tau)
    cv = R_KJ * (
        -(tau**2) * (g0_tau_tau + gr_tau_tau)
        - (1.0 + pi * gr_pi - tau * pi * gr_pi_tau) ** 2 / (1.0 - pi**2 * gr_pi_pi)
    )
    numer = R_KJ * temperature_k * 1000.0 * (1.0 + 2.0 * pi * gr_pi + pi**2 * gr_pi**2)
    denom = (
        1.0
        - pi**2 * gr_pi_pi
        + (1.0 + pi * gr_pi - tau * pi * gr_pi_tau) ** 2 / (tau**2 * (g0_tau_tau + gr_tau_tau))
    )
    return {
        "v": v,
        "h": h,
        "s": s,
        "cp": cp,
        "cv": cv,
        "w": float(np.sqrt(numer / denom)),
        "u": h - pressure_mpa * v * 1000.0,
    }


def viscosity_pa_s(density_kg_m3: float, temperature_k: float) -> float:
    """Вязкость обычной воды, IAPWS 2008, без критической добавки."""
    if temperature_k <= 0.0 or density_kg_m3 < 0.0:
        raise DomainError("Вязкость запрошена при нефизичных rho или T")
    tr = temperature_k / TC_K
    dr = density_kg_m3 / RHOC
    mu0 = 100.0 * np.sqrt(tr) / sum(h / tr**i for i, h in enumerate(_MU_H0))
    series = 0.0
    for i_exp, j_exp, hij in zip(_MU_I, _MU_J, _MU_H):
        series += hij * (1.0 / tr - 1.0) ** i_exp * (dr - 1.0) ** j_exp
    mu1 = np.exp(dr * series)
    return float(mu0 * mu1 * 1.0e-6)


def conductivity_w_m_k(density_kg_m3: float, temperature_k: float) -> float:
    """Теплопроводность, IAPWS 2011, без критической добавки."""
    if temperature_k <= 0.0 or density_kg_m3 < 0.0:
        raise DomainError("Теплопроводность запрошена при нефизичных rho или T")
    tr = temperature_k / TC_K
    d_coef = density_kg_m3 / RHOC
    k0 = np.sqrt(tr) / sum(n / tr**i for i, n in enumerate(_K_N0))
    series = 0.0
    for i_exp, j_exp, nij in zip(_K_I, _K_J, _K_N):
        series += nij * (1.0 / tr - 1.0) ** i_exp * (d_coef - 1.0) ** j_exp
    return float((k0 * np.exp(d_coef * series)) * 1.0e-3)


def _pack(
    raw: dict, temperature_k: float, pressure_pa: float, quality: float, region: int
) -> ThermoState:
    rho = 1.0 / raw["v"]
    return ThermoState(
        T_k=temperature_k,
        p_pa=pressure_pa,
        rho_kg_m3=rho,
        v_m3_kg=raw["v"],
        u_j_kg=raw["u"] * 1000.0,
        h_j_kg=raw["h"] * 1000.0,
        s_j_kg_k=raw["s"] * 1000.0,
        cp_j_kg_k=raw["cp"] * 1000.0,
        cv_j_kg_k=raw["cv"] * 1000.0,
        sound_speed_m_s=raw["w"],
        viscosity_pa_s=viscosity_pa_s(rho, temperature_k),
        conductivity_w_m_k=conductivity_w_m_k(rho, temperature_k),
        quality=quality,
        region=region,
    )


_R3_N0 = 1.0658070028513
_R3_I = np.array(
    [
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 8, 9, 9, 10, 10, 11,
    ],
    dtype=float,
)
_R3_J = np.array(
    [
        0, 1, 2, 7, 10, 12, 23, 2, 6, 15, 17, 0, 2, 6, 7, 22, 26, 0, 2, 4, 16, 26, 0, 2, 4, 26, 1, 3, 26, 0, 2, 26, 2, 26, 2, 26, 0, 1, 26,
    ],
    dtype=float,
)
_R3_N = np.array(
    [
        -0.15732845290239e2,
        0.20944396974307e2,
        -0.76867707878716e1,
        0.26185947787954e1,
        -0.28080781148620e1,
        0.12053369696517e1,
        -0.84566812812502e-2,
        -0.12654315477714e1,
        -0.11524407806681e1,
        0.88521043984318,
        -0.64207765181607,
        0.38493460186671,
        -0.85214708824206,
        0.48972281541877e1,
        -0.30502617256965e1,
        0.39420536879154e-1,
        0.12558408424308,
        -0.27999329698710,
        0.13899799569460e1,
        -0.20189915023570e1,
        -0.82147637173963e-2,
        -0.47596035734923,
        0.43984074473500e-1,
        -0.44476435428739,
        0.90572070719733,
        0.70522450087967,
        0.10770512626332,
        -0.32913623258954,
        -0.50871062041158,
        -0.22175400873096e-1,
        0.94260751665092e-1,
        0.16436278447961,
        -0.13503372241348e-1,
        -0.14834345352472e-1,
        0.57922953628084e-3,
        0.32308904703711e-2,
        0.80964802996215e-4,
        -0.16557679795037e-3,
        -0.44923899061815e-4,
    ]
)
_R5_I = np.array([1, 1, 1, 2, 2, 3], dtype=float)
_R5_J = np.array([1, 2, 3, 3, 9, 7], dtype=float)
_R5_N = np.array(
    [
        0.15736404855259e-2,
        0.90153761673944e-3,
        -0.50270077677648e-2,
        0.22440037409485e-5,
        -0.41163275453471e-5,
        0.37919454822955e-7,
    ]
)
_R5_J0 = np.array([0, 1, -3, -2, -1, 2], dtype=float)
_R5_N0 = np.array(
    [
        -0.13179983674201e2,
        0.68540841634434e1,
        -0.24805148933466e-1,
        0.36901534980333,
        -0.31161318213925e1,
        -0.32961626538917,
    ]
)


def _region3_raw(density_kg_m3: float, temperature_k: float) -> dict:
    """Фундаментальное уравнение области 3: давление и энергия по плотности и температуре."""
    if density_kg_m3 <= 0.0 or temperature_k <= 0.0:
        raise DomainError("Область 3 запрошена при нефизичных rho или T")
    delta = density_kg_m3 / RHOC
    tau = TC_K / temperature_k
    phi = _R3_N0 * np.log(delta) + float(np.sum(_R3_N * delta**_R3_I * tau**_R3_J))
    i_pos = _R3_I > 0.0
    phi_d = _R3_N0 / delta + float(
        np.sum(_R3_N[i_pos] * _R3_I[i_pos] * delta ** (_R3_I[i_pos] - 1.0) * tau ** _R3_J[i_pos])
    )
    i_second = _R3_I > 1.0
    phi_dd = -_R3_N0 / delta**2 + float(
        np.sum(
            _R3_N[i_second]
            * _R3_I[i_second]
            * (_R3_I[i_second] - 1.0)
            * delta ** (_R3_I[i_second] - 2.0)
            * tau ** _R3_J[i_second]
        )
    )
    j_pos = _R3_J != 0.0
    phi_t = float(np.sum(_R3_N[j_pos] * _R3_J[j_pos] * delta ** _R3_I[j_pos] * tau ** (_R3_J[j_pos] - 1.0)))
    j_second = (_R3_J != 0.0) & (_R3_J != 1.0)
    phi_tt = float(
        np.sum(
            _R3_N[j_second]
            * _R3_J[j_second]
            * (_R3_J[j_second] - 1.0)
            * delta ** _R3_I[j_second]
            * tau ** (_R3_J[j_second] - 2.0)
        )
    )
    cross = i_pos & j_pos
    phi_dt = float(
        np.sum(
            _R3_N[cross]
            * _R3_I[cross]
            * _R3_J[cross]
            * delta ** (_R3_I[cross] - 1.0)
            * tau ** (_R3_J[cross] - 1.0)
        )
    )
    p_mpa = delta * phi_d * R_KJ * temperature_k * density_kg_m3 / 1000.0
    h = R_KJ * temperature_k * (tau * phi_t + delta * phi_d)
    s = R_KJ * (tau * phi_t - phi)
    cv = -R_KJ * tau**2 * phi_tt
    cp_den = 2.0 * delta * phi_d + delta**2 * phi_dd
    cp = R_KJ * (-tau**2 * phi_tt + (delta * phi_d - delta * tau * phi_dt) ** 2 / cp_den)
    sound_arg = R_KJ * temperature_k * 1000.0 * (
        2.0 * delta * phi_d + delta**2 * phi_dd - (delta * phi_d - delta * tau * phi_dt) ** 2 / (tau**2 * phi_tt)
    )
    with np.errstate(invalid="ignore"):
        sound = float(np.sqrt(sound_arg)) if sound_arg > 0.0 else float("nan")
    v = 1.0 / density_kg_m3
    return {
        "v": v,
        "h": h,
        "s": s,
        "cp": cp,
        "cv": cv,
        "w": sound,
        "u": h - p_mpa * v * 1000.0,
        "p_mpa": p_mpa,
    }


def _invert_region3_density(pressure_mpa: float, temperature_k: float) -> float:
    def residual(density: float) -> float:
        return _region3_raw(density, temperature_k)["p_mpa"] - pressure_mpa

    grid = np.geomspace(5.0, 900.0, 24)
    values = [residual(float(density)) for density in grid]
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if f_left == 0.0:
            return float(left)
        if f_left * f_right < 0.0:
            return _brent(lambda value: residual(value), float(left), float(right))
    raise DomainError("Плотность области 3 не найдена при заданных p и T")


def _region3_from_pressure(temperature_k: float, pressure_mpa: float) -> dict:
    density = _invert_region3_density(pressure_mpa, temperature_k)
    raw = _region3_raw(density, temperature_k)
    raw["quality"] = 0.0 if temperature_k < TC_K and pressure_mpa < PC_MPA else 1.0
    return raw


def _region5_raw(temperature_k: float, pressure_mpa: float) -> dict:
    tau = 1000.0 / temperature_k
    pi = pressure_mpa
    if pi <= 0.0:
        raise DomainError("Область 5 запрошена при неположительном давлении")
    j_pos = _R5_J0 != 0.0
    j_second = (_R5_J0 != 0.0) & (_R5_J0 != 1.0)
    gamma0 = np.log(pi) + float(np.sum(_R5_N0 * tau**_R5_J0))
    gamma0_pi = 1.0 / pi
    gamma0_pi_pi = -1.0 / pi**2
    gamma0_tau = float(np.sum(_R5_N0[j_pos] * _R5_J0[j_pos] * tau ** (_R5_J0[j_pos] - 1.0)))
    gamma0_tau_tau = float(
        np.sum(_R5_N0[j_second] * _R5_J0[j_second] * (_R5_J0[j_second] - 1.0) * tau ** (_R5_J0[j_second] - 2.0))
    )
    gamma_r = float(np.sum(_R5_N * pi**_R5_I * tau**_R5_J))
    gamma_r_pi = float(np.sum(_R5_N * _R5_I * pi ** (_R5_I - 1.0) * tau**_R5_J))
    i_second = _R5_I > 1.0
    gamma_r_pi_pi = float(
        np.sum(_R5_N[i_second] * _R5_I[i_second] * (_R5_I[i_second] - 1.0) * pi ** (_R5_I[i_second] - 2.0) * tau ** _R5_J[i_second])
    )
    gamma_r_tau = float(np.sum(_R5_N * _R5_J * pi**_R5_I * tau ** (_R5_J - 1.0)))
    j_r = (_R5_J != 0.0) & (_R5_J != 1.0)
    gamma_r_tau_tau = float(
        np.sum(_R5_N[j_r] * _R5_J[j_r] * (_R5_J[j_r] - 1.0) * pi ** _R5_I[j_r] * tau ** (_R5_J[j_r] - 2.0))
    )
    gamma_r_pi_tau = float(np.sum(_R5_N * _R5_I * _R5_J * pi ** (_R5_I - 1.0) * tau ** (_R5_J - 1.0)))
    gamma_pi = gamma0_pi + gamma_r_pi
    v = pi * gamma_pi * R_KJ * temperature_k / pressure_mpa / 1000.0
    h = tau * (gamma0_tau + gamma_r_tau) * R_KJ * temperature_k
    s = R_KJ * (tau * (gamma0_tau + gamma_r_tau) - (gamma0 + gamma_r))
    cp = -R_KJ * tau**2 * (gamma0_tau_tau + gamma_r_tau_tau)
    cv = R_KJ * (
        -tau**2 * (gamma0_tau_tau + gamma_r_tau_tau)
        + (gamma_pi - tau * gamma_r_pi_tau) ** 2 / (gamma0_pi_pi + gamma_r_pi_pi)
    )
    numer = R_KJ * temperature_k * 1000.0 * (1.0 + 2.0 * pi * gamma_r_pi + pi**2 * gamma_r_pi**2)
    denom = 1.0 - pi**2 * gamma_r_pi_pi + (1.0 + pi * gamma_r_pi - tau * pi * gamma_r_pi_tau) ** 2 / (
        tau**2 * (gamma0_tau_tau + gamma_r_tau_tau)
    )
    return {"v": v, "h": h, "s": s, "cp": cp, "cv": cv, "w": float(np.sqrt(numer / denom)), "u": h - pressure_mpa * v * 1000.0}


def properties_pt(pressure_pa: float, temperature_k: float) -> ThermoState:
    """Свойства однофазной воды по давлению и температуре. Без экстраполяции."""
    p_mpa = pressure_pa * 1.0e-6
    region = classify_region(temperature_k, p_mpa)
    if region == 1:
        raw = _region1_raw(temperature_k, p_mpa)
        quality = 0.0
    elif region == 2:
        raw = _region2_raw(temperature_k, p_mpa)
        quality = 1.0
    elif region == 3:
        raw = _region3_from_pressure(temperature_k, p_mpa)
        quality = raw.pop("quality")
    else:
        raw = _region5_raw(temperature_k, p_mpa)
        quality = 1.0
    return _pack(raw, temperature_k, pressure_pa, quality, region)


def saturation_properties(pressure_pa: float) -> tuple:
    """Насыщенная жидкость и пар. Возвращает (liquid, vapor)."""
    p_mpa = pressure_pa * 1.0e-6
    temperature = saturation_temperature_k(p_mpa)
    if temperature > T_REGION1_MAX:
        raise DomainError(
            "Плотности насыщения в области 3 не заданы отдельным замыканием; однофазная область 3 доступна через properties_pt"
        )
    liquid = _pack(_region1_raw(temperature, p_mpa), temperature, pressure_pa, 0.0, 1)
    vapor = _pack(_region2_raw(temperature, p_mpa), temperature, pressure_pa, 1.0, 2)
    return liquid, vapor


def _enthalpy_residual(
    temperature_k: float, pressure_mpa: float, enthalpy_kj: float, region: int
) -> float:
    if region == 1:
        raw = _region1_raw(temperature_k, pressure_mpa)
    elif region == 2:
        raw = _region2_raw(temperature_k, pressure_mpa)
    elif region == 3:
        raw = _region3_from_pressure(temperature_k, pressure_mpa)
    elif region == 5:
        raw = _region5_raw(temperature_k, pressure_mpa)
    else:
        raise DomainError("Область %s не имеет обратного p,h" % region)
    return raw["h"] - enthalpy_kj


def _temperature_in_region(
    pressure_mpa: float, enthalpy_kj: float, region: int, low: float, high: float
) -> float:
    def residual(temperature: float) -> float:
        return _enthalpy_residual(temperature, pressure_mpa, enthalpy_kj, region)

    return _brent(residual, low, high)


def properties_ph(pressure_pa: float, enthalpy_j_kg: float) -> ThermoState:
    """Состояние по давлению и энтальпии, включая области 3 и 5."""
    p_mpa = pressure_pa * 1.0e-6
    h_kj = enthalpy_j_kg / 1000.0
    if p_mpa <= P_MIN_MPA or p_mpa > 100.0:
        raise DomainError("Давление вне поддерживаемого диапазона IF97")
    if p_mpa <= P_REGION13_MPA:
        liquid, vapor = saturation_properties(pressure_pa)
        hf = liquid.h_j_kg / 1000.0
        hg = vapor.h_j_kg / 1000.0
        if hf <= h_kj <= hg:
            quality = 0.0 if hg == hf else (h_kj - hf) / (hg - hf)
            return _mix_saturation(liquid, vapor, quality)
        if h_kj < hf:
            temperature = _temperature_in_region(p_mpa, h_kj, 1, T_MIN + 1e-4, liquid.T_k)
            return properties_pt(pressure_pa, temperature)
        try:
            temperature = _temperature_in_region(p_mpa, h_kj, 2, vapor.T_k, T_REGION2_MAX - 1.0)
            return properties_pt(pressure_pa, temperature)
        except DomainError:
            temperature = _temperature_in_region(p_mpa, h_kj, 5, 1073.15, 2273.15)
            return properties_pt(pressure_pa, temperature)
    try:
        temperature = _temperature_in_region(p_mpa, h_kj, 1, T_MIN + 1e-4, T_REGION1_MAX)
        state = properties_pt(pressure_pa, temperature)
        enthalpy_gap = abs(state.h_j_kg - enthalpy_j_kg)
        if state.region == 1 and enthalpy_gap <= 1.0 + 1e-6 * abs(enthalpy_j_kg):
            return state
    except DomainError:
        pass
    lows = []
    for temperature in np.linspace(623.2, 863.0, 25):
        try:
            if classify_region(float(temperature), p_mpa) == 3:
                lows.append(float(temperature))
        except DomainError:
            continue
    if len(lows) < 2:
        raise DomainError("p,h не попали в реализованную однофазную область")
    temperature = _temperature_in_region(p_mpa, h_kj, 3, lows[0], lows[-1])
    return properties_pt(pressure_pa, temperature)


def properties_pu(pressure_pa: float, internal_energy_j_kg: float) -> ThermoState:
    liquid, vapor = saturation_properties(pressure_pa)
    if liquid.u_j_kg <= internal_energy_j_kg <= vapor.u_j_kg:
        quality = (internal_energy_j_kg - liquid.u_j_kg) / (vapor.u_j_kg - liquid.u_j_kg)
        return _mix_saturation(liquid, vapor, quality)

    # Однофазный поиск температуры при фиксированном давлении.
    def residual(temperature: float) -> float:
        return properties_pt(pressure_pa, temperature).u_j_kg - internal_energy_j_kg

    if internal_energy_j_kg < liquid.u_j_kg:
        temperature = _brent(residual, T_MIN + 1e-6, liquid.T_k)
        return properties_pt(pressure_pa, temperature)
    try:
        temperature = _brent(residual, vapor.T_k, min(T_REGION2_MAX - 1.0, vapor.T_k + 400.0))
        return properties_pt(pressure_pa, temperature)
    except DomainError:
        temperature = _brent(residual, 1073.15, 2273.15)
        return properties_pt(pressure_pa, temperature)


def _project_equilibrium(
    density_kg_m3: float, internal_energy_j_kg: float
) -> ThermoState:
    """Ближайшее равновесное состояние, если пробная точка якобиана вышла из купола.

    Масса и энергия интегратора не переписываются. Несовпадение сохраняется в полях
    состояния и проверяется на принятом шаге.
    """
    specific_volume = 1.0 / density_kg_m3
    best = None
    best_score = None
    for pressure in np.geomspace(P_MIN_MPA * 1.5, P_REGION13_MPA * 0.98, 32):
        liquid, vapor = saturation_properties(float(pressure) * 1.0e6)
        span = vapor.u_j_kg - liquid.u_j_kg
        quality = (internal_energy_j_kg - liquid.u_j_kg) / span
        quality = min(1.0, max(0.0, quality))
        mixture = _mix_saturation(liquid, vapor, quality)
        score = ((mixture.v_m3_kg - specific_volume) / specific_volume) ** 2 + (
            (mixture.u_j_kg - internal_energy_j_kg) / 1.0e6
        ) ** 2
        if best_score is None or score < best_score:
            best_score = score
            best = mixture
    if best is None:
        raise DomainError("Не найдено ближайшее равновесное состояние")
    return ThermoState(
        **{
            **best.__dict__,
            "energy_mismatch_j_kg": internal_energy_j_kg - best.u_j_kg,
            "volume_mismatch": specific_volume - best.v_m3_kg,
            "applicability": Applicability.OUT_OF_DOMAIN
            if abs(internal_energy_j_kg - best.u_j_kg) > 5.0e3
            or abs(specific_volume - best.v_m3_kg) / specific_volume > 0.02
            else Applicability.IN_DOMAIN,
        }
    )


def properties_rho_u(
    density_kg_m3: float,
    internal_energy_j_kg: float,
    pressure_guess_pa: float,
    temperature_guess_k: float,
    allow_projection: bool = False,
) -> ThermoState:
    """Обратное (rho, u). Проекция на купол только для численной пробы, не для принятого шага."""
    if density_kg_m3 <= 0.0:
        raise DomainError("Плотность должна быть положительной")
    v_target = 1.0 / density_kg_m3
    p_mpa = pressure_guess_pa * 1.0e-6
    if P_MIN_MPA < p_mpa < P_REGION13_MPA:
        liquid, vapor = saturation_properties(pressure_guess_pa)
        if liquid.u_j_kg <= internal_energy_j_kg <= vapor.u_j_kg:
            two_phase = _invert_two_phase(v_target, internal_energy_j_kg, pressure_guess_pa)
            if two_phase is not None:
                return two_phase
    try:
        return _invert_single_phase(
            v_target, internal_energy_j_kg, pressure_guess_pa, temperature_guess_k
        )
    except DomainError:
        two_phase = _invert_two_phase(v_target, internal_energy_j_kg, pressure_guess_pa)
        if two_phase is not None:
            return two_phase
        try:
            return _invert_region3_rho_u(density_kg_m3, internal_energy_j_kg)
        except DomainError:
            if allow_projection:
                return _project_equilibrium(density_kg_m3, internal_energy_j_kg)
            raise DomainError(
                "Обращение (rho, u) не сошлось; проекция разрешена только для численной пробы"
            )


def _invert_region3_rho_u(density_kg_m3: float, internal_energy_j_kg: float) -> ThermoState:
    """Область 3 задана как (rho, T). Ищем T и проверяем классификацию области."""
    target_kj = internal_energy_j_kg / 1000.0

    def residual(temperature: float) -> float:
        return _region3_raw(density_kg_m3, temperature)["u"] - target_kj

    temperature = _brent(residual, 623.2, 863.15)
    raw = _region3_raw(density_kg_m3, temperature)
    if classify_region(temperature, raw["p_mpa"]) != 3:
        raise DomainError("Найденная точка (rho, u) не лежит в области 3")
    state = _pack(raw, temperature, raw["p_mpa"] * 1.0e6, 0.0, 3)
    if abs(state.u_j_kg - internal_energy_j_kg) > 1.0 + 1e-5 * abs(internal_energy_j_kg):
        raise DomainError("Обращение области 3 не сошлось по энергии")
    if abs(state.rho_kg_m3 - density_kg_m3) / density_kg_m3 > 1e-6:
        raise DomainError("Обращение области 3 не сошлось по плотности")
    return state


def _mix_saturation(liquid: ThermoState, vapor: ThermoState, quality: float) -> ThermoState:
    if quality < -1e-10 or quality > 1.0 + 1e-10:
        raise DomainError("Паросодержание вне [0, 1]")
    quality = float(np.clip(quality, 0.0, 1.0))
    v = (1.0 - quality) * liquid.v_m3_kg + quality * vapor.v_m3_kg
    u = (1.0 - quality) * liquid.u_j_kg + quality * vapor.u_j_kg
    h = (1.0 - quality) * liquid.h_j_kg + quality * vapor.h_j_kg
    s = (1.0 - quality) * liquid.s_j_kg_k + quality * vapor.s_j_kg_k
    rho = 1.0 / v
    # cp смеси в двухфазной области вдоль изобары бесконечен в равновесной модели;
    # для теплоёмкости жидкости/пара вне скачка используем массовое среднее только как
    # справочную величину однофазных ветвей и не применяем его как dH/dT насыщения.
    cp = (1.0 - quality) * liquid.cp_j_kg_k + quality * vapor.cp_j_kg_k
    cv = (1.0 - quality) * liquid.cv_j_kg_k + quality * vapor.cv_j_kg_k
    sound = (1.0 - quality) * liquid.sound_speed_m_s + quality * vapor.sound_speed_m_s
    return ThermoState(
        T_k=liquid.T_k,
        p_pa=liquid.p_pa,
        rho_kg_m3=rho,
        v_m3_kg=v,
        u_j_kg=u,
        h_j_kg=h,
        s_j_kg_k=s,
        cp_j_kg_k=cp,
        cv_j_kg_k=cv,
        sound_speed_m_s=sound,
        viscosity_pa_s=(1.0 - quality) * liquid.viscosity_pa_s + quality * vapor.viscosity_pa_s,
        conductivity_w_m_k=(1.0 - quality) * liquid.conductivity_w_m_k
        + quality * vapor.conductivity_w_m_k,
        quality=quality,
        region=4,
    )


def void_fraction(quality: float, rho_liquid: float, rho_vapor: float) -> float:
    """Объёмное паросодержание HEM. Формула не продолжается вне x∈[0,1]."""
    if quality < 0.0 or quality > 1.0:
        raise DomainError("Объёмное паросодержание HEM определено только для x в [0, 1]")
    if quality == 0.0:
        return 0.0
    if quality == 1.0:
        return 1.0
    return (quality / rho_vapor) / (quality / rho_vapor + (1.0 - quality) / rho_liquid)


def _invert_temperature(
    pressure_mpa: float, enthalpy_kj: float, guess: float, region: int
) -> float:
    def residual(temperature: float) -> float:
        raw = (
            _region1_raw(temperature, pressure_mpa)
            if region == 1
            else _region2_raw(temperature, pressure_mpa)
        )
        return raw["h"] - enthalpy_kj

    if region == 1:
        low, high = T_MIN + 1e-4, T_REGION1_MAX
    else:
        low, high = T_MIN + 1e-4, T_REGION2_MAX - 1.0
    # Сдвигаем границы внутрь области, где формула определена при данном p.
    try:
        return _brent(residual, low, high)
    except DomainError:
        return _brent(residual, max(low, guess - 80.0), min(high, guess + 80.0))


def _invert_two_phase(
    specific_volume: float, internal_energy_j_kg: float, pressure_guess_pa: float
) -> ThermoState | None:
    def residual(p_mpa: float):
        liquid, vapor = saturation_properties(p_mpa * 1.0e6)
        span = vapor.u_j_kg - liquid.u_j_kg
        quality = (internal_energy_j_kg - liquid.u_j_kg) / span
        if quality < -1e-6 or quality > 1.0 + 1e-6:
            return None
        v_mix = (1.0 - quality) * liquid.v_m3_kg + quality * vapor.v_m3_kg
        return v_mix - specific_volume

    guess = min(max(pressure_guess_pa * 1.0e-6, P_MIN_MPA * 2.0), P_REGION13_MPA * 0.999)
    value = residual(guess)
    if value is not None and abs(value) < 1e-8 * specific_volume:
        return properties_pu(guess * 1.0e6, internal_energy_j_kg)
    if value is not None:
        step = 1.0e-4 * guess
        for _ in range(12):
            other = residual(guess + step)
            if other is None or other == value:
                break
            guess_next = guess - value * step / (other - value)
            guess_next = min(max(guess_next, P_MIN_MPA * 1.5), P_REGION13_MPA * 0.999)
            value_next = residual(guess_next)
            if value_next is None:
                break
            if abs(value_next) < 1e-8 * specific_volume:
                return properties_pu(guess_next * 1.0e6, internal_energy_j_kg)
            step = guess_next - guess
            guess, value = guess_next, value_next
    return _bracket_two_phase(residual, specific_volume, internal_energy_j_kg)


def _bracket_two_phase(residual, specific_volume: float, internal_energy_j_kg: float):
    """Запасной поиск давления, если секущая срывается у границы купола."""
    pressures = np.geomspace(P_MIN_MPA * 1.5, P_REGION13_MPA * 0.98, 28)
    grid = []
    values = []
    for pressure in pressures:
        value = residual(float(pressure))
        if value is None or not np.isfinite(value):
            continue
        grid.append(float(pressure))
        values.append(float(value))
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if f_left == 0.0 or abs(f_left) < 1e-8 * specific_volume:
            return properties_pu(left * 1.0e6, internal_energy_j_kg)
        if f_left * f_right < 0.0:
            root = _brent(residual, left, right)
            return properties_pu(root * 1.0e6, internal_energy_j_kg)
    return None


def _invert_single_phase(
    specific_volume: float,
    internal_energy_j_kg: float,
    pressure_guess_pa: float,
    temperature_guess_k: float,
) -> ThermoState:
    from scipy.optimize import root

    u_kj = internal_energy_j_kg / 1000.0

    def residual(x: np.ndarray) -> np.ndarray:
        p_mpa, temperature = float(x[0]), float(x[1])
        try:
            region = classify_region(temperature, p_mpa)
            if region == 1:
                raw = _region1_raw(temperature, p_mpa)
            elif region == 2:
                raw = _region2_raw(temperature, p_mpa)
            elif region == 5:
                raw = _region5_raw(temperature, p_mpa)
            else:
                return np.array([1.0e6, 1.0e6])
        except DomainError:
            return np.array([1.0e6, 1.0e6])
        return np.array([raw["v"] - specific_volume, raw["u"] - u_kj])

    guesses = [
        np.array([pressure_guess_pa * 1e-6, temperature_guess_k]),
        np.array([pressure_guess_pa * 1e-6, max(280.0, temperature_guess_k - 40.0)]),
        np.array([max(pressure_guess_pa * 1e-6, 0.01), min(620.0, temperature_guess_k + 40.0)]),
        np.array([max(pressure_guess_pa * 1e-6, 0.1), max(temperature_guess_k, 1500.0)]),
        np.array([0.5, 1500.0]),
    ]
    solved = None
    for guess in guesses:
        candidate = root(residual, guess, method="hybr")
        if candidate.success:
            solved = candidate
            break
    if solved is None:
        raise DomainError("Не удалось обратить (rho, u) в поддерживаемой области IF97")
    p_mpa, temperature = float(solved.x[0]), float(solved.x[1])
    state = properties_pt(p_mpa * 1e6, temperature)
    if abs(state.v_m3_kg - specific_volume) / specific_volume > 1e-6:
        raise DomainError("Обращение (rho, u) не сошлось к заданной плотности")
    if abs(state.u_j_kg - internal_energy_j_kg) > 1.0 + 1e-6 * abs(internal_energy_j_kg):
        raise DomainError("Обращение (rho, u) не сошлось к заданной энергии")
    return state


def _brent(func, a: float, b: float) -> float:
    fa = float(func(a))
    fb = float(func(b))
    if not np.isfinite(fa) or not np.isfinite(fb) or fa * fb > 0.0:
        raise DomainError("Корень обратного преобразования не локализован")
    for _ in range(80):
        mid = 0.5 * (a + b)
        fm = float(func(mid))
        if abs(fm) < 1e-10 or abs(b - a) < 1e-12:
            return mid
        if fa * fm <= 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)
