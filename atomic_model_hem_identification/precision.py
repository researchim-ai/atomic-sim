"""Worst-case reported-box precision requirements, not confidence intervals."""

from atomic_model_hem_qualified import eos
from atomic_model_workspace.common import number


def interval(row, scale):
    scale = number(scale, 0, 1, "reported p/Twall bound scale")
    for field in ("pressure_pa", "wall_temperature_k", "heat_flux_w_m2"):
        values = [
            number(row[field + suffix], 1e-12, 1e12, field + suffix)
            for suffix in ("_lower", "", "_upper")
        ]
        if not values[0] <= values[1] <= values[2]:
            raise ValueError("Reversed or non-enclosing reported interval")
    p = row["pressure_pa"]
    t = row["wall_temperature_k"]
    lo = p + scale * (row["pressure_pa_lower"] - p)
    hi = p + scale * (row["pressure_pa_upper"] - p)
    if not eos.P_MIN <= lo <= hi <= eos.P_MAX:
        raise ValueError("Pressure uncertainty leaves EOS domain")
    tl = t + scale * (row["wall_temperature_k_lower"] - t)
    th = t + scale * (row["wall_temperature_k_upper"] - t)
    dl = tl - eos.px(hi, 0.0).temperature_k
    dh = th - eos.px(lo, 0.0).temperature_k
    if dh <= 0:
        raise ValueError("No positive-superheat observational intersection")
    lower = row["heat_flux_w_m2_lower"] / dh
    upper = None if dl <= 0 else row["heat_flux_w_m2_upper"] / dl
    return dict(
        superheat_k=[dl, dh],
        htc_lower_w_m2_k=lower,
        htc_upper_w_m2_k=upper,
        relative_width_from_lower=None if upper is None else upper / lower - 1,
    )


def requirement(row, relative_width=0.2):
    target = number(relative_width, 1e-6, 10, "relative HTC box width target")
    current = interval(row, 1.0)
    perfect = interval(row, 0.0)
    if (
        perfect["relative_width_from_lower"] is None
        or perfect["relative_width_from_lower"] > target
    ):
        return dict(
            status="heat_flux_uncertainty_floor_exceeds_target",
            current=current,
            perfect_p_and_twall=perfect,
            maximum_joint_scale=None,
            physical_acceptance=False,
        )

    def allowed(scale):
        width = interval(row, scale)["relative_width_from_lower"]
        return width is not None and width <= target

    if allowed(1.0):
        lo, hi = 1.0, 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(48):
            mid = (lo + hi) / 2
            if allowed(mid):
                lo = mid
            else:
                hi = mid
            if hi - lo <= 1e-9:
                break
    return dict(
        status="conditional_precision_requirement",
        current=current,
        perfect_p_and_twall=perfect,
        target_relative_width=target,
        maximum_joint_scale=lo,
        rejected_scale_upper=hi,
        required_wall_halfwidth_k=(
            row["wall_temperature_k_upper"] - row["wall_temperature_k_lower"]
        )
        / 2
        * lo,
        required_pressure_halfwidth_pa=(row["pressure_pa_upper"] - row["pressure_pa_lower"])
        / 2
        * lo,
        reduction_factor=None if lo == 0 else 1 / lo,
        meaning="Hypothetical joint scaling of published p/Twall endpoints about nominal; q endpoints fixed; worst-case box, no covariance/confidence assumption",
        physical_acceptance=False,
    )
