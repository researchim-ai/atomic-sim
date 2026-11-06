"""Closed-form/independent-oracle checks; synthetic cases never count as experiments."""

from atomic_model_hem_assessment.audit import dataset
from atomic_model_hem_throughflow.benchmark import evaluate as production_benchmark
from atomic_model_workspace.common import digest, keys, read

from . import precision, prediction, reference, region
from .workflow import agree, identity

CRITERIA = dict(
    bound_error=2e-6,
    sensitivity_relative_error=2e-4,
    sensitivity_absolute_error=2e-6,
    root_bracket_width_k=1e-9,
    root_residual_w_m2=0.001,
    precision_scale_bracket_width=1e-9,
)


def cases():
    def obs(name, low, high, values):
        return dict(id=name, lower_basis=low, upper_basis=high, observed_q_bounds=values)

    return [
        (
            "rectangle",
            [obs("a", [1, 0], [1, 0], [1, 2]), obs("b", [0, 1], [0, 1], [3, 4])],
            "bounded",
            [(1, 2), (3, 4)],
        ),
        (
            "unbounded_wedge",
            [obs("a", [0, 0], [1, 2], [3, 4])],
            "unbounded",
            [(0, None), (0, None)],
        ),
        (
            "infeasible",
            [obs("a", [1, 0], [1, 0], [0, 1]), obs("b", [1, 0], [1, 0], [2, 3])],
            "infeasible",
            None,
        ),
        (
            "rank_deficient_but_bounded",
            [obs("a", [1, 1], [1, 1], [1, 2])],
            "bounded",
            [(0, 2), (0, 2)],
        ),
        ("zero_response", [obs("a", [0, 0], [0, 0], [1, 2])], "infeasible", None),
        ("one_free_gain", [obs("a", [1, 0], [1, 0], [2, 3])], "unbounded", [(2, 3), (0, None)]),
        (
            "small_rectangle",
            [obs("a", [1, 0], [1, 0], [0.001, 0.002]), obs("b", [0, 1], [0, 1], [0.003, 0.004])],
            "bounded",
            [(0.001, 0.002), (0.003, 0.004)],
        ),
    ]


def qualify():
    source = identity()
    records = []
    for name, observations, status, expected in cases():
        result = region.solve(observations)
        A, b = region.constraints(observations)
        oracle = reference.enumerate_region(A, b)
        agree(result, oracle)
        if result["status"] != status:
            raise ValueError("Closed-form status mismatch")
        if expected is not None:
            for actual, (lo, hi) in zip(result["bounds"], expected):
                if abs(actual["lower"] - lo) > CRITERIA["bound_error"] or (
                    actual["upper"] is None
                ) != (hi is None):
                    raise ValueError("Closed-form bound mismatch")
                if hi is not None and abs(actual["upper"] - hi) > CRITERIA["bound_error"]:
                    raise ValueError("Closed-form upper mismatch")
        records.append(
            dict(
                name=name,
                kind="synthetic numerical solver oracle; not physical experiment",
                result=result,
                reference=oracle,
            )
        )
    _, data = dataset()
    by_id = {r["row_id"]: r for r in data}
    selected = [by_id[r["row_id"]] for r in production_benchmark()["rows"] if r["candidate"]]
    maxima = dict(
        relative_sensitivity_error=0.0,
        absolute_sensitivity_error=0.0,
        root_width_k=0.0,
        root_residual_w_m2=0.0,
    )
    for r in selected:
        p, x, G, D, q = [
            r[k]
            for k in ("pressure_pa", "quality", "mass_flux_kg_m2_s", "diameter_m", "heat_flux_w_m2")
        ]
        L = r["length_over_diameter"] * D
        out = prediction.predict(p, x, G, D, q, L)
        maxima["root_width_k"] = max(maxima["root_width_k"], out["root_bracket_width_k"])
        maxima["root_residual_w_m2"] = max(
            maxima["root_residual_w_m2"], abs(out["heat_flux_residual_w_m2"])
        )
        for j in (0, 1):
            a = [1.0, 1.0]
            b = [1.0, 1.0]
            a[j] += 0.001
            b[j] -= 0.001
            fd = (
                prediction.predict(p, x, G, D, q, L, a)["wall_temperature_k"]
                - prediction.predict(p, x, G, D, q, L, b)["wall_temperature_k"]
            ) / 0.002
            analytic = out["sensitivity_k_per_gain"][j]
            error = abs(fd - analytic)
            maxima["absolute_sensitivity_error"] = max(maxima["absolute_sensitivity_error"], error)
            maxima["relative_sensitivity_error"] = max(
                maxima["relative_sensitivity_error"], error / max(abs(analytic), 1e-12)
            )
            if error > CRITERIA["sensitivity_absolute_error"] + CRITERIA[
                "sensitivity_relative_error"
            ] * abs(analytic):
                raise ValueError("Analytic/FD gain sensitivity mismatch")
        requirement = precision.requirement(r)
        if requirement["maximum_joint_scale"] is not None:
            lo = requirement["maximum_joint_scale"]
            hi = requirement["rejected_scale_upper"]
            width = precision.interval(r, lo)["relative_width_from_lower"]
            if width is None or width > 0.2 or hi - lo > CRITERIA["precision_scale_bracket_width"]:
                raise ValueError("Measurement precision lower certificate")
            if hi > lo:
                outside = precision.interval(r, hi)["relative_width_from_lower"]
                if outside is not None and outside <= 0.2:
                    raise ValueError("Measurement precision upper certificate")
    if (
        maxima["root_width_k"] > CRITERIA["root_bracket_width_k"]
        or maxima["root_residual_w_m2"] > CRITERIA["root_residual_w_m2"]
    ):
        raise ValueError("Numerical root qualification")
    if identity() != source:
        raise ValueError("Qualification source changed")
    return dict(
        schema_version=1,
        source=source,
        criteria=CRITERIA,
        synthetic_solver_cases=records,
        primary_states_checked=len(selected),
        analytic_fd_sensitivities_checked=2 * len(selected),
        maxima=maxima,
        physical_acceptance=False,
        validated=False,
    )


def audit(path):
    record = read(path)
    keys(record, ("payload", "sha256"))
    if record["sha256"] != digest(record["payload"]) or record["payload"]["source"] != identity():
        raise ValueError("Numerical qualification digest/source mismatch")
    expected = qualify()
    if digest(expected) != record["sha256"]:
        raise ValueError("Numerical qualification replay mismatch")
    return expected
