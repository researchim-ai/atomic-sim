"""Numerical qualification and exact, source-bound local experimental assessment."""

from copy import deepcopy

from atomic_model_workspace.common import digest, keys, read

from . import benchmark, cases, reference
from .model import SteadyChannel, signature

CRITERIA = dict(
    pressure_error_pa=0.1,
    enthalpy_error_j_kg=0.001,
    energy_balance_j_kg=1e-5,
    momentum_balance_pa=0.001,
    corrected_entropy_balance_j_kg_k=1e-5,
    phase_root_residual_j_kg=2e-6,
    maximum_spatial_refinement_ratio=0.2,
    minimum_resolved_spatial_error_pa=1e-6,
)
CAPS = (0.1, 0.05, 0.025)


def execute(spec, targets, checkpoint=None):
    model = SteadyChannel(spec)
    if checkpoint is not None:
        model.restore(checkpoint)
    observations = [model.snapshot()]
    checkpoints = []
    for target in targets:
        observations.append(model.advance_to(target))
        cp = model.checkpoint()
        other = SteadyChannel(spec)
        other.restore(cp)
        if other.checkpoint() != cp:
            raise ValueError("Exact steady restart failed")
        checkpoints.append(cp["sha256"])
    return dict(
        observations=observations,
        checkpoints=checkpoints,
        final_checkpoint=model.checkpoint(),
        contract=model.contract(),
    )


def refinement():
    spec = next(s for n, s in cases.matrix() if n == "transition_friction")
    results = {}
    for cap in (0.4, 0.2, 0.1, 0.05, 0.025):
        case = deepcopy(spec)
        case["max_step_m"] = cap
        m = SteadyChannel(case)
        out = m.advance_to(case["length_m"])
        results[str(cap)] = out["state"]["pressure_pa"]
    finest = results["0.025"]
    errors = {k: abs(v - finest) for k, v in results.items() if k != "0.025"}
    ratios = [
        errors[str(b)] / errors[str(a)]
        for a, b in ((0.4, 0.2), (0.2, 0.1), (0.1, 0.05))
        if errors[str(a)] > CRITERIA["minimum_resolved_spatial_error_pa"]
    ]
    if not ratios or any(r > CRITERIA["maximum_spatial_refinement_ratio"] for r in ratios):
        raise ValueError(f"Steady spatial convergence failed: {errors}")
    return dict(
        kind="Smooth steady ODE spatial integration, not transient finite-volume PDE mesh qualification",
        case="transition_friction",
        pressure_pa=results,
        errors_to_finest_pa=errors,
        refinement_ratios=ratios,
    )


def qualify():
    source = signature()
    records = []
    for name, spec in cases.matrix():
        targets = [spec["length_m"] * 0.2, spec["length_m"] * 0.5, spec["length_m"]]
        refs = [reference.solve(spec, targets, method) for method in ("DOP853", "Radau")]
        outputs = []
        for cap in CAPS:
            request = deepcopy(spec)
            request["max_step_m"] = cap
            out = execute(request, targets)
            measured = {k: 0.0 for k in ("pressure_error_pa", "enthalpy_error_j_kg")}
            for observed, a, b in zip(
                out["observations"][1:], refs[0]["observations"], refs[1]["observations"]
            ):
                for criterion, field in [
                    ("pressure_error_pa", "pressure_pa"),
                    ("enthalpy_error_j_kg", "enthalpy_j_kg"),
                ]:
                    error = max(abs(observed["state"][field] - a[field]), abs(a[field] - b[field]))
                    measured[criterion] = max(measured[criterion], error)
                    if error > CRITERIA[criterion]:
                        raise ValueError(f"{name}/{cap} reference mismatch: {criterion}={error}")
                ledger = observed["ledgers"]
                for criterion, field in [
                    ("energy_balance_j_kg", "energy_residual_j_kg"),
                    ("momentum_balance_pa", "momentum_residual_pa"),
                    (
                        "corrected_entropy_balance_j_kg_k",
                        "corrected_numerical_entropy_residual_j_kg_k",
                    ),
                ]:
                    if abs(ledger[field]) > CRITERIA[criterion]:
                        raise ValueError(f"{name}/{cap}: {field}={ledger[field]}")
                if observed["state"]["density_kg_m3"] <= 0 or ledger["drag_entropy_j_kg_k"] < 0:
                    raise ValueError("Positive density/drag production")
            events = out["final_checkpoint"]["payload"]["phase_events"]
            if any(
                abs(e["enthalpy_residual_j_kg"]) > CRITERIA["phase_root_residual_j_kg"]
                for e in events
            ):
                raise ValueError("Phase root residual")
            outputs.append(
                dict(
                    max_step_m=cap,
                    errors=measured,
                    checkpoint_sha256=out["checkpoints"],
                    phase_events=events,
                    final=out["observations"][-1],
                )
            )
        records.append(
            dict(
                name=name,
                reference_evaluations={r["method"]: r["evaluations"] for r in refs},
                trajectories=outputs,
            )
        )
    spatial = refinement()
    if source != signature():
        raise ValueError("Qualification source changed")
    return dict(
        schema_version=1,
        source=source,
        model="steady-open-hem-ivp-v1",
        criteria=CRITERIA,
        case_count=len(records),
        trajectories=len(records) * len(CAPS),
        reference_ivps=2 * len(records),
        cases=records,
        spatial=spatial,
        validated=False,
        plant_referenced=False,
        entropy_scope="Raw entropy residual and explicit IF97 mixture Gibbs-fit defect reported separately; only corrected numerical residual is an integration check",
    )


def audit(path, kind):
    saved = read(path)
    keys(saved, ("payload", "sha256"))
    if saved["sha256"] != digest(saved["payload"]):
        raise ValueError("Evidence digest mismatch")
    expected = qualify() if kind == "qualification" else benchmark.evaluate()
    if digest(expected) != saved["sha256"]:
        raise ValueError("Evidence source/recomputation mismatch")
    return expected
