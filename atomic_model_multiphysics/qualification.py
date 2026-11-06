"""Independent numerical qualification and explicit separation of external gates."""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from atomic_model_phase_assessment.dataset import load as pool_dataset
from atomic_model_spatial import eigenvalue
from atomic_model_spatial.cases import slab
from atomic_model_validation.pipe import dataset as pipe_dataset
from atomic_model_validation.pipe import native_factor

from .absorber import study
from .agent import SpatialAgent
from .coupled import solve
from .depletion import advance, cecm, operator
from .feedback import numerical_table
from .nuclear import model_from_mgxs
from .provenance import identity, sha, unseal


def qualify_depletion():
    rows = []
    for decay, capture, n, dt in [
        ([0.1, 0.01, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 10.0),
        ([1e-8, 1e3, 0.0], [1e-28, 2e-28, 0.0], [1e12, 1e8, 1.0], 1e4),
        ([0.03, 0.02, 0.0], [0.0, 0.0, 0.0], [2.0, 3.0, 4.0], 30.0),
    ]:
        a = operator(decay, capture, 1e14)
        result = advance(a, n, dt, [2e-13, 4e-13, 0.0])
        exact = expm(a * dt) @ n
        error = float(
            np.max(abs(np.array(result["composition"]) - exact) / np.maximum(1.0, abs(exact)))
        )
        rows.append(
            dict(
                input=dict(decay_per_s=decay, capture_m2=capture, composition=n, dt_s=dt),
                result=result,
                composition_relative_error=error,
            )
        )
    # A state-dependent destruction rate has an independent rational solution.
    errors = []
    for steps in (10, 20, 40):
        n = np.array([1.0, 0.0])
        dt = 1 / steps

        def rates(state):
            r = 0.2 * state[0]
            return np.array([[-r, 0.0], [r, 0.0]])

        for _ in range(steps):
            n = cecm(n, dt, rates)
        errors.append(abs(float(n[0]) - 1 / 1.2))
    orders = [float(np.log2(a / b)) for a, b in zip(errors, errors[1:])]
    return dict(
        cases=rows,
        cecm_errors=errors,
        cecm_orders=orders,
        accepted=max(r["composition_relative_error"] for r in rows) < 1e-8 and min(orders) > 1.8,
        data_scope="Explicit capture/decay chains; evaluated fuel inventories and ANS decay-heat certification are excluded",
    )


def uncertainty_report():
    protocol, rows = pipe_dataset()
    calibration = [r for r in rows if r["series"] == "test_1"]
    heldout = [r for r in rows if r["series"] != "test_1"]
    results = []
    for row in heldout:
        center = native_factor(row["reynolds"], protocol)
        data_bound = row["darcy_f_reading_bound"]
        results.append(
            dict(
                series=row["series"],
                reynolds=row["reynolds"],
                observed=row["darcy_f"],
                prediction=center,
                digitization_bound=data_bound,
                relative_error=abs(center - row["darcy_f"]) / row["darcy_f"],
            )
        )
    pool, clocks = pool_dataset()
    # Compute numerical ODE uncertainty separately from measured-data reading bounds.
    temporal = []
    for rtol in (1e-5, 1e-7, 1e-9):
        solution = solve_ivp(
            lambda t, y: -0.3 * y, (0, 10), [1.0], method="Radau", rtol=rtol, atol=rtol * 0.01
        )
        temporal.append(dict(rtol=rtol, absolute_error=abs(float(solution.y[0, -1]) - np.exp(-3))))
    return dict(
        calibration_cases=["test_1"],
        calibration_rows=len(calibration),
        validation_cases=["test_2", "test_3", "test_4"],
        validation_rows=len(heldout),
        parameter_fitting_performed=False,
        heldout_comparisons=results,
        components=dict(
            data="Digitization bounds only; total experimental covariance absent",
            parameters="No distribution supplied; not inferred from residuals",
            closures="Model discrepancy retained as per-case residuals",
            mesh="Independent spatial qualification; not probabilistic uncertainty",
            integrator=temporal,
        ),
        transient_candidate=dict(
            case=pool["case"],
            license=pool["data_license"],
            clocks={k: len(v) for k, v in clocks.items()},
            clocks_joined=False,
        ),
        experimental_coverage_accepted=False,
        blocking_evidence=[
            "Per-case measurement uncertainty/covariance",
            "Applicable coupled-system measured series",
        ],
        validated=False,
        plant_referenced=False,
    )


def qualify(openmc_root=None):
    table = numerical_table()
    coupled = []
    for cells, channels in ((6, 5), (12, 9), (24, 17)):
        config = slab(cells)
        result = solve(config, table, channel_faces=np.linspace(0, 1, channels + 1))
        coupled.append(dict(cells=cells, channels=channels, result=result))
    agent = SpatialAgent(slab(6), table)
    for power in (2500.0, 5000.0, 7500.0):
        agent.step(dict(thermal_power_w=power), expected_revision=agent.session.revision)
    saved = json.loads(json.dumps(agent.checkpoint()))
    restart = SpatialAgent.restore(saved).checkpoint() == saved
    transport = []
    if openmc_root:
        for p in sorted(Path(openmc_root).glob("replica-*/mgxs.json")):
            envelope = json.loads(p.read_text())
            source = unseal(envelope)
            execution = unseal(json.loads(p.with_name("execution.json").read_text()))
            if (
                execution["status"] != "completed"
                or execution["result_sha256"] != sha(p)
                or source["producer_worker_sha256"]
                != sha(Path(__file__).with_name("openmc_worker.py"))
            ):
                raise ValueError("Transport execution/producer identity differs")
            for name, expected in source["retained"].items():
                if Path(name).name != name or sha(p.parent / name) != expected:
                    raise ValueError("Retained transport XML/statepoint differs")
            result = eigenvalue(model_from_mgxs(envelope))
            transport.append(
                dict(
                    input=str(p),
                    input_sha256=envelope["sha256"],
                    seed=source["specification"]["seed"],
                    k_mc=source["k_mean"],
                    k_std=source["k_std"],
                    k_diffusion=result["k"],
                    difference_standard_errors=(result["k"] - source["k_mean"]) / source["k_std"],
                    evidence_kind="intercode",
                    validated=False,
                )
            )
    depletion = qualify_depletion()
    absorber = study()
    return dict(
        source=identity(),
        coupled=coupled,
        agent_checkpoint=saved,
        agent_restart_equal=restart,
        depletion=depletion,
        absorber=absorber,
        transport=transport,
        uncertainty=uncertainty_report(),
        numerical_accepted=restart
        and depletion["accepted"]
        and absorber["accepted"]
        and all(c["result"]["thermal"]["energy_relative"] < 1e-9 for c in coupled),
        openmc_pipeline_executed=len(transport) == 3,
        external_gates=dict(
            independent_review=False,
            another_machine=False,
            applicable_system_experiment=False,
            experimental_coverage=False,
        ),
        validated=False,
        plant_referenced=False,
    )
