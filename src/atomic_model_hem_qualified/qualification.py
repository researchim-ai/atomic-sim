"""Reproducible bounded numerical qualification, never physical validation."""

from copy import deepcopy
from math import pi, sin

import numpy as np

from atomic_model_hem_qualified import eos
from atomic_model_hem_qualified.cases import baseline, programs
from atomic_model_hem_qualified.model import QualifiedHEMNetwork, signature
from atomic_model_hem_qualified.reference import solve

# Frozen admission criteria; absolute physical units unless explicitly relative.
CRITERIA = dict(
    pressure_error_pa=0.1,
    temperature_error_k=1e-5,
    velocity_error_m_s=1e-6,
    temporal_refinement_ratio=0.8,
    mass_balance_kg=1e-9,
    energy_balance_j=1e-5,
    numerical_entropy_lower_j_k=-2e-5,
    spatial_refinement_ratio=0.9,
)


def execute(request, calls):
    m = QualifiedHEMNetwork(request)
    observations = [m.snapshot()]
    replay = QualifiedHEMNetwork(request)
    replay.restore(m.checkpoint())
    if replay.checkpoint() != m.checkpoint():
        raise ValueError("Nonexact initial replay")
    checkpoints = []
    for call in calls:
        command = call["command"]
        if command is None:
            observations.append(m.advance(call["duration_s"]))
        else:
            observations.append(
                m.step(
                    command["controls"],
                    call["duration_s"],
                    expected_revision=command["expected_revision"],
                    controller_state=command["controller_state"],
                )
            )
        cp = m.checkpoint()
        replay = QualifiedHEMNetwork(request)
        replay.restore(cp)
        if replay.checkpoint() != cp:
            raise ValueError("Nonexact replay")
        checkpoints.append(cp["sha256"])
    return observations, checkpoints


def errors(a, b):
    return {
        key: max(
            abs(
                (x[field] if field in x else x["momentum_kg_m_s"] / x["mass_kg"])
                - (y[field] if field in y else y["momentum_kg_m_s"] / y["mass_kg"])
            )
            for x, y in zip(a["cells"], b["cells"])
        )
        for key, field in [
            ("pressure_error_pa", "pressure_pa"),
            ("temperature_error_k", "temperature_k"),
            ("velocity_error_m_s", "velocity_m_s"),
        ]
    }


def spatial(two_phase=False):
    """Same physical smooth graph, conservative restriction to a 4-cell mesh.

    Uses 64 cells per edge as a numerical comparison, NOT an exact PDE oracle.
    """
    profiles = {}
    stocks = {}
    for n in (4, 8, 16, 32, 64):
        spec = baseline(cells=n)
        spec["friction"] = {}
        spec["model"]["max_step_s"] = 1e-6
        spec["model"]["walls"][0]["conductance_w_k"] = 0.0
        for node in spec["model"]["nodes"]:
            node["initial"] = dict(
                kind="pt", pressure_pa=3e6, temperature_k=350.0, velocity_m_s=0.0
            )
        for edge in spec["model"]["edges"]:
            edge["initial"] = [
                dict(
                    kind="pt",
                    pressure_pa=float(3e6 + 1000 * sin(pi * (j + 0.5) / n) * np.sinc(0.5 / n)),
                    temperature_k=350.0,
                    velocity_m_s=0.0,
                )
                for j in range(n)
            ]
        if two_phase:
            for node in spec["model"]["nodes"]:
                node["initial"] = dict(kind="px", pressure_pa=1e6, quality=0.5, velocity_m_s=0.0)
            for edge in spec["model"]["edges"]:
                edge["initial"] = [
                    dict(
                        kind="px",
                        pressure_pa=float(1e6 + 1000 * sin(pi * (j + 0.5) / n) * np.sinc(0.5 / n)),
                        quality=0.5,
                        velocity_m_s=0.0,
                    )
                    for j in range(n)
                ]
        m = QualifiedHEMNetwork(spec)
        m.advance(2e-5)
        mass, P, C, _, _, _, _ = m.primitive(m._y)
        energy = C + mass * m.eref
        blocks = []
        for ix in m.edge_indices:
            for group in np.array_split(ix, 4):
                mm = sum(mass[group])
                pp = sum(P[group])
                ee = sum(energy[group])
                state = eos.rho_u(
                    float(mm / sum(m.volumes[group])),
                    float(ee / mm - 0.5 * (pp / mm) ** 2),
                    1e6 if two_phase else 3e6,
                    eos.px(1e6, 0.5).temperature_k if two_phase else 350.0,
                )
                blocks.append(state.pressure_pa)
        profiles[n] = np.array(blocks)
        stocks[str(n)] = dict(minimum_mass_kg=float(min(mass)), fluid_cells=m.n)
    reference = profiles[64]
    errs = {str(n): float(np.mean(abs(profiles[n] - reference))) for n in (4, 8, 16, 32)}
    ratios = [errs[str(b)] / errs[str(a)] for a, b in [(4, 8), (8, 16), (16, 32)]]
    if not all(r < CRITERIA["spatial_refinement_ratio"] for r in ratios):
        raise ValueError(f"Graph spatial convergence failed: {errs}")
    return dict(
        kind="graph self-convergence; finite 64-cell comparison, not exact continuum truth",
        duration_s=2e-5,
        common_restriction_cells_per_edge=4,
        mean_pressure_error_pa=errs,
        refinement_ratios=ratios,
        stocks=stocks,
    )


def thermodynamic_checks():
    rng = np.random.default_rng(7406)
    maxima = dict(density_relative=0.0, internal_energy_j_kg=0.0, pressure_relative=0.0)
    for i in range(300):
        p = float(np.exp(rng.uniform(np.log(1.01e5), np.log(15.9e6))))
        state = eos.px(p, float(rng.random())) if i % 2 else eos.pt(p, float(rng.uniform(310, 890)))
        inverse = eos.rho_u(state.density_kg_m3, state.internal_energy_j_kg)
        residual = dict(
            density_relative=abs(inverse.density_kg_m3 / state.density_kg_m3 - 1),
            internal_energy_j_kg=abs(inverse.internal_energy_j_kg - state.internal_energy_j_kg),
            pressure_relative=abs(inverse.pressure_pa / state.pressure_pa - 1),
        )
        for key, value in residual.items():
            maxima[key] = max(maxima[key], value)
    if (
        maxima["density_relative"] > eos.RHO_RTOL
        or maxima["internal_energy_j_kg"] > eos.U_ATOL
        or maxima["pressure_relative"] > 2e-8
    ):
        raise ValueError("Far-seed thermodynamic roundtrip failed")
    return dict(
        count=300,
        seed=7406,
        maxima=maxima,
        initial_seed=dict(pressure_pa=1e6, temperature_k=450.0),
        projection=False,
    )


def qualify():
    source = signature()
    records = []
    for name, request, calls in programs():
        base, checkpoints = execute(request, calls)
        fine = deepcopy(request)
        fine["model"]["max_step_s"] /= 2
        refined, _ = execute(fine, calls)
        refs = [solve(request, calls, method) for method in ("DOP853", "Radau")]
        measured = errors(base[-1], refs[0]["observations"][-1])
        independent = errors(refs[0]["observations"][-1], refs[1]["observations"][-1])
        refined_error = errors(refined[-1], refs[0]["observations"][-1])
        if any(measured[k] > CRITERIA[k] or independent[k] > CRITERIA[k] for k in measured):
            raise ValueError(f"{name}: independent trajectory mismatch {measured} {independent}")
        # Only demand an observed rate when well above floating-point/IVP noise.
        if measured["pressure_error_pa"] > 1e-4 and refined_error["pressure_error_pa"] > max(
            1e-4, CRITERIA["temporal_refinement_ratio"] * measured["pressure_error_pa"]
        ):
            raise ValueError(f"{name}: temporal refinement failed")
        expected_phase = {
            "vapor_phase_exit": "vapor",
            "liquid_phase_exit": "liquid",
            "condensation_entry": "two_phase",
        }.get(name)
        if expected_phase and not any(s["phase"] == expected_phase for s in base[-1]["cells"]):
            raise ValueError(f"{name}: expected phase transition absent")
        ledger = base[-1]["global_ledger"]
        for observed in base:
            balance = observed["global_ledger"]
            if (
                abs(balance["mass_residual_kg"]) > CRITERIA["mass_balance_kg"]
                or abs(balance["energy_residual_j"]) > CRITERIA["energy_balance_j"]
                or balance["numerical_entropy_residual_j_k"]
                < CRITERIA["numerical_entropy_lower_j_k"]
                or min(s["mass_kg"] for s in observed["cells"]) <= 0
            ):
                raise ValueError(f"{name}: conservation/positivity/entropy qualification failed")
        records.append(
            dict(
                name=name,
                errors=measured,
                refined_errors=refined_error,
                independent_method_errors=independent,
                reference_evaluations={r["method"]: r["evaluations"] for r in refs},
                checkpoint_sha256=checkpoints,
                phase_history=[[s["phase"] for s in o["cells"]] for o in base],
                minimum_mass_kg=min(s["mass_kg"] for o in base for s in o["cells"]),
                global_ledger=ledger,
            )
        )
    sensitivity = []
    from atomic_model_hem_qualified import closures

    for p in (1e6, 8e6, 15.5e6):
        for x in (0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 1.0):
            state = eos.px(p, x)
            choices = {
                name: closures.friction(
                    state,
                    0.01,
                    0.008,
                    0.0,
                    viscosity_model=name,
                    scope={flag: True for flag in closures.FLAGS},
                )["force_density_n_m3"]
                for name in closures.MODELS
            }
            sensitivity.append(
                dict(
                    pressure_pa=p,
                    quality=x,
                    force_density_n_m3=choices,
                    relative_spread=abs(choices["mcadams"] - choices["cicchitti"])
                    / max(abs(v) for v in choices.values()),
                )
            )
    mesh = dict(liquid=spatial(), two_phase=spatial(True))
    if signature() != source:
        raise ValueError("Qualification source changed")
    return dict(
        schema_version=1,
        model="active-chart-pre-chf-graph-v1",
        source=source,
        criteria=CRITERIA,
        cases=records,
        spatial=mesh,
        thermodynamic_roundtrips=thermodynamic_checks(),
        closure_sensitivity=sensitivity,
        sensitivity_scope="Model-choice spread only, not calibrated physical uncertainty",
        scope="Numerical verification; shared IF97 EOS; no experimental/plant validation; no post-CHF",
    )
