"""Independent steady branches, no-warmup handoff, rollback and replay."""

from scripts.buoyant_graph_references import IndependentBuoyant, loop

from .steady import audit_certificate, identity, initialize


def qualify():
    cases = []
    for name, request, guess in [
        ("heated", loop(), 0.13),
        ("mirror", loop(mirror=True), -0.13),
        ("eight_nodes", loop(nodes=8), 0.13),
        ("branch", loop(branch=True), 0.13),
    ]:
        certificate = initialize(request, flow_guess_kg_s=guess)
        reference = IndependentBuoyant(request).stationary(guess)
        initial = certificate["payload"]["state"]
        errors = {}
        for field, tolerance in [
            ("flows_kg_s", 1e-7),
            ("temperatures_k", 1e-6),
            ("pressures_pa", 1e-4),
        ]:
            error = max(abs(initial[field][k] - reference[field][k]) for k in initial[field])
            if error > tolerance:
                raise ValueError(f"{name}: independent steady {field} mismatch")
            errors[field] = error
        drift, restarts = [], []
        for cap in [0.5, 0.25]:
            runtime = audit_certificate(certificate)
            # Only integration step changes, not states or physical coefficients.
            runtime = type(runtime)(certificate["payload"]["initialized_request"], max_step_s=cap)
            runtime.advance(20.0)
            saved = runtime.checkpoint()
            final = runtime.advance(20.0)["state"]
            restored = type(runtime)(certificate["payload"]["initialized_request"], max_step_s=cap)
            restored.restore(saved)
            resumed = restored.advance(20.0)["state"]
            for field, tolerance in [
                ("flows_kg_s", 1e-7),
                ("temperatures_k", 1e-6),
                ("pressures_pa", 1e-4),
            ]:
                d = max(abs(final[field][k] - initial[field][k]) for k in initial[field])
                if d > tolerance:
                    raise ValueError("Steady handoff developed hidden transient")
            if resumed != final:
                raise ValueError("Steady handoff restart mismatch")
            drift.append(dict(max_step_s=cap, final_time_s=40.0, state=final))
            restarts.append(True)
        runtime = audit_certificate(certificate)
        saved = runtime.checkpoint()
        try:
            runtime.advance(1001.0)
        except ValueError:
            if runtime.checkpoint() != saved:
                raise ValueError("Rejected interval mutated certified runtime")
        else:
            raise ValueError("Budget overflow accepted")
        cases.append(
            dict(
                case=name,
                certificate=certificate,
                independent_stationary_errors=errors,
                hold_and_restart=drift,
                restart_equal=restarts,
                rollback_equal=True,
            )
        )
    # A closed unheated loop has an undetermined mean temperature.
    try:
        initialize(loop(heated=False), flow_guess_kg_s=0.0, temperature_guess_k=300.0)
    except ValueError as exc:
        rank_refusal = str(exc)
    else:
        raise ValueError("Underdetermined thermal equilibrium accepted")
    return dict(
        schema_version=1,
        source=identity(),
        cases=cases,
        missing_thermal_constraint_refusal=rank_refusal,
        scope="Bounded closed Boussinesq graph, static coefficients and explicit gauge; no variable-density water or full-plant acceptance",
        validated=False,
        plant_referenced=False,
    )
