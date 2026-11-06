"""Exact temperature errors, sampled physical constraints and complete-pair selection."""

from atomic_model_reference.metrics import compare_curve
from atomic_model_workspace.common import canonical, digest


def evaluate(job, pack):
    rows = [r["model"] for r in job["rows"]]
    objective = job["scenario"]["objective"]
    times = [r["time_s"] for r in rows]
    values = [
        next(c["value"] for c in r["observation"]["channels"] if c["name"] == objective["sensor"])
        for r in rows
    ]
    duration = times[-1] - times[0]
    quality = (
        compare_curve(
            times, values, [times[0], times[-1]], [objective["target_k"]] * 2, [times[0], times[-1]]
        )
        if duration
        else None
    )
    peaks = dict(
        energy_residual_j=max(abs(r["ledger"]["energy_residual_j"]) for r in rows),
        entropy_residual_j_k=max(abs(r["ledger"]["entropy_residual_j_k"]) for r in rows),
        mass_residual_kg=max(
            abs(c["mass_residual_kg"]) for r in rows for c in r["circuits"].values()
        ),
        momentum_residual_kg_m_s=max(
            abs(c["momentum_residual_kg_m_s"]) for r in rows for c in r["circuits"].values()
        ),
        circuit_energy_residual_j=max(
            abs(c["energy_residual_j"]) for r in rows for c in r["circuits"].values()
        ),
        absolute_power_w=max(
            (abs(v) for r in rows for v in (r["heat_power_w"] or {}).values()), default=0.0
        ),
    )
    checks = {k: value <= pack["constraints"][k] for k, value in peaks.items()}
    physical = [
        {
            k: r[k]
            for k in ("decision", "time_s", "observation", "heat_power_w", "ledger", "circuits")
        }
        for r in rows
    ]
    completed = job["status"] == "completed"
    return dict(
        status=job["status"],
        reason=job["reason"],
        completed=completed,
        eligible=completed and all(checks.values()),
        accepted_decisions=len(rows) - 1,
        completed_time_s=times[-1],
        quality=quality,
        quality_metric_units={
            key: (
                "1"
                if key in ("comparison_knots", "fraction_time_within_declared_uncertainty")
                else "s"
                if key.endswith("_s")
                else "K s"
                if "integral" in key
                else "K"
            )
            for key in quality
        }
        if quality is not None
        else None,
        quality_base_unit="K",
        quality_integral_unit="K s",
        sampled_constraint_peaks=peaks,
        constraint_checks=checks,
        target_met=completed
        and abs(values[-1] - objective["target_k"]) <= objective["tolerance_k"],
        trajectory_sha256=digest(physical),
        episode_sha256=digest(job["episode"]),
        agent_decision_wall_s=sum(a["telemetry"].get("wall_s", 0) for a in job["attempts"]),
        agent_peak_rss_bytes=max(
            (a["telemetry"].get("peak_rss_bytes", 0) for a in job["attempts"]), default=0
        ),
        captured_stdout_bytes=sum(a["telemetry"].get("stdout_bytes", 0) for a in job["attempts"]),
        token_usage=None,
        token_cost=None,
    )


def select(pack, records):
    train = [r for r in records if r["phase"] == "train"]
    expected = sum(c["split"] == "train" for c in pack["cases"].values()) * pack["repeats"]
    scores = {}
    for id in sorted(pack["agents"]):
        rows = [r for r in train if r["agent_id"] == id]
        eligible = len(rows) == expected and all(r["metrics"]["eligible"] for r in rows)
        scores[id] = dict(
            eligible=eligible,
            episodes=len(rows),
            mean_mae_k=sum(r["metrics"]["quality"]["mae"] for r in rows) / expected
            if eligible
            else None,
        )
    candidates = [
        (s["mean_mae_k"], id)
        for id, s in scores.items()
        if id != pack["baseline_id"] and s["eligible"]
    ]
    selected = min(candidates)[1] if candidates else None
    return dict(
        schema_version=1,
        protocol_sha256=digest(pack),
        selected_agent=selected,
        selected_descriptor_sha256=digest(pack["agents"][selected]) if selected else None,
        training_task_ids=[r["task_id"] for r in train],
        scores=scores,
        selection_rule="minimum equal-case/repeat mean train MAE among complete constraint-admitted candidates; lexical ID breaks exact ties; no dev/test input",
    )


def report(pack, records, selection, hardware):
    phases = {}
    for phase in ("train", "dev", "test"):
        phase_rows = [r for r in records if r["phase"] == phase]
        ids = sorted({r["agent_id"] for r in phase_rows})
        expected = sum(c["split"] == phase for c in pack["cases"].values()) * pack["repeats"]
        agents = {}
        for id in ids:
            rows = [r for r in phase_rows if r["agent_id"] == id]
            eligible = len(rows) == expected and all(r["metrics"]["eligible"] for r in rows)
            repeated = {
                case: [r for r in rows if r["case_id"] == case]
                for case, c in pack["cases"].items()
                if c["split"] == phase
            }
            exact = all(
                len(group) == pack["repeats"]
                and len({r["metrics"]["episode_sha256"] for r in group}) == 1
                and len({r["metrics"]["trajectory_sha256"] for r in group}) == 1
                for group in repeated.values()
            )
            agents[id] = dict(
                eligible=eligible,
                episodes=len(rows),
                completed=sum(r["metrics"]["completed"] for r in rows),
                mean_mae_k=sum(r["metrics"]["quality"]["mae"] for r in rows) / expected
                if eligible
                else None,
                native_repeat_exact=exact,
                mean_observed_wall_s=sum(r["observed_wall_s"] for r in rows) / len(rows),
                mean_agent_wall_s=sum(r["metrics"]["agent_decision_wall_s"] for r in rows)
                / len(rows),
                violations=sum(not all(r["metrics"]["constraint_checks"].values()) for r in rows),
            )
        ranking = []
        previous = None
        rank = 0
        for index, (score, id) in enumerate(
            sorted((r["mean_mae_k"], id) for id, r in agents.items() if r["eligible"]), 1
        ):
            if previous is None or score != previous:
                rank = index
            ranking.append(dict(agent_id=id, rank=rank, mean_mae_k=score))
            previous = score
        phases[phase] = dict(agents=agents, ranking=ranking, expected_episodes_per_agent=expected)
    return dict(
        schema_version=1,
        benchmark_id=pack["benchmark_id"],
        protocol_sha256=digest(pack),
        hardware_sha256=digest(hardware),
        selection_sha256=digest(selection),
        status="completed" if selection["selected_agent"] else "no_valid_train_candidate",
        phases=phases,
        episodes=records,
        semantics=dict(
            quality="Exact piecewise-linear time-weighted temperature error; ranking within phase only, equal case/repeat weight",
            incomplete="Entire agent/phase excluded if any required episode incomplete or violates sampled constraint; no survivorship ranking",
            physical_constraints="Peaks on accepted decision samples; internal native admission remains active; no continuous-time safety certificate",
            seeds="EOS and demo policies deterministic; recorded seed does not create new physics or independent stochastic evidence",
            cost="Observed create/run/publish wall; agent startup included, later audit excluded. Shared CPU affinity/threads, no host-load/frequency isolation or speed winner",
            labels="Synthetic local policy responses, not expert/causal annotations; reward undefined",
            holdout="Selection uses train only and freezes before dev/test. Public cases and exported test transitions are not secret; do not retune on this test set",
        ),
        token_usage=None,
        token_cost=None,
        validated=False,
        plant_referenced=False,
    )


def transitions(pack, records, jobs):
    lines = []
    for entry in records:
        job = jobs[entry["task_id"]]
        rows = job["rows"]
        for index, row in enumerate(rows[1:], 1):
            req = row["request"]
            desc = job["agent"]
            after = dict(
                time_s=row["model"]["time_s"],
                channels=[
                    c
                    for c in row["model"]["observation"]["channels"]
                    if c["name"] in desc["observations"]
                ],
            )
            item = dict(
                schema_version=1,
                case_id=entry["case_id"],
                agent_id=entry["agent_id"],
                split=entry["phase"],
                repeat=entry["repeat"],
                decision=req["decision"],
                observation=req["observation"],
                next_observation=after,
                action_w=row["response"]["heat_power_w"],
                state=req["state"],
                next_state=row["response"]["state"],
                terminated=False,
                truncated=index == len(rows) - 1,
                episode_completed=job["status"] == "completed" and index == len(rows) - 1,
                completion_reason=job["reason"] if index == len(rows) - 1 else None,
                reward=None,
                label_origin="synthetic local policy response; not expert or causal annotation",
                protocol_sha256=digest(pack),
            )
            lines.append(canonical(item) + "\n")
    return "".join(lines).encode()
