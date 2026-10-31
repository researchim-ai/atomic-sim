"""Independent endpoint-quadrature metrics and paired stratified bootstrap."""

import math

import numpy as np


def metrics(raw):
    cfg = raw["case"]["config"]
    target = cfg["target_power_fraction"]
    elapsed = 0.0
    iae = squared = effort = violations = 0.0
    points = [raw["initial_truth"]]
    trips = 0
    latched = bool(points[0]["scram_latched"])
    last_diagnosis = "healthy"
    for row in raw["steps"]:
        dt = row["time_s"] - elapsed
        if dt < 0 or not math.isfinite(dt):
            raise ValueError("Nonmonotonic raw episode")
        if row["time_s"] > cfg["episode_horizon_s"]:
            raise ValueError("Episode exceeds requested horizon")
        truth = row["truth"]
        error = truth["power_fraction"] - target
        iae += abs(error) * dt
        squared += error**2 * dt
        violations += dt * int(abs(error) > cfg["power_error_limit"])
        effort += (row["action"][0] / 2) ** 2 * dt
        trips += int(bool(truth["scram_latched"]) and not latched)
        latched = bool(truth["scram_latched"])
        elapsed = row["time_s"]
        points.append(truth)
        last_diagnosis = row["diagnosis"]
    complete = raw["reason"] == "time_limit" and elapsed == cfg["episode_horizon_s"]
    return dict(
        completed=complete,
        reason=raw["reason"],
        completed_time_s=elapsed,
        requested_time_s=cfg["episode_horizon_s"],
        iae_s=iae,
        rmse_fraction=math.sqrt(squared / elapsed) if elapsed else None,
        violation_time_s=violations,
        action_cost=effort,
        max_power_fraction=max(p["power_fraction"] for p in points),
        max_fuel_temperature_k=max(p["fuel_temperature_k"] for p in points),
        max_primary_pressure_pa=max(p["primary_pressure_pa"] for p in points),
        protection_count=trips,
        diagnosis_correct=last_diagnosis == raw["case"]["diagnosis_label"],
        cumulative_reward=sum(r["reward"] for r in raw["steps"]),
        wall_time_s=raw["wall_time_s"],
    )


def summarize(records, seed=2026, resamples=2000):
    if not records:
        raise ValueError("Empty campaign")
    records = sorted(records, key=lambda row: (row["policy"], row["case_id"]))
    policies = sorted({r["policy"] for r in records})
    seen = set()
    for r in records:
        key = (r["policy"], r["case_id"])
        if key in seen:
            raise ValueError("Duplicate campaign case")
        seen.add(key)
    aggregates = {}
    for policy in policies:
        subset = [r for r in records if r["policy"] == policy]
        complete = [r for r in subset if r["metrics"]["completed"]]
        aggregates[policy] = dict(
            episodes=len(subset),
            completed=len(complete),
            completion_rate=len(complete) / len(subset),
            mean_iae_completed_s=float(np.mean([r["metrics"]["iae_s"] for r in complete]))
            if complete
            else None,
            diagnosis_accuracy=float(np.mean([r["metrics"]["diagnosis_correct"] for r in subset])),
            total_wall_time_s=sum(r["metrics"]["wall_time_s"] for r in subset),
            failure_cases=[r["case_id"] for r in subset if not r["metrics"]["completed"]],
        )
        family_results = {}
        for family in sorted({r["family"] for r in subset}):
            family_rows = [r for r in subset if r["family"] == family]
            accepted = [r for r in family_rows if r["metrics"]["completed"]]
            family_results[family] = dict(
                episodes=len(family_rows),
                completed=len(accepted),
                mean_iae_completed_s=float(np.mean([r["metrics"]["iae_s"] for r in accepted]))
                if accepted
                else None,
                diagnosis_accuracy=float(
                    np.mean([r["metrics"]["diagnosis_correct"] for r in family_rows])
                )
                if family == "diagnosis"
                else None,
            )
        aggregates[policy]["by_family"] = family_results
    pairs = []
    rng = np.random.default_rng(seed)
    for i, a in enumerate(policies):
        for b in policies[i + 1 :]:
            left = {r["case_id"]: r for r in records if r["policy"] == a}
            right = {r["case_id"]: r for r in records if r["policy"] == b}
            if set(left) != set(right):
                raise ValueError("Unpaired campaign coverage")
            groups = {}
            for key, row in left.items():
                other = right[key]
                if row["family"] != other["family"]:
                    raise ValueError("Family mismatch")
                if row["metrics"]["completed"] and other["metrics"]["completed"]:
                    groups.setdefault(row["family"], []).append(
                        row["metrics"]["iae_s"] - other["metrics"]["iae_s"]
                    )
            count = sum(len(v) for v in groups.values())
            if not count:
                pairs.append(dict(a=a, b=b, common_completed=0, mean_delta_iae_s=None, ci95=None))
                continue
            samples = np.zeros(resamples)
            for values in groups.values():
                values = np.array(values)
                samples += (
                    rng.choice(values, size=(resamples, len(values)), replace=True).sum(axis=1)
                    / count
                )
            mean = sum(sum(v) for v in groups.values()) / count
            pairs.append(
                dict(
                    a=a,
                    b=b,
                    common_completed=count,
                    mean_delta_iae_s=mean,
                    ci95=np.quantile(samples, [0.025, 0.975]).tolist(),
                )
            )
    return dict(
        policies=aggregates,
        paired=pairs,
        bootstrap=dict(
            seed=seed,
            resamples=resamples,
            method="paired case resampling stratified by family, percentile 95%; public corpus only",
        ),
        ranking_rule="completion rate first; IAE only on common completed cases; reward is diagnostic",
        validated=False,
        plant_referenced=False,
    )
