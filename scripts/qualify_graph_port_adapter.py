"""Source-bound qualification of typed nonlinear residuals and state lifting.

Independent mixed hydraulic roots and instantaneous enthalpy RHS comparisons;
these are not experimental observations or new independent transient IVPs.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from scripts.graph_port_adapter import BUDGET, GraphPortAdapter
from scripts.graph_port_adapter import signature as model_signature
from scripts.nonlinear_graph_references import IndependentSystem, cases
from scripts.nonlinear_thermal_graph import NonlinearThermalGraph
from scripts.solve_hydraulic_graph import sha

EXTRA = (
    "scripts/qualify_graph_port_adapter.py",
    "scripts/nonlinear_graph_references.py",
    "scripts/hydraulic_graph_references.py",
    "scripts/resistance_feedback_references.py",
    "scripts/thermal_graph_references.py",
)
TIMES = (0.0, 37.5, 120.0)


def signature():
    value = model_signature()
    value["qualification_drivers"] = {p: sha(p) for p in EXTRA}
    return value


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def jacobian_error(adapter, state, rate):
    n = adapter._n
    y = np.array(state)
    analytic = adapter.algebraic_jacobian(y)
    numerical = np.zeros_like(analytic)
    for j in range(n, len(y) - 1):
        h = 0.01 if j < 2 * n else 1e-6
        plus = y.copy()
        minus = y.copy()
        plus[j] += h
        minus[j] -= h
        numerical[:, j - n] = (
            adapter.residual(plus, rate)[n:-1] - adapter.residual(minus, rate)[n:-1]
        ) / (2 * h)
    return float(np.max(abs(numerical - analytic) / (1.0 + abs(analytic))))


def evaluate(spec):
    adapter = GraphPortAdapter(spec)
    kernel = NonlinearThermalGraph(spec)
    independent = IndependentSystem(spec)
    budget = json.loads(BUDGET.read_text())
    observations = []
    for time in TIMES:
        if time > 0:
            kernel.advance(time - kernel.snapshot()["time_s"])
        result = adapter.initialize() if time == 0 else adapter.lift_checkpoint(kernel.checkpoint())
        snapshot = kernel.snapshot()
        state = np.array(result["state"])
        n = adapter._n
        if (
            result["time_s"] != time
            or result["state"][:n] != list(snapshot["temperatures_k"].values())
            or state[-1] != snapshot["integrated_external_energy_j"]
        ):
            raise ValueError("Adapter state transfer changed clock, stocks or energy")
        _, hyd = independent.hydraulic(state[:n])
        rhs = independent.rhs(
            time, np.r_[state[:n] - independent.initial, state[-1] / independent.scale]
        )
        errors = dict(
            pressure_pa=max(
                abs(state[n + i] - hyd["pressures_pa"][name])
                for i, name in enumerate(adapter._names)
            ),
            flow_kg_s=max(
                abs(state[2 * n + i] - hyd["flows_kg_s"][e["id"]])
                for i, e in enumerate(adapter._edges)
            ),
            rate_k_s=float(np.max(abs(np.array(result["rate"][:n]) - rhs[:n]))),
            power_w=abs(result["rate"][-1] - rhs[-1] * independent.scale),
            relative_algebraic_jacobian=jacobian_error(adapter, state, result["rate"]),
        )
        for name, key in [
            ("pressure_pa", "maximum_reference_pressure_error_pa"),
            ("flow_kg_s", "maximum_reference_flow_error_kg_s"),
            ("rate_k_s", "maximum_reference_rate_error_k_s"),
            ("power_w", "maximum_reference_power_error_w"),
            ("relative_algebraic_jacobian", "maximum_relative_algebraic_jacobian_error"),
        ]:
            if not np.isfinite(errors[name]) or errors[name] > budget[key]:
                raise ValueError("Adapter reference budget exceeded: " + name)
        contract = adapter.contract()
        for link in contract["links"]:
            a, b = (result["ports"][link[k]] for k in ("left", "right"))
            if a["pressure_pa"] != b["pressure_pa"] or any(
                a[field] + b[field] != 0 for field in ("mass_flow_kg_s", "energy_flow_w")
            ):
                raise ValueError("Adapter connection does not cancel pairwise")
        observations.append(
            dict(
                initialization=result,
                independent_pressures_pa=hyd["pressures_pa"],
                independent_flows_kg_s=hyd["flows_kg_s"],
                independent_temperature_rates_k_s=rhs[:n].tolist(),
                independent_external_power_w=float(rhs[-1] * independent.scale),
                errors=errors,
            )
        )
    return dict(contract=adapter.contract(), observations=observations)


def qualify(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    captured = signature()
    for group in ("drivers", "budgets", "qualification_drivers"):
        (folder / group).mkdir()
        for path in captured[group]:
            shutil.copyfile(path, folder / group / Path(path).name)
    records = []
    for name, spec in cases():
        path = folder / name
        path.mkdir()
        write(path / "request.json", spec)
        result = evaluate(spec)
        write(path / "result.json", result)
        records.append(dict(name=name, files={p.name: sha(p) for p in sorted(path.iterdir())}))
        print(json.dumps(dict(name=name, passed=True)), flush=True)
    if signature() != captured:
        raise ValueError("Adapter sources changed during campaign")
    report = dict(
        schema_version=1,
        passed=True,
        signature=captured,
        cases=records,
        validated=False,
        plant_referenced=False,
    )
    write(folder / "qualification.json", report)
    return report


def audit(folder):
    folder = Path(folder)
    report = json.loads((folder / "qualification.json").read_text())
    captured = signature()
    if report.get("passed") is not True or report["signature"] != captured:
        raise ValueError("Adapter source signature mismatch")
    for group in ("drivers", "budgets", "qualification_drivers"):
        for path, h in captured[group].items():
            if sha(folder / group / Path(path).name) != h:
                raise ValueError("Adapter frozen companion mismatch")
    expected = list(cases())
    if [r["name"] for r in report["cases"]] != [n for n, _ in expected]:
        raise ValueError("Incomplete adapter matrix")
    for record, (name, spec) in zip(report["cases"], expected):
        path = folder / name
        if set(record["files"]) != {"request.json", "result.json"}:
            raise ValueError("Incomplete adapter packet")
        for p, h in record["files"].items():
            if sha(path / p) != h:
                raise ValueError("Adapter packet integrity mismatch")
        if json.loads((path / "request.json").read_text()) != spec:
            raise ValueError("Adapter fixture mismatch")
        if evaluate(spec) != json.loads((path / "result.json").read_text()):
            raise ValueError("Adapter numerical replay differs")
    if signature() != captured:
        raise ValueError("Adapter sources changed during audit")
    return dict(
        schema_version=1,
        passed=True,
        signature=captured,
        cases=len(expected),
        aligned_state_comparisons=len(expected) * len(TIMES),
        checkpoint_lifts=len(expected) * (len(TIMES) - 1),
        maximum_nodes=max(len(s["volumes_m3"]) for _, s in expected),
        horizon_s=TIMES[-1],
        qualification_sha256=sha(folder / "qualification.json"),
        validated=False,
        plant_referenced=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        write(args.output, audit(args.audit))
    else:
        qualify(args.output)


if __name__ == "__main__":
    main()
