"""Portable model/data study with passing and failing synthetic comparisons."""

import argparse
from pathlib import Path

from atomic_model_study import evaluate, store
from atomic_model_study.manager import Manager
from atomic_model_workspace.bundle import _project_files
from atomic_model_workspace.common import atomic_json, canonical, read


def create_fixture(reference_root, output):
    declaration = read(Path(reference_root) / "energy_reference.json")
    project = dict(
        schema_version=1,
        project_id="eos-reference-study-v1",
        model_id=declaration["problem"]["model_id"],
        scenarios={
            "uniform_dev": dict(
                schema_version=1,
                config=declaration["problem"]["config"],
                seed=17,
                split="dev",
                objective=dict(sensor="wall.w0", target_k=300, tolerance_k=1),
            )
        },
        policies={"hold": declaration["problem"]["policy"]},
        limits=dict(decision_timeout_s=1, job_timeout_s=60),
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in _project_files(project).items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return output / "project/project.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=Path("examples/eos_reference"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    project = create_fixture(args.references, args.output / "input")
    folder = args.output / "study"
    store.create(project, folder)
    for id, name in [("energy", "energy_reference.json"), ("thermal", "reference.json")]:
        declaration = read(args.references / name)
        raw = (args.references / declaration["csv"]["path"]).read_bytes()
        view = store.inspect(folder)
        request = dict(
            schema_version=1,
            expected_revision=view["revision"],
            expected_checkpoint_sha256=view["checkpoint_sha256"],
            id=id,
            scenario_id="uniform_dev",
            policy_id="hold",
            declaration=declaration,
            note="synthetic software regression only",
        )
        preview = store.preview_data(folder, request, raw)
        assert preview["committed"] is False
        store.attach_data(folder, request, raw)
    states = {}
    with Manager(folder) as manager:
        for id in ["energy", "thermal"]:
            view = store.inspect(folder)
            state = manager.start(
                id,
                expected_revision=view["revision"],
                expected_checkpoint_sha256=view["checkpoint_sha256"],
            )
            state = manager.wait(state["evaluation_id"])
            assert state["status"] == "completed", state
            states[id] = state
            report = evaluate.summary(folder, state["evaluation_id"])
            assert report["passed"] is (id == "energy")
            channel = next(iter(report["channels"]))
            atomic_json(
                args.output / (id + "-curve.json"),
                evaluate.curve(folder, state["evaluation_id"], channel),
            )
    exported = store.export(folder, args.output / "snapshot")
    store.import_snapshot(exported["study"], args.output / "imported")
    assert (
        store.load(folder)[0]["payload"]["references"]
        == store.load(args.output / "imported")[0]["payload"]["references"]
    )
    with Manager(args.output / "imported") as manager:
        view = store.inspect(args.output / "imported")
        state = manager.start(
            "energy", expected_revision=0, expected_checkpoint_sha256=view["checkpoint_sha256"]
        )
        rerun = manager.wait(state["evaluation_id"])
        assert rerun["comparison_passed"] is True
        old = evaluate.summary(folder, states["energy"]["evaluation_id"])
        new = evaluate.summary(args.output / "imported", state["evaluation_id"])
        assert old["channels"] == new["channels"] and old["events"] == new["events"]
    result = dict(
        schema_version=1,
        evaluations=states,
        portable_metrics_exact=True,
        validated=False,
        plant_referenced=False,
    )
    atomic_json(args.output / "proof.json", result)
    print(canonical(result))


if __name__ == "__main__":
    main()
