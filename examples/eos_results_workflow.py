"""Multiple policy/reference runs, portable trajectories and native exact replay."""

import argparse
import copy
from pathlib import Path

from atomic_model_results import archive, bundle
from atomic_model_study import store
from atomic_model_study.manager import Manager
from atomic_model_workspace.common import atomic_json, read
from examples.eos_study_workflow import create_fixture


def create_runs(references, output):
    references, output = Path(references), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    folder = output / "study"
    store.create(create_fixture(references, output / "input"), folder)
    v = store.inspect(folder)
    policy = copy.deepcopy(v["project"]["policies"]["hold"])
    policy["heat_power_w"]["wall.w0"] = 500.0
    store.edit(
        folder,
        dict(
            schema_version=1,
            expected_revision=v["revision"],
            expected_project_sha256=v["project_sha256"],
            operations=[dict(op="add_policy", id="heated", value=policy)],
            note="synthetic policy comparison",
        ),
    )
    for name in ["energy", "thermal"]:
        for policy_id in ["hold", "heated"]:
            v = store.inspect(folder)
            declaration = read(
                references / ("energy_reference.json" if name == "energy" else "reference.json")
            )
            declaration["reference_id"] = name + "-" + policy_id
            declaration["problem"]["policy"] = v["project"]["policies"][policy_id]
            if name == "thermal":
                declaration["criteria"]["events"] = [
                    dict(
                        id="wall-threshold",
                        channel="wall_temperature",
                        threshold=300.2,
                        direction="up",
                        max_time_error_s=0.005,
                    )
                ]
            raw = (references / declaration["csv"]["path"]).read_bytes()
            store.attach_data(
                folder,
                dict(
                    schema_version=1,
                    expected_revision=v["revision"],
                    expected_checkpoint_sha256=v["checkpoint_sha256"],
                    id=name + "-" + policy_id,
                    scenario_id="uniform_dev",
                    policy_id=policy_id,
                    declaration=declaration,
                    note="synthetic software comparison; not measurements",
                ),
                raw,
            )
    states = {}
    with Manager(folder) as manager:
        for binding in store.inspect(folder)["references"]:
            v = store.inspect(folder)
            state = manager.start(
                binding,
                expected_revision=v["revision"],
                expected_checkpoint_sha256=v["checkpoint_sha256"],
            )
            state = manager.wait(state["evaluation_id"])
            assert state["status"] == "completed", state
            states[binding] = state
    atomic_json(output / "evaluations.json", states)
    return folder, states


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--references", type=Path, default=Path("examples/eos_reference"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    folder, states = create_runs(a.references, a.output)
    bundle.export(folder, [s["evaluation_id"] for s in states.values()], a.output / "results")
    summary = bundle.verify(a.output / "results")
    assert len(summary["groups"]) == 2
    assert all(len(g["evaluations"]) == 2 for g in summary["groups"].values())
    pair = bundle.pair(
        a.output / "results",
        states["thermal-hold"]["evaluation_id"],
        states["thermal-heated"]["evaluation_id"],
        "wall_temperature",
    )
    assert any(value != 0 for value in pair["difference"])
    atomic_json(a.output / "pair.json", pair)
    archive.pack(a.output / "results", a.output / "results.zip")
    archive.pack(a.output / "results", a.output / "results-rebuilt.zip")
    assert (a.output / "results.zip").read_bytes() == (
        a.output / "results-rebuilt.zip"
    ).read_bytes()
    archive.unpack(a.output / "results.zip", a.output / "imported")
    assert bundle.verify(a.output / "imported") == summary
    bundle.replay(a.output / "imported", a.output / "replayed")
    proof = dict(
        schema_version=1,
        evaluations=4,
        protocol_groups=2,
        native_and_metric_replay_exact=True,
        deterministic_zip=True,
        mixed_unit_ranking=False,
        validated=False,
        plant_referenced=False,
    )
    atomic_json(a.output / "proof.json", proof)
    print(proof)


if __name__ == "__main__":
    main()
