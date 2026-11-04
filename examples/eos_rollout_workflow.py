"""Parallel durable seeds with global deadline/resume and native control replay."""

import argparse
from pathlib import Path

from atomic_model_rollouts import reports, store
from atomic_model_rollouts.controller import Controller
from atomic_model_workspace.common import atomic_json
from atomic_model_workspace.jobs import create_job, run_job


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path("examples/eos_workspace/project.json"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = store.default_options()
    cfg.update(seeds=[17, 18], decision_slice=2)
    folder = a.output / "campaign"
    store.create(a.project, folder, cfg)
    manager = Controller(folder)
    paused = manager.run(wall_s=0.05)
    assert paused["payload"]["status"] == "paused"
    final = manager.run(resume=True, expected_sha256=paused["sha256"])
    assert final["payload"]["status"] == "completed", final["payload"]["reason"]
    matrix = store.expected(final["payload"]["project"], cfg)
    for id, item in matrix.items():
        target = a.output / "controls" / id
        create_job(item["spec"], target)
        native = run_job(target)
        actual = store.native(folder, id)
        assert native["episode"] == actual["episode"]
    reports.export(folder, a.output / "export")
    verified = reports.verify(a.output / "export")
    assert verified["counts"]["completed"] == len(matrix)
    result = dict(
        schema_version=1,
        jobs=len(matrix),
        native_control_episodes_exact=True,
        global_deadline_resume=True,
        decision_slice=2,
        effective_workers=verified["effective_workers"],
        validated=False,
        plant_referenced=False,
    )
    atomic_json(a.output / "proof.json", result)
    print(result)


if __name__ == "__main__":
    main()
