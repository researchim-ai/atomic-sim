"""Cold-wheel proof: all JSON/YAML resources, real-core project, spatial replay."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected_source")
    parser.add_argument("project_archive")
    parser.add_argument("directory")
    parser.add_argument("output")
    args = parser.parse_args()
    import yaml

    import atomic_model_research
    from atomic_model_multiphysics.agent import SpatialAgent
    from atomic_model_multiphysics.feedback import numerical_table
    from atomic_model_multiphysics.nuclear import model_from_mgxs
    from atomic_model_multiphysics.provenance import seal
    from atomic_model_multiphysics.resources import reference_mgxs
    from atomic_model_research.project import Project
    from atomic_model_research.release import environment, identity
    from atomic_model_spatial import eigenvalue
    from atomic_model_spatial.cases import slab
    from atomic_model_training.spec import atomic_json, read_json

    installed = Path(atomic_model_research.__file__).resolve()
    if not installed.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("Imported from checkout")
    if identity() != read_json(args.expected_source):
        raise ValueError("Cold source identity differs")
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=False)
    project = Project.import_archive(args.project_archive, directory / "project")
    report = project.run(directory / "run")
    if not report["payload"]["report"]["completed"]:
        raise ValueError("Installed project failed")
    reference = reference_mgxs()
    k = eigenvalue(model_from_mgxs(reference))["k"]
    agent = SpatialAgent(slab(6), numerical_table())
    agent.step(dict(thermal_power_w=5000), expected_revision=0)
    saved = json.loads(json.dumps(agent.checkpoint()))
    if SpatialAgent.restore(saved).checkpoint() != saved:
        raise ValueError("Installed spatial replay differs")
    resources = []
    root = installed.parent.parent
    for package in ["scenarios", "data", "validation"] + [
        p.name for p in root.glob("atomic_model_*") if p.is_dir()
    ]:
        for p in sorted((root / package).rglob("*")):
            if p.is_file() and p.suffix.lower() in (".json", ".yaml", ".yml"):
                raw = p.read_bytes()
                if p.suffix == ".json":
                    json.loads(raw)
                else:
                    yaml.safe_load(raw)
                resources.append(
                    dict(path=str(p.relative_to(root)), sha256=hashlib.sha256(raw).hexdigest())
                )
    if not resources or not any(r["path"].endswith(".yaml") for r in resources):
        raise ValueError("Missing installed scenarios")
    receipt = dict(
        source=identity(),
        installed_root=str(installed.parent),
        environment=environment(),
        project_run=report,
        spatial_replay_equal=True,
        evaluated_reference_k=k,
        resources=resources,
        resource_count=len(resources),
        another_machine_proven=False,
        validated=False,
        plant_referenced=False,
    )
    atomic_json(args.output, seal(receipt))
    print(
        json.dumps(
            dict(
                installed=True,
                resources=len(resources),
                spatial_replay_equal=True,
                project_completed=True,
            )
        )
    )


if __name__ == "__main__":
    main()
