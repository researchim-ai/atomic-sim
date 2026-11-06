"""Portable controlled-core projects; inspection never upgrades physical claims."""

import argparse
import json

from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json, read_json

from .project import Project
from .release import dependency_manifest, identity, matrix_entry, sbom


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("spec")
    create.add_argument("directory")
    run = sub.add_parser("run")
    run.add_argument("project")
    run.add_argument("directory")
    run.add_argument("--resume", action="store_true")
    export = sub.add_parser("export")
    export.add_argument("project")
    export.add_argument("archive")
    imp = sub.add_parser("import")
    imp.add_argument("archive")
    imp.add_argument("directory")
    release = sub.add_parser("release-manifest")
    release.add_argument("output")
    args = p.parse_args()
    if args.command == "create":
        from pathlib import Path

        spec = Path(args.spec)
        result = {
            "revision": Project.create(
                args.directory, read_json(spec), data_root=spec.parent
            ).read()[1]
        }
    elif args.command == "run":
        result = Project(args.project).run(args.directory, resume=args.resume)
    elif args.command == "export":
        result = Project(args.project).export(args.archive)
    elif args.command == "import":
        result = {"revision": Project.import_archive(args.archive, args.directory).read()[1]}
    else:
        result = dict(
            source=identity(),
            software_version=__import__("importlib.metadata", fromlist=["version"]).version(
                "atomic-sim"
            ),
            matrix=matrix_entry(),
            dependencies=dependency_manifest(),
            sbom=sbom(),
            validated=False,
            plant_referenced=False,
        )
        atomic_json(args.output, seal(result))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
