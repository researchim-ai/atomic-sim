"""Atomic project/data revisions; imported objects are immutable and content-addressed."""

import hashlib
import json
import os
import secrets
import shutil
import tempfile
from pathlib import Path

from atomic_model_projects import store as editor
from atomic_model_reference.bundle import check_catalog, import_reference, load_bundle
from atomic_model_reference.bundle import source_identity as reference_source
from atomic_model_reference.data import prepare, problem
from atomic_model_workspace.bundle import _project_files
from atomic_model_workspace.common import (
    Lock,
    atomic_json,
    canonical,
    digest,
    identifier,
    keys,
    local_file,
    portable,
    read,
)
from atomic_model_workspace.project import load_project

MAX_REVISIONS = 64
ROLE = {"train": "calibration", "dev": "validation", "test": "test"}


def source_identity():
    return dict(
        study_code_sha256=digest(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            }
        ),
        editor=editor.source_identity(),
        reference=reference_source(),
    )


def _hash(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("Expected SHA-256 identity")
    return value


def object_path(folder, sha):
    return Path(folder) / "objects" / _hash(sha)


def _binding(project, binding, ref):
    keys(binding, ("object_sha256", "scenario_id", "policy_id"))
    _hash(binding["object_sha256"])
    scenario = project["scenarios"].get(binding["scenario_id"])
    policy = project["policies"].get(binding["policy_id"])
    if scenario is None or policy is None:
        raise ValueError("Reference binding names an unknown scenario/policy")
    spec, _, _ = problem(
        dict(model_id=project["model_id"], config=scenario["config"], policy=policy)
    )
    if digest(spec) != ref["problem_sha256"]:
        raise ValueError(
            "Reference binding does not match model problem; detach/redeclare explicitly"
        )
    if ROLE[scenario["split"]] != ref["role"]:
        raise ValueError("Scenario split does not match reference data role")


def validate_state(folder, project, references):
    project = editor.normalized(project)
    if not isinstance(references, dict) or len(references) > 32:
        raise ValueError("Study reference binding count limit")
    objects = {}
    metadata = {}
    for name, binding in references.items():
        identifier(name)
        keys(binding, ("object_sha256", "scenario_id", "policy_id"))
        sha = _hash(binding["object_sha256"])
        if sha not in objects:
            path = local_file(folder, "objects/" + sha + "/manifest.json").parent
            ref, manifest = load_bundle(path)
            if digest(manifest) != sha:
                raise ValueError("Reference object content identity mismatch")
            objects[sha] = (ref, manifest, path)
        ref, manifest, path = objects[sha]
        _binding(project, binding, ref)
        metadata[name] = dict(
            **binding,
            reference_id=ref["reference_id"],
            physical_case_id=ref["physical_case_id"],
            role=ref["role"],
            kind=ref["kind"],
            provenance=ref["provenance"],
            signals={
                k: dict(signal=c["signal"], unit=c["canonical_unit"])
                for k, c in ref["channels"].items()
            },
            criteria=ref["criteria"],
        )
    if objects:
        check_catalog([v[2] for v in objects.values()])
    return project, metadata


def load(folder):
    folder = Path(folder)
    head = read(folder / "HEAD.json")
    keys(head, ("schema_version", "revision", "checkpoint", "sha256"))
    if (
        type(head["schema_version"]) is not int
        or head["schema_version"] != 1
        or type(head["revision"]) is not int
        or not 0 <= head["revision"] < MAX_REVISIONS
    ):
        raise ValueError("Unknown study HEAD schema")
    expected = _hash(head["sha256"])
    revision = head["revision"]
    chain = []
    source = source_identity()
    while True:
        name = "revisions/" + expected + ".json"
        if not chain and head["checkpoint"] != name:
            raise ValueError("Study HEAD path mismatch")
        value = read(local_file(folder, name))
        keys(value, ("payload", "sha256"))
        p = value["payload"]
        keys(
            p,
            (
                "schema_version",
                "revision",
                "parent_sha256",
                "project",
                "project_sha256",
                "references",
                "source",
                "note",
                "operation",
            ),
        )
        if (
            value["sha256"] != expected
            or digest(p) != expected
            or type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or type(p["revision"]) is not int
            or p["revision"] != revision
            or p["source"] != source
        ):
            raise ValueError("Study revision hash/schema/source mismatch")
        if (
            not isinstance(p["note"], str)
            or len(p["note"]) > 512
            or p["operation"] not in ("create", "edit", "attach", "detach", "restore")
        ):
            raise ValueError("Unknown study operation or note")
        if (
            digest(p["project"]) != p["project_sha256"]
            or not isinstance(p["references"], dict)
            or len(p["references"]) > 32
        ):
            raise ValueError("Study project/reference identity mismatch")
        for key, binding in p["references"].items():
            identifier(key)
            keys(binding, ("object_sha256", "scenario_id", "policy_id"))
            _hash(binding["object_sha256"])
            identifier(binding["scenario_id"])
            identifier(binding["policy_id"])
        chain.append(value)
        if revision == 0:
            if p["parent_sha256"] is not None or p["operation"] != "create":
                raise ValueError("Invalid initial study revision")
            break
        expected = _hash(p["parent_sha256"])
        revision -= 1
    current = chain[0]
    project, metadata = validate_state(
        folder, current["payload"]["project"], current["payload"]["references"]
    )
    if canonical(project) != canonical(current["payload"]["project"]):
        raise ValueError("Noncanonical study project")
    return current, chain, metadata


def _guard(value, revision, checkpoint=None, project_sha256=None):
    if (
        type(revision) is not int
        or revision != value["payload"]["revision"]
        or (checkpoint is not None and checkpoint != value["sha256"])
        or (project_sha256 is not None and project_sha256 != value["payload"]["project_sha256"])
    ):
        raise ValueError("Stale study revision/content identity")
    if revision + 1 >= MAX_REVISIONS:
        raise ValueError("Study revision budget exhausted")


def _commit(folder, value, project, refs, note, operation):
    if not isinstance(note, str) or len(note) > 512:
        raise ValueError("Study note limit")
    project, _ = validate_state(folder, project, refs)
    old = value["payload"]
    p = dict(
        schema_version=1,
        revision=old["revision"] + 1,
        parent_sha256=value["sha256"],
        project=project,
        project_sha256=digest(project),
        references=refs,
        source=old["source"],
        note=note,
        operation=operation,
    )
    if source_identity() != old["source"]:
        raise ValueError("Study source changed during transaction")
    return editor._publish(folder, p)


def create(project_path, folder):
    project = load_project(project_path)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    for name in ["revisions", "objects", "evaluations"]:
        (folder / name).mkdir()
    p = dict(
        schema_version=1,
        revision=0,
        parent_sha256=None,
        project=project,
        project_sha256=digest(project),
        references={},
        source=source_identity(),
        note="created model/data study",
        operation="create",
    )
    with Lock(folder):
        return editor._publish(folder, p)


def _copy_bundle(source, target):
    _, manifest = load_bundle(source)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    try:
        for name in (*manifest["artifacts"], "manifest.json"):
            shutil.copy2(local_file(source, name), target / name)
            with (target / name).open("rb") as stream:
                os.fsync(stream.fileno())
        fd = os.open(target, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        _, copied = load_bundle(target)
        if copied != manifest:
            raise ValueError("Reference changed during copy")
    except BaseException:
        shutil.rmtree(target)
        raise


def _data_object(spec, raw, target):
    portable(spec["csv"]["path"])
    if len(spec["csv"]["path"]) > 1024 or len(Path(spec["csv"]["path"]).parts) > 16:
        raise ValueError("Reference logical path limit")
    prepare(spec, raw)
    with tempfile.TemporaryDirectory(prefix="atomic-study-data-") as tmp:
        root = Path(tmp)
        data = root / spec["csv"]["path"]
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_bytes(raw)
        declaration = root / (secrets.token_hex(16) + ".json")
        atomic_json(declaration, spec)
        import_reference(declaration, root / "bundle")
        ref, manifest = load_bundle(root / "bundle")
        sha = digest(manifest)
        destination = object_path(target, sha)
        if destination.exists():
            old, old_manifest = load_bundle(destination)
            if old_manifest != manifest or old != ref:
                raise ValueError("Immutable object collision")
        else:
            staged = Path(target) / "objects" / (".staging-" + secrets.token_hex(8))
            _copy_bundle(root / "bundle", staged)
            try:
                staged.rename(destination)
            finally:
                if staged.exists():
                    shutil.rmtree(staged)
        return sha, ref


def _attach(folder, request, raw, commit):
    keys(
        request,
        (
            "schema_version",
            "expected_revision",
            "expected_checkpoint_sha256",
            "id",
            "scenario_id",
            "policy_id",
            "declaration",
            "note",
        ),
    )
    if type(request["schema_version"]) is not int or request["schema_version"] != 1:
        raise ValueError("Expected study data schema1")
    identifier(request["id"])
    identifier(request["scenario_id"])
    identifier(request["policy_id"])
    if not isinstance(request["note"], str) or len(request["note"]) > 512:
        raise ValueError("Study note limit")
    # Preview uses an isolated store shell: no orphan objects in the live store.
    value, _, _ = load(folder)
    _guard(value, request["expected_revision"], request["expected_checkpoint_sha256"])
    if request["id"] in value["payload"]["references"]:
        raise ValueError("Reference binding ID already exists")
    with tempfile.TemporaryDirectory(prefix="atomic-study-preview-") as tmp:
        stage = Path(tmp)
        (stage / "objects").mkdir()
        sha, ref = _data_object(request["declaration"], raw, stage)
        binding = dict(
            object_sha256=sha, scenario_id=request["scenario_id"], policy_id=request["policy_id"]
        )
        _binding(value["payload"]["project"], binding, ref)
        current_paths = [
            object_path(folder, s)
            for s in sorted({b["object_sha256"] for b in value["payload"]["references"].values()})
        ]
        if any(load_bundle(p)[0]["reference_id"] == ref["reference_id"] for p in current_paths):
            raise ValueError("Reference ID already bound; use a distinct declared reference")
        check_catalog([*current_paths, object_path(stage, sha)])
        preview = dict(
            schema_version=1,
            committed=False,
            base_revision=value["payload"]["revision"],
            base_checkpoint_sha256=value["sha256"],
            object_sha256=sha,
            reference_id=ref["reference_id"],
            physical_case_id=ref["physical_case_id"],
            role=ref["role"],
            kind=ref["kind"],
            provenance=ref["provenance"],
            channels=ref["channels"],
            criteria=ref["criteria"],
            scenario_id=request["scenario_id"],
            policy_id=request["policy_id"],
        )
        if not commit:
            return preview
        with Lock(folder):
            latest, _, _ = load(folder)
            _guard(latest, request["expected_revision"], request["expected_checkpoint_sha256"])
            destination = object_path(folder, sha)
            if not destination.exists():
                _copy_bundle(object_path(stage, sha), destination)
            refs = dict(latest["payload"]["references"])
            refs[request["id"]] = binding
            result = _commit(
                folder, latest, latest["payload"]["project"], refs, request["note"], "attach"
            )
        return dict(revision=result, reference=preview, committed=True)


def preview_data(folder, request, raw):
    return _attach(folder, request, raw, False)


def attach_data(folder, request, raw):
    return _attach(folder, request, raw, True)


def detach(folder, *, expected_revision, expected_checkpoint_sha256, id):
    identifier(id)
    with Lock(folder):
        value, _, _ = load(folder)
        _guard(value, expected_revision, expected_checkpoint_sha256)
        refs = dict(value["payload"]["references"])
        if id not in refs:
            raise ValueError("Unknown reference binding")
        del refs[id]
        return _commit(folder, value, value["payload"]["project"], refs, "detached " + id, "detach")


def preview(folder, request):
    value, _, _ = load(folder)
    project, changes = editor._prepare(value, request)
    validate_state(folder, project, value["payload"]["references"])
    return dict(
        schema_version=1,
        base_revision=value["payload"]["revision"],
        base_project_sha256=value["payload"]["project_sha256"],
        proposed_project_sha256=digest(project),
        changes=changes,
        model=editor.model_card(),
        project=project,
        committed=False,
    )


def edit(folder, request):
    with Lock(folder):
        value, _, _ = load(folder)
        project, changes = editor._prepare(value, request)
        result = _commit(
            folder, value, project, value["payload"]["references"], request["note"], "edit"
        )
        return dict(revision=result, changes=changes)


def restore(folder, *, expected_revision, expected_project_sha256, target_revision):
    with Lock(folder):
        value, chain, _ = load(folder)
        _guard(value, expected_revision, project_sha256=expected_project_sha256)
        if type(target_revision) is not int or not 0 <= target_revision < expected_revision:
            raise ValueError("Invalid study restore target")
        target = next(v["payload"] for v in chain if v["payload"]["revision"] == target_revision)
        return _commit(
            folder,
            value,
            target["project"],
            target["references"],
            "restored model/data revision " + str(target_revision),
            "restore",
        )


def inspect(folder):
    value, chain, metadata = load(folder)
    p = value["payload"]
    channels = {}
    for name, spec in p["project"]["scenarios"].items():
        env = editor.EOSEpisode(spec["config"])
        channels[name] = dict(
            observations=env.observation_metadata()["channels"],
            actions=env.action_units,
            config=env.config.as_dict(),
        )
    return dict(
        schema_version=1,
        study=True,
        revision=p["revision"],
        project_sha256=p["project_sha256"],
        checkpoint_sha256=value["sha256"],
        project=p["project"],
        model=editor.model_card(),
        channels=channels,
        references=metadata,
        history=[
            dict(
                revision=v["payload"]["revision"],
                sha256=v["sha256"],
                note=v["payload"]["note"],
                operation=v["payload"]["operation"],
                project_sha256=v["payload"]["project_sha256"],
            )
            for v in chain
        ],
        source=source_identity(),
    )


def export(folder, output):
    with Lock(folder):
        value, _, _ = load(folder)
        p = value["payload"]
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        try:
            for name, data in _project_files(p["project"]).items():
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            for sha in sorted({b["object_sha256"] for b in p["references"].values()}):
                path = output / "references" / sha
                path.parent.mkdir(parents=True, exist_ok=True)
                _copy_bundle(object_path(folder, sha), path)
            refs = {
                key: dict(**b, bundle="references/" + b["object_sha256"])
                for key, b in p["references"].items()
            }
            atomic_json(
                output / "study.json",
                dict(
                    schema_version=1,
                    format="atomic-model-data-study-v1",
                    project="project/project.json",
                    references=refs,
                    origin=dict(
                        revision=p["revision"],
                        checkpoint_sha256=value["sha256"],
                        source=p["source"],
                    ),
                ),
            )
            atomic_json(
                output / "manifest.json",
                dict(
                    schema_version=1,
                    artifacts={
                        str(x.relative_to(output)): hashlib.sha256(x.read_bytes()).hexdigest()
                        for x in sorted(output.rglob("*"))
                        if x.is_file()
                    },
                    validated=False,
                    plant_referenced=False,
                ),
            )
        except BaseException:
            shutil.rmtree(output)
            raise
    return dict(
        project=str(output / "project/project.json"),
        study=str(output / "study.json"),
        project_sha256=p["project_sha256"],
        revision=p["revision"],
    )


def import_snapshot(path, folder):
    path = Path(path)
    spec = read(path)
    keys(spec, ("schema_version", "format", "project", "references", "origin"))
    if (
        type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or spec["format"] != "atomic-model-data-study-v1"
    ):
        raise ValueError("Unknown portable study schema")
    if path.name != "study.json" or spec["project"] != "project/project.json":
        raise ValueError("Noncanonical study entry point")
    inventory = read(local_file(path.parent, "manifest.json"))
    keys(inventory, ("schema_version", "artifacts", "validated", "plant_referenced"))
    if (
        type(inventory["schema_version"]) is not int
        or inventory["schema_version"] != 1
        or inventory["validated"] is not False
        or inventory["plant_referenced"] is not False
        or not isinstance(inventory["artifacts"], dict)
        or len(inventory["artifacts"]) > 10000
    ):
        raise ValueError("Snapshot inventory schema")
    total = 0
    for name, sha in inventory["artifacts"].items():
        _hash(sha)
        with local_file(path.parent, name).open("rb") as stream:
            data = stream.read(32 * 1024 * 1024 + 1)
        total += len(data)
        if (
            len(data) > 32 * 1024 * 1024
            or total > 1024**3
            or hashlib.sha256(data).hexdigest() != sha
        ):
            raise ValueError("Snapshot bytes/hash limit mismatch")
    project = load_project(local_file(path.parent, spec["project"]))
    if not isinstance(spec["references"], dict) or len(spec["references"]) > 32:
        raise ValueError("Reference binding count limit")
    # Validate references and model/data coherence in staging before publishing output.
    with tempfile.TemporaryDirectory(prefix="atomic-study-import-") as tmp:
        stage = Path(tmp)
        (stage / "objects").mkdir()
        refs = {}
        for name, b in spec["references"].items():
            identifier(name)
            keys(b, ("object_sha256", "scenario_id", "policy_id", "bundle"))
            if b["bundle"] != "references/" + _hash(b["object_sha256"]):
                raise ValueError("Reference snapshot path mismatch")
            source = local_file(path.parent, b["bundle"] + "/manifest.json").parent
            ref, manifest = load_bundle(source)
            if digest(manifest) != _hash(b["object_sha256"]):
                raise ValueError("Portable reference object mismatch")
            target = object_path(stage, b["object_sha256"])
            if not target.exists():
                _copy_bundle(source, target)
            refs[name] = {k: b[k] for k in ("object_sha256", "scenario_id", "policy_id")}
        expected_files = {"study.json", *_project_files(project)}
        for sha in {b["object_sha256"] for b in refs.values()}:
            _, manifest = load_bundle(object_path(stage, sha))
            expected_files.update(
                "references/" + sha + "/" + name
                for name in (*manifest["artifacts"], "manifest.json")
            )
        if set(inventory["artifacts"]) != expected_files:
            raise ValueError("Snapshot has missing or untracked artifacts")
        for name, raw in _project_files(project).items():
            if read(local_file(path.parent, name)) != json.loads(raw):
                raise ValueError("Noncanonical project reference layout")
        validate_state(stage, project, refs)
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        try:
            for name in ["revisions", "evaluations"]:
                (folder / name).mkdir()
            shutil.copytree(stage / "objects", folder / "objects")
            p = dict(
                schema_version=1,
                revision=0,
                parent_sha256=None,
                project=project,
                project_sha256=digest(project),
                references=refs,
                source=source_identity(),
                note="imported model/data snapshot " + digest(spec["origin"]),
                operation="create",
            )
            with Lock(folder):
                return editor._publish(folder, p)
        except BaseException:
            shutil.rmtree(folder)
            raise
