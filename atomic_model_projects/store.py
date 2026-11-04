"""CAS project revisions with full preflight, immutable history and atomic HEAD."""

import hashlib
import os
import shutil
from copy import deepcopy
from pathlib import Path

from atomic_model_runtime.episode import EOSEpisode, model_card
from atomic_model_workspace.bundle import _project_files
from atomic_model_workspace.common import (
    Lock,
    atomic_json,
    canonical,
    digest,
    identifier,
    keys,
    local_file,
    read,
)
from atomic_model_workspace.project import load_project, normalize, provenance

MAX_REVISIONS = 64


def source_identity():
    return dict(
        editor_sha256=digest(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            }
        ),
        workspace=provenance(),
    )


def normalized(value):
    keys(value, ("schema_version", "project_id", "model_id", "scenarios", "policies", "limits"))
    return normalize(value, value["scenarios"], value["policies"])


def _envelope(payload):
    return dict(payload=payload, sha256=digest(payload))


def _publish(folder, payload):
    value = _envelope(payload)
    name = "revisions/" + value["sha256"] + ".json"
    path = Path(folder) / name
    data = (canonical(value) + "\n").encode()
    try:
        with path.open("xb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError("Revision collision or corruption")
    if os.name == "posix":
        fd = os.open(Path(folder) / "revisions", os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    atomic_json(
        Path(folder) / "HEAD.json",
        dict(
            schema_version=1, revision=payload["revision"], checkpoint=name, sha256=value["sha256"]
        ),
    )
    return value


def _read_revision(folder, name):
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
            "source",
            "note",
            "operation",
        ),
    )
    if digest(p) != value["sha256"] or name != "revisions/" + value["sha256"] + ".json":
        raise ValueError("Revision checksum/path mismatch")
    if (
        type(p["schema_version"]) is not int
        or p["schema_version"] != 1
        or type(p["revision"]) is not int
        or not 0 <= p["revision"] < MAX_REVISIONS
        or not isinstance(p["note"], str)
        or len(p["note"]) > 512
    ):
        raise ValueError("Unsupported revision schema or bounds")
    operation = p["operation"]
    if (
        not isinstance(operation, str)
        or (p["revision"] == 0 and operation != "create")
        or (
            p["revision"] > 0
            and operation != "edit"
            and not (
                operation.startswith("restore-")
                and operation[8:].isdigit()
                and int(operation[8:]) < p["revision"] - 1
            )
        )
    ):
        raise ValueError("Unsupported revision operation")
    project = normalized(p["project"])
    if project != p["project"] or digest(project) != p["project_sha256"]:
        raise ValueError("Revision project identity mismatch")
    return value


def load(folder):
    head = read(Path(folder) / "HEAD.json")
    keys(head, ("schema_version", "revision", "checkpoint", "sha256"))
    if (
        type(head["schema_version"]) is not int
        or head["schema_version"] != 1
        or type(head["revision"]) is not int
        or not 0 <= head["revision"] < MAX_REVISIONS
    ):
        raise ValueError("Unsupported project HEAD")
    value = _read_revision(folder, head["checkpoint"])
    if value["sha256"] != head["sha256"] or value["payload"]["revision"] != head["revision"]:
        raise ValueError("HEAD identity mismatch")
    current = value
    chain = []
    source = source_identity()
    while True:
        p = current["payload"]
        if p["source"] != source:
            raise ValueError("Editor/workspace source changed; explicit migration required")
        chain.append(current)
        if p["revision"] == 0:
            if p["parent_sha256"] is not None:
                raise ValueError("Initial revision has parent")
            break
        parent = p["parent_sha256"]
        if (
            not isinstance(parent, str)
            or len(parent) != 64
            or any(c not in "0123456789abcdef" for c in parent)
        ):
            raise ValueError("Invalid parent identity")
        current = _read_revision(folder, "revisions/" + parent + ".json")
        if current["payload"]["revision"] != p["revision"] - 1:
            raise ValueError("Broken revision chain")
    return value, chain


def create(project_path, folder):
    project = load_project(project_path)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "revisions").mkdir()
    p = dict(
        schema_version=1,
        revision=0,
        parent_sha256=None,
        project=project,
        project_sha256=digest(project),
        source=source_identity(),
        note="imported portable project",
        operation="create",
    )
    with Lock(folder):
        return _publish(folder, p)


def _guard(value, revision, project_sha256):
    if (
        type(revision) is not int
        or revision != value["payload"]["revision"]
        or project_sha256 != value["payload"]["project_sha256"]
    ):
        raise ValueError("Stale project revision or content identity")
    if revision + 1 >= MAX_REVISIONS:
        raise ValueError("Project revision budget exhausted")


def _changes(before, after, path=()):
    if isinstance(before, dict) and isinstance(after, dict):
        rows = []
        for key in sorted(set(before) | set(after)):
            if key not in before:
                rows.append(
                    dict(
                        path=list((*path, key)),
                        before_present=False,
                        after_present=True,
                        before=None,
                        after=after[key],
                    )
                )
            elif key not in after:
                rows.append(
                    dict(
                        path=list((*path, key)),
                        before_present=True,
                        after_present=False,
                        before=before[key],
                        after=None,
                    )
                )
            else:
                rows.extend(_changes(before[key], after[key], (*path, key)))
        return rows
    return (
        []
        if before == after
        else [
            dict(
                path=list(path), before_present=True, after_present=True, before=before, after=after
            )
        ]
    )


def _edited(project, operations):
    if not isinstance(operations, list) or not 1 <= len(operations) <= 64:
        raise ValueError("Edit requires 1..64 operations")
    result = deepcopy(project)
    seen = set()
    for op in operations:
        if not isinstance(op, dict):
            raise ValueError("Edit operation must be an object")
        kind = op.get("op")
        groups = {
            "set_scenario_config": "scenarios",
            "set_objective": "scenarios",
            "replace_scenario": "scenarios",
            "add_scenario": "scenarios",
            "remove_scenario": "scenarios",
            "replace_policy": "policies",
            "add_policy": "policies",
            "remove_policy": "policies",
        }
        if not isinstance(kind, str) or kind not in groups:
            raise ValueError("Unsupported edit operation")
        group = groups[kind]
        remove = kind.startswith("remove_")
        add = kind.startswith("add_")
        keys(op, ("op", "id") if remove else ("op", "id", "value"))
        name = identifier(op["id"])
        if (group, name) in seen:
            raise ValueError("At most one operation per scenario/policy in a transaction")
        seen.add((group, name))
        exists = name in result[group]
        if (add and exists) or (not add and not exists):
            raise ValueError("Missing or duplicate project object")
        if remove:
            del result[group][name]
        elif kind == "set_scenario_config":
            if not isinstance(op["value"], dict) or not op["value"]:
                raise ValueError("Config patch must be a nonempty object")
            result[group][name]["config"].update(deepcopy(op["value"]))
        elif kind == "set_objective":
            result[group][name]["objective"] = deepcopy(op["value"])
        else:
            result[group][name] = deepcopy(op["value"])
    return normalized(result)


def _prepare(value, request):
    keys(
        request,
        ("schema_version", "expected_revision", "expected_project_sha256", "operations", "note"),
    )
    if (
        type(request["schema_version"]) is not int
        or request["schema_version"] != 1
        or not isinstance(request["note"], str)
        or len(request["note"]) > 512
    ):
        raise ValueError("Invalid edit request schema or note")
    _guard(value, request["expected_revision"], request["expected_project_sha256"])
    project = _edited(value["payload"]["project"], request["operations"])
    changes = _changes(value["payload"]["project"], project)
    if not changes:
        raise ValueError("Edit has no changes")
    return project, changes


def preview(folder, request):
    value, _ = load(folder)
    project, changes = _prepare(value, request)
    return dict(
        schema_version=1,
        base_revision=value["payload"]["revision"],
        base_project_sha256=value["payload"]["project_sha256"],
        proposed_project_sha256=digest(project),
        changes=changes,
        model=model_card(),
        project=project,
        committed=False,
    )


def edit(folder, request):
    with Lock(folder):
        value, _ = load(folder)
        project, changes = _prepare(value, request)
        old = value["payload"]
        p = dict(
            schema_version=1,
            revision=old["revision"] + 1,
            parent_sha256=value["sha256"],
            project=project,
            project_sha256=digest(project),
            source=source_identity(),
            note=request["note"],
            operation="edit",
        )
        result = _publish(folder, p)
        return dict(revision=result, changes=changes)


def restore(
    folder,
    *,
    expected_revision,
    expected_project_sha256,
    target_revision,
    note="restore configuration",
):
    with Lock(folder):
        value, chain = load(folder)
        _guard(value, expected_revision, expected_project_sha256)
        if (
            type(target_revision) is not int
            or not 0 <= target_revision < expected_revision
            or not isinstance(note, str)
            or len(note) > 512
        ):
            raise ValueError("Invalid restore target or note")
        target = next(v for v in chain if v["payload"]["revision"] == target_revision)
        project = target["payload"]["project"]
        p = dict(
            schema_version=1,
            revision=expected_revision + 1,
            parent_sha256=value["sha256"],
            project=project,
            project_sha256=digest(project),
            source=source_identity(),
            note=note,
            operation="restore-" + str(target_revision),
        )
        return _publish(folder, p)


def inspect(folder):
    value, chain = load(folder)
    p = value["payload"]
    channels = {}
    for name, spec in p["project"]["scenarios"].items():
        env = EOSEpisode(spec["config"])
        channels[name] = dict(
            observations=env.observation_metadata()["channels"],
            actions=env.action_units,
            config=env.config.as_dict(),
        )
    return dict(
        schema_version=1,
        revision=p["revision"],
        project_sha256=p["project_sha256"],
        checkpoint_sha256=value["sha256"],
        project=p["project"],
        model=model_card(),
        channels=channels,
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
        supported_operations=[
            "set_scenario_config",
            "set_objective",
            "replace_scenario",
            "add_scenario",
            "remove_scenario",
            "replace_policy",
            "add_policy",
            "remove_policy",
        ],
    )


def export(folder, output):
    with Lock(folder):
        value, _ = load(folder)
        files = _project_files(value["payload"]["project"])
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        try:
            for name, data in files.items():
                path = output / name.removeprefix("project/")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            atomic_json(
                output / "editor_origin.json",
                dict(
                    schema_version=1,
                    revision=value["payload"]["revision"],
                    checkpoint_sha256=value["sha256"],
                    source=value["payload"]["source"],
                ),
            )
            if load_project(output / "project.json") != value["payload"]["project"]:
                raise ValueError("Export project identity mismatch")
        except BaseException:
            shutil.rmtree(output)
            raise
    return dict(
        project=str(output / "project.json"),
        project_sha256=value["payload"]["project_sha256"],
        revision=value["payload"]["revision"],
    )
