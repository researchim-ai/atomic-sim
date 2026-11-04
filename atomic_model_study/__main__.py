"""Data-only study lifecycle and supervised reference evaluations."""

import argparse
from pathlib import Path

from atomic_model_reference.data import load_input
from atomic_model_study import evaluate, store
from atomic_model_study.manager import Manager
from atomic_model_workspace.common import canonical, read


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("create")
    a.add_argument("project", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("import")
    a.add_argument("snapshot", type=Path)
    a.add_argument("--output", type=Path, required=True)
    for command in (
        "inspect",
        "export",
        "preview",
        "edit",
        "attach-preview",
        "attach",
        "detach",
        "restore",
        "evaluate",
        "status",
        "summary",
        "curve",
        "recover",
        "resume",
    ):
        a = sub.add_parser(command)
        a.add_argument("study", type=Path)
        if command == "export":
            a.add_argument("--output", type=Path, required=True)
        if command in ("preview", "edit", "attach-preview", "attach"):
            a.add_argument("request", type=Path)
        if command in ("attach-preview", "attach"):
            a.add_argument("declaration", type=Path)
        if command in ("detach", "evaluate", "restore"):
            a.add_argument("--expected-revision", type=int, required=True)
            a.add_argument("--expected-sha256", required=True)
        if command in ("detach", "evaluate"):
            a.add_argument("binding")
        if command == "restore":
            a.add_argument("target_revision", type=int)
        if command == "evaluate":
            a.add_argument("--rss-limit-mib", type=int, default=1024)
        if command in ("status", "summary", "curve", "recover", "resume"):
            a.add_argument("evaluation_id")
        if command == "curve":
            a.add_argument("channel")
    a = parser.parse_args(argv)
    try:
        c = a.command
        if c == "create":
            result = store.create(a.project, a.output)
        elif c == "import":
            result = store.import_snapshot(a.snapshot, a.output)
        elif c == "inspect":
            result = store.inspect(a.study)
        elif c == "export":
            result = store.export(a.study, a.output)
        elif c in ("preview", "edit"):
            result = getattr(store, c)(a.study, read(a.request))
        elif c in ("attach-preview", "attach"):
            declaration, raw, _ = load_input(a.declaration)
            request = read(a.request)
            request["declaration"] = declaration
            fn = store.preview_data if c == "attach-preview" else store.attach_data
            result = fn(a.study, request, raw)
        elif c == "detach":
            result = store.detach(
                a.study,
                expected_revision=a.expected_revision,
                expected_checkpoint_sha256=a.expected_sha256,
                id=a.binding,
            )
        elif c == "restore":
            result = store.restore(
                a.study,
                expected_revision=a.expected_revision,
                expected_project_sha256=a.expected_sha256,
                target_revision=a.target_revision,
            )
        elif c == "evaluate":
            with Manager(a.study, rss_limit_bytes=a.rss_limit_mib * 1024**2) as manager:
                state = manager.start(
                    a.binding,
                    expected_revision=a.expected_revision,
                    expected_checkpoint_sha256=a.expected_sha256,
                )
                result = manager.wait(state["evaluation_id"])
        elif c in ("recover", "resume"):
            with Manager(a.study) as manager:
                result = (
                    manager.recover(a.evaluation_id)
                    if c == "recover"
                    else manager.wait(manager.resume(a.evaluation_id)["evaluation_id"])
                )
        elif c == "status":
            with Manager(a.study) as manager:
                result = manager.status(a.evaluation_id)
        elif c == "summary":
            result = evaluate.summary(a.study, a.evaluation_id)
        else:
            result = evaluate.curve(a.study, a.evaluation_id, a.channel)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(canonical(result))
    return (
        2
        if result.get("comparison_passed") is False
        or result.get("status")
        in ("failed", "rejected", "timed_out", "rss_limit", "cancelled", "paused")
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
