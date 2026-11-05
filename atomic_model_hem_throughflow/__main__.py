"""Run/resume steady open channels, qualify numerics and audit HIPER screening."""

import argparse
import os
import tempfile
from pathlib import Path

from atomic_model_workspace.common import canonical, digest, keys, read

from . import benchmark, cases, workflow
from .model import signature


def publish(path, value):
    """Same-filesystem atomic link publication: never overwrite a concurrent result."""
    path = Path(path)
    data = (canonical(value) + "\n").encode()
    if len(data) > 32 * 1024**2:
        raise ValueError("Evidence size limit")
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            name = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(name, path)
        if os.name == "posix":
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("demo", "qualify", "benchmark"):
        a = sub.add_parser(name)
        a.add_argument("--output", type=Path, required=True)
        if name == "benchmark":
            a.add_argument("--require-pass", action="store_true")
    a = sub.add_parser("run")
    a.add_argument("input", type=Path)
    a.add_argument("--output", type=Path, required=True)
    for name in ("audit-qualification", "audit-benchmark", "replay"):
        sub.add_parser(name).add_argument("input", type=Path)
    a = p.parse_args(argv)
    try:
        if hasattr(a, "output") and a.output.exists():
            raise FileExistsError(a.output)
        if a.command == "qualify":
            result = workflow.qualify()
        elif a.command == "benchmark":
            result = benchmark.evaluate()
        elif a.command.startswith("audit-"):
            result = workflow.audit(a.input, a.command.removeprefix("audit-"))
        elif a.command == "replay":
            record = read(a.input)
            keys(record, ("payload", "sha256"))
            saved = record["payload"]
            keys(saved, ("schema_version", "source", "input", "result"))
            if type(saved["schema_version"]) is not int or saved["schema_version"] != 1:
                raise ValueError("Run schema1 required")
            if record["sha256"] != digest(saved) or saved["source"] != signature():
                raise ValueError("Run digest/source mismatch")
            spec = saved["input"]
            keys(spec, ("request", "targets_m", "checkpoint"))
            if not isinstance(spec["targets_m"], list) or not 1 <= len(spec["targets_m"]) <= 128:
                raise ValueError("Run target budget")
            actual = workflow.execute(spec["request"], spec["targets_m"], spec["checkpoint"])
            result = dict(saved, result=actual)
            if digest(result) != record["sha256"]:
                raise ValueError("Run full replay mismatch")
        else:
            if a.command == "demo":
                request = cases.baseline()
                spec = dict(
                    request=request, targets_m=[2.0, 4.0, request["length_m"]], checkpoint=None
                )
            else:
                spec = read(a.input)
            keys(spec, ("request", "targets_m", "checkpoint"))
            if not isinstance(spec["targets_m"], list) or not 1 <= len(spec["targets_m"]) <= 128:
                raise ValueError("Run target budget")
            result = dict(
                schema_version=1,
                source=signature(),
                input=spec,
                result=workflow.execute(spec["request"], spec["targets_m"], spec["checkpoint"]),
            )
        if hasattr(a, "output"):
            publish(a.output, dict(payload=result, sha256=digest(result)))
        print(
            dict(
                command=a.command,
                case_count=result.get("case_count"),
                screening_status=result.get("screening_status"),
                validated=False,
            )
        )
        if (
            a.command == "benchmark"
            and a.require_pass
            and result["screening_status"] != "passed_unvalidated"
        ):
            return 3
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
