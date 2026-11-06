"""Bounded JSON Lines transport. File access is host CLI configuration only."""

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

from atomic_agent_tools.protocol import LIMITS, ToolError, canonical
from atomic_agent_tools.server import AgentToolServer, failure


def parse(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    return json.loads(text, object_pairs_hook=unique, parse_constant=constant)


def publish(path, value):
    """Publish a complete JSON file without replacing an existing artifact."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as out:
            temporary = Path(out.name)
            out.write(canonical(value) + "\n")
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, help="Host-only full checkpoint input")
    parser.add_argument("--checkpoint-out", type=Path, help="Host-only exclusive output at EOF")
    parser.add_argument("--session-id", default="episode")
    args = parser.parse_args(argv)
    if args.checkpoint_out is not None and args.checkpoint_out.exists():
        parser.error("Checkpoint output already exists")
    server = AgentToolServer()
    try:
        if args.resume is not None:
            with args.resume.open("rb") as stream:
                raw = stream.read(LIMITS["max_checkpoint_bytes"] + 1)
            if len(raw) > LIMITS["max_checkpoint_bytes"]:
                raise ValueError("Checkpoint input exceeds limit")
            server.import_session(args.session_id, parse(raw.decode("utf-8")))
        count = 0
        while True:
            raw = sys.stdin.buffer.readline(LIMITS["max_line_bytes"] + 1)
            if not raw:
                break
            count += 1
            if count > LIMITS["max_requests"]:
                print(
                    canonical(
                        failure(None, "resource_limit", "Transport request budget exhausted")
                    ),
                    flush=True,
                )
                return 2
            if len(raw) > LIMITS["max_line_bytes"]:
                while raw and not raw.endswith(b"\n"):
                    raw = sys.stdin.buffer.readline(LIMITS["max_line_bytes"] + 1)
                result = failure(None, "request_too_large", "JSON line exceeds byte limit")
            else:
                try:
                    request = parse(raw.decode("utf-8"))
                except (ValueError, UnicodeError, RecursionError):
                    result = failure(
                        None, "invalid_json", "Expected one strict JSON object per line"
                    )
                else:
                    with contextlib.redirect_stdout(sys.stderr):
                        result = server.handle_request(request)
            print(canonical(result), flush=True)
        if args.checkpoint_out is not None:
            publish(args.checkpoint_out, server.export_session(args.session_id))
    except (OSError, ValueError, ToolError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
