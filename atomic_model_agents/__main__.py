"""Host-launched explicit-state agents; executable commands never come from project files."""

import argparse
from pathlib import Path

from atomic_model_agents import jobs
from atomic_model_workspace.common import canonical, read


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["create", "run", "resume", "inspect", "cancel", "export", "replay"]:
        a = sub.add_parser(name)
        a.add_argument("folder", type=Path)
        if name == "create":
            a.add_argument("--scenario", type=Path, required=True)
            a.add_argument("--descriptor", type=Path, required=True)
        if name in ("create", "run", "resume"):
            a.add_argument("--identity-file", type=Path, action="append", default=[])
            if name == "run":
                a.add_argument("--max-decisions", type=int)
                a.add_argument("--deadline-s", type=float, default=3600)
            if name == "resume":
                a.add_argument("--expected-sha256", required=True)
            a.add_argument("--agent-command", nargs=argparse.REMAINDER, required=True)
        if name == "export":
            a.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        if a.command == "create":
            result = jobs.create(
                read(a.scenario),
                read(a.descriptor),
                a.folder,
                a.agent_command,
                identity_files=a.identity_file,
            )
        elif a.command == "run":
            result = jobs.run(
                a.folder,
                a.agent_command,
                identity_files=a.identity_file,
                max_decisions=a.max_decisions,
                deadline_s=a.deadline_s,
            )
        elif a.command == "resume":
            result = jobs.resume(
                a.folder,
                expected_sha256=a.expected_sha256,
                command=a.agent_command,
                identity_files=a.identity_file,
            )
        elif a.command == "inspect":
            result = jobs.load(a.folder)
        elif a.command == "cancel":
            jobs.cancel(a.folder)
            result = dict(cancel_requested=True)
        elif a.command == "export":
            result = jobs.export(a.folder, a.output)
        else:
            result = jobs.replay(a.folder)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.error(str(exc))
    if "payload" in result:
        value = result["payload"]
        result = dict(
            schema_version=1,
            checkpoint_sha256=result["sha256"],
            job_id=value["job_id"],
            revision=value["revision"],
            status=value["status"],
            reason=value["reason"],
            accepted_decisions=len(value["rows"]) - 1,
            agent_state=value["agent_state"],
            attempts=value["attempts"],
            source=value["source"],
        )
    print(canonical(result))
    return 2 if result.get("status") in ("failed", "rejected", "cancelled") else 0


if __name__ == "__main__":
    raise SystemExit(main())
