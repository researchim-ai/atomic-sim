"""Run bounded JSON policy cases; exit nonzero for incomplete campaigns."""

import argparse
import json

from .runner import Campaign
from .spec import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases")
    parser.add_argument("directory")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--memory-mb", type=int, default=512)
    parser.add_argument("--deadline-s", type=float, default=60)
    parser.add_argument("--cpu-s", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = Campaign(
        read_json(args.cases),
        args.directory,
        workers=args.workers,
        memory_mb=args.memory_mb,
        deadline_s=args.deadline_s,
        cpu_s=args.cpu_s,
    ).run(resume=args.resume)
    print(
        json.dumps(
            dict(
                completed=report["completed"],
                statuses={r["id"]: r["status"] for r in report["results"]},
            )
        )
    )
    return 0 if report["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
