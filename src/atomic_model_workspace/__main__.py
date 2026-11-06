"""Headless portable EOS projects, jobs, campaigns, reports and data archives."""

import argparse
from pathlib import Path

from atomic_model_workspace.bundle import export_bundle, import_bundle, verify_bundle
from atomic_model_workspace.campaign import (
    cancel_campaign,
    create_campaign,
    pause_campaign,
    run_campaign,
)
from atomic_model_workspace.common import canonical
from atomic_model_workspace.jobs import cancel_job, load_job, pause_job, resume_job, run_job
from atomic_model_workspace.project import validate_project
from atomic_model_workspace.reports import (
    campaign_report,
    compare_campaigns,
    export_report,
    job_metrics,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate")
    p.add_argument("project", type=Path)
    p = sub.add_parser("initialize")
    p.add_argument("project", type=Path)
    p.add_argument("--output", required=True, type=Path)
    for command in ("run", "resume"):
        p = sub.add_parser(command)
        p.add_argument("campaign", type=Path)
        p.add_argument("--max-jobs", type=int)
        p.add_argument("--worker-timeout-s", type=float)
    p = sub.add_parser("cancel")
    p.add_argument("campaign", type=Path)
    p = sub.add_parser("pause")
    p.add_argument("campaign", type=Path)
    p = sub.add_parser("inspect")
    p.add_argument("campaign", type=Path)
    p = sub.add_parser("report")
    p.add_argument("campaign", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("compare")
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p = sub.add_parser("export")
    p.add_argument("project", type=Path)
    p.add_argument("--campaign", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("import")
    p.add_argument("archive", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("verify")
    p.add_argument("archive", type=Path)
    for command in ("job-run", "job-inspect", "job-cancel", "job-resume", "job-pause"):
        p = sub.add_parser(command)
        p.add_argument("job", type=Path)
        if command == "job-run":
            p.add_argument("--max-decisions", type=int)
            p.add_argument("--deadline-s", type=float)
    args = parser.parse_args(argv)
    try:
        c = args.command
        if c == "validate":
            result = validate_project(args.project)
        elif c == "initialize":
            result = create_campaign(args.project, args.output)
        elif c in ("run", "resume"):
            result = run_campaign(
                args.campaign,
                max_jobs=args.max_jobs,
                resume=c == "resume",
                worker_timeout_s=args.worker_timeout_s,
            )
        elif c == "cancel":
            result = cancel_campaign(args.campaign)
        elif c == "pause":
            result = pause_campaign(args.campaign)
        elif c == "inspect":
            result = campaign_report(args.campaign)
        elif c == "report":
            result = export_report(args.campaign, args.output)
        elif c == "compare":
            result = compare_campaigns(args.left, args.right)
        elif c == "export":
            result = export_bundle(args.project, args.output, campaign=args.campaign)
        elif c == "import":
            result = import_bundle(args.archive, args.output)
        elif c == "verify":
            result = verify_bundle(args.archive)["manifest"]
        elif c == "job-run":
            result = job_metrics(
                run_job(args.job, max_decisions=args.max_decisions, deadline_s=args.deadline_s)
            )
        elif c == "job-resume":
            result = job_metrics(resume_job(args.job))
        elif c == "job-pause":
            result = pause_job(args.job)
        elif c == "job-cancel":
            result = cancel_job(args.job)
        else:
            result = job_metrics(load_job(args.job))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(canonical(result))
    status = result.get("status")
    return 2 if status in ("rejected", "failed", "completed_with_failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
