import argparse
from pathlib import Path

from atomic_model_reference.bundle import check_catalog, import_reference, load_bundle
from atomic_model_reference.compare import compare_job
from atomic_model_reference.replay import replay_report
from atomic_model_workspace.common import canonical


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Strict reference data, independent metrics and physical-case split guards"
    )
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("import")
    a.add_argument("input", type=Path)
    a.add_argument("--output", required=True, type=Path)
    a = sub.add_parser("inspect")
    a.add_argument("bundle", type=Path)
    a = sub.add_parser("catalog")
    a.add_argument("bundles", nargs="+", type=Path)
    a = sub.add_parser("replay")
    a.add_argument("report", type=Path)
    a.add_argument("--output", required=True, type=Path)
    a = sub.add_parser("compare")
    a.add_argument("job", type=Path)
    a.add_argument("reference", type=Path)
    a.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    try:
        if a.command == "import":
            r = import_reference(a.input, a.output)
            result = {
                k: r[k]
                for k in (
                    "reference_id",
                    "physical_case_id",
                    "kind",
                    "role",
                    "physical_conditions_sha256",
                )
            }
        elif a.command == "inspect":
            r, _ = load_bundle(a.bundle)
            result = {k: v for k, v in r.items() if k not in ("times_s", "values", "uncertainty")}
            result["samples"] = len(r["times_s"])
        elif a.command == "catalog":
            result = check_catalog(a.bundles)
        elif a.command == "replay":
            result = replay_report(a.report, a.output)
        else:
            result = compare_job(a.job, a.reference, a.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.error(str(exc))
    print(canonical(result))
    return 2 if result.get("passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
