import argparse
from pathlib import Path

from atomic_model_pool.runner import compare
from atomic_model_workspace.common import canonical, read


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare cold per-case EOS processes with bounded trusted worker reuse"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare(read(args.input), args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(
        canonical(
            {
                k: report[k]
                for k in (
                    "passed",
                    "numerical_identity_exact",
                    "two_x_vs_classic_same_workers",
                    "summaries",
                )
            }
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
