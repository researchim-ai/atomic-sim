"""Measure fixed EOS workload scaling, native CPU profile and Linux RSS."""

import argparse
from pathlib import Path

from atomic_model_bench.harness import benchmark
from atomic_model_workspace.common import canonical, read


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-profile", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = benchmark(read(args.input), args.output, profile=not args.no_profile)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(
        canonical(
            {
                k: result[k]
                for k in ("passed", "numerical_identity_exact", "two_x_goal_met", "summaries")
            }
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
