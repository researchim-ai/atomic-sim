"""Run/verify the independent limited HEM experimental evidence audit."""

import argparse

from .audit import run, verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "verify"])
    parser.add_argument("output")
    args = parser.parse_args(argv)
    try:
        result = (run if args.action == "run" else verify)(args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Audit rejected: {exc}")
        return 2
    print(f"samples={result['samples']} passed={result['passed']} "
          f"production_max_error={result['comparisons']['production_colebrook']['max_abs_relative_error']:.8%} "
          f"haaland_max_error={result['comparisons']['independent_haaland']['max_abs_relative_error']:.8%}; HEM validated=False")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
