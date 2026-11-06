"""Prepare portable requests and compare replay without trusting host claims."""

import argparse
import json

from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json, read_json

from .candidates import admit, motel_candidate
from .replay import compare, make_request, verify_packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("spec")
    prepare.add_argument("directory")
    check = sub.add_parser("compare")
    check.add_argument("request")
    check.add_argument("receipt")
    check.add_argument("--output", required=True)
    packet = sub.add_parser("verify-packet")
    packet.add_argument("directory")
    candidates = sub.add_parser("candidates")
    candidates.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = make_request(read_json(args.spec), args.directory)
    elif args.command == "compare":
        result = compare(read_json(args.request), read_json(args.receipt))
        atomic_json(args.output, seal(result))
    elif args.command == "verify-packet":
        result = verify_packet(args.directory)
    else:
        candidate = motel_candidate()
        result = dict(candidate=candidate, admission=admit(candidate))
        atomic_json(args.output, seal(result))
    print(json.dumps(dict(status="completed", execution_scope="local", validated=False)))
    return 2 if args.command == "compare" and not result["numerical_replay_accepted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
