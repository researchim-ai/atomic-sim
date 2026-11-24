"""Compatibility entry point for local PWR cladding component delivery."""
import argparse
from pathlib import Path
from scripts.verify_component_delivery import run as _run,TESTS,COLD_TESTS


def run(output,timeout_s=600):
    return _run(output,timeout_s,component='local-cladding')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase-timeout-s',type=int,default=600)
    args=parser.parse_args();result=run(args.output,args.phase_timeout_s)
    print(dict(passed=result['passed'],source_tests=result['source_tests']['counts'],installed_tests=result['installed_tests']['counts']))
