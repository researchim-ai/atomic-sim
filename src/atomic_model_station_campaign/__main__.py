"""Run a sealed, versioned episode manifest entirely on the local computer."""
import argparse
import json
from pathlib import Path
from atomic_model_multiphysics.provenance import unseal
from .spec import Episode,Limits
from .runner import run_campaign


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    p=unseal(json.loads(args.manifest.read_text()))
    if set(p)!={'format','episodes','limits'} or p['format']!='station-campaign/v1':raise ValueError('Campaign manifest format')
    report=run_campaign(args.output,[Episode.from_payload(e) for e in p['episodes']],limits=Limits(**p['limits']))
    print(json.dumps(report,sort_keys=True))
    return 0 if report['accepted'] else 1


if __name__=='__main__':raise SystemExit(main())
