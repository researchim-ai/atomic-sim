"""Print the offline candidate registry / Реестр кандидатов физических данных."""
import argparse
import json
from pathlib import Path
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from .model import candidate_registry


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=seal(candidate_registry())
    if args.output:
        out=args.output.absolute()
        if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):parser.error('New ordinary output required')
        out.parent.mkdir(parents=True,exist_ok=True);atomic_json(out,result)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
