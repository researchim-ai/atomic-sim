import argparse
from pathlib import Path
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from .model import prepare_hot_state


def main():
    p=argparse.ArgumentParser(description='Prepare a hot station without resetting time or inventories')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--maximum-warmup-s',type=float,default=7200.)
    p.add_argument('--required-followup-s',type=float,default=0.)
    p.add_argument('--inventory-deadband-m',type=float,default=0.);args=p.parse_args()
    out=args.output.absolute()
    if any(q.is_symlink() for q in (out,*out.parents)):raise ValueError('Symlink output refused')
    out.mkdir(parents=True,exist_ok=False)
    result=prepare_hot_state(maximum_warmup_s=args.maximum_warmup_s,required_followup_s=args.required_followup_s,
        inventory_deadband_m=args.inventory_deadband_m)
    atomic_json(out/'report.json',seal(result))
    print('prepared='+str(result['prepared'])+'; time_s='+str(result['completed_time_s']))
    return 0 if result['prepared'] else 1


if __name__=='__main__':raise SystemExit(main())
