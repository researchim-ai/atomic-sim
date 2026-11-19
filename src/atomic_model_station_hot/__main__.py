"""Local preparation/hold with periodic accepted-prefix checkpoints."""
import argparse
from dataclasses import asdict
import json
import hashlib
import sys
from pathlib import Path

from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from .model import HotConfig, HotSession, signature
from atomic_model_station_readiness import ReadinessConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--maximum-warmup-s',type=float,default=3600.)
    parser.add_argument('--hold-s',type=float,default=3600.)
    parser.add_argument('--tick-s',type=float,default=5.)
    parser.add_argument('--inventory-deadband-m',type=float,default=.0002)
    parser.add_argument('--max-step-s',type=float,default=1.)
    parser.add_argument('--rtol',type=float,default=1e-7)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--equilibrium-start',action='store_true')
    parser.add_argument('--film-model',choices=['dittus_boelter','gnielinski'],default='dittus_boelter')
    args=parser.parse_args()
    parent_sha=None;saved=None
    if args.resume is not None:
        controls={'--maximum-warmup-s','--hold-s','--tick-s','--inventory-deadband-m','--max-step-s','--rtol','--equilibrium-start','--film-model'}
        if any(arg.partition('=')[0] in controls for arg in sys.argv[1:]):
            parser.error('Resume preserves the checkpoint configuration; omit configuration flags')
        raw=args.resume.read_bytes();parent_sha=hashlib.sha256(raw).hexdigest();saved=json.loads(raw)
        from atomic_model_multiphysics.provenance import unseal
        config=HotConfig.from_payload(unseal(saved)['config'])
    else:
        config=HotConfig(maximum_warmup_s=args.maximum_warmup_s,hold_s=args.hold_s,tick_s=args.tick_s,
            inventory_deadband_m=args.inventory_deadband_m,max_step_s=args.max_step_s,rtol=args.rtol,
            equilibrium_start=args.equilibrium_start,film_model=args.film_model,
            readiness=ReadinessConfig(maximum_gap_s=args.tick_s))
    out=args.output.absolute()
    if any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Symlink output refused')
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'protocol.json',seal(dict(config=asdict(config),source=signature(),parent_checkpoint_sha256=parent_sha,stationary_accepted=False)))
    if saved is not None:print('Verifying accepted prefix before resume',flush=True)
    session=HotSession(config) if saved is None else HotSession.restore(saved)
    atomic_json(out/'layout.json',seal(session.layout))
    while not session.finished:
        report=session.step()
        if session.attempts%20==0 or session.finished:
            atomic_json(out/('checkpoint-'+str(session.attempts).zfill(5)+'.json'),session.checkpoint())
            print(json.dumps(dict(time_s=report['time_s'],phase=report['phase'],
                readiness_reason=report['readiness']['reason'],
                maximum_span=report['readiness'].get('maximum_normalized_span')),sort_keys=True),flush=True)
    atomic_json(out/'checkpoint.json',session.checkpoint())
    atomic_json(out/'report.json',seal(dict(**session.report(),inventory=session.inventory_summary())))
    print(json.dumps(session.report(),sort_keys=True),flush=True)
    return 0 if session.phase=='complete' else 2


if __name__=='__main__':raise SystemExit(main())
