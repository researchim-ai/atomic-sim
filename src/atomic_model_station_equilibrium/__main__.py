"""Construct a declared steady time-zero state; never rewrites an operating run."""
import argparse
from pathlib import Path
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from atomic_model_station.faults import PWRFaultSession
from .model import EquilibriumConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--power-w',type=float,default=3.15e9)
    parser.add_argument('--pressure-pa',type=float,default=6.3e6)
    parser.add_argument('--speed-rad-s',type=float,default=157.)
    args=parser.parse_args();config=EquilibriumConfig(args.power_w,args.pressure_pa,args.speed_rad_s)
    out=args.output.absolute()
    if any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Symlink output refused')
    out.mkdir(parents=True,exist_ok=False)
    owner=PWRFaultSession(equilibrium=config)
    atomic_json(out/'checkpoint.json',owner.checkpoint())
    atomic_json(out/'report.json',seal(owner.station_equilibrium['report']))
    print('Steady initialization solved; rank='+str(owner.station_equilibrium['report']['constrained_rank']))


if __name__=='__main__':main()
