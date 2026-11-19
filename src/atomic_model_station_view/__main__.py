"""Export a short native demonstration; files contain privileged physical data."""
import argparse
from pathlib import Path
from atomic_model_station_env import EnvConfig, StationEnv
from atomic_model_station_faults import Fault
from .model import capture, document, write


def demo():
    env=StationEnv(EnvConfig(max_steps=4,tick_s=.05,automatic=False,
        faults=(Fault('pressure_transport_failure','failure','pressure',1,3),)))
    env.reset();frames=[capture(env)]
    for action in ({'operator':{'feed_pump_on':False}},
                   {'operator':{'feed_pump_on':True}},
                   {'operator':{'dhr_requested':True}},
                   {'operator':{'scram':True}}):
        env.step(action);frames.append(capture(env))
    return document(frames)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();write(args.output,demo())
    print('Exported five native research frames; privileged data, not agent input.')
