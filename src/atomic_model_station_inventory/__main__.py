"""Bounded local inventory experiment; no CI or external runner required."""
import argparse
import math
from pathlib import Path

from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from atomic_model_station_faults import FaultSession
from atomic_model_station_faults.model import signature
from .model import capture, summarize


def main():
    parser=argparse.ArgumentParser(description='Privileged station stock/continuity diagnostics')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration-s', type=float, default=60.)
    parser.add_argument('--tick-s', type=float, default=1.)
    parser.add_argument('--forecast-horizon-s', type=float, default=3600.)
    args=parser.parse_args()
    if not math.isfinite(args.duration_s) or not 0<args.duration_s<=7200 or not math.isfinite(args.tick_s) or not .01<=args.tick_s<=10:
        parser.error('Duration/tick outside station domain')
    steps=round(args.duration_s/args.tick_s)
    if steps<1 or steps>8192 or abs(steps*args.tick_s-args.duration_s)>1e-8:
        parser.error('Duration requires 1..8192 complete control ticks')
    if not math.isfinite(args.forecast_horizon_s) or not 0<args.forecast_horizon_s<=7200:
        parser.error('Forecast horizon outside 0..7200 seconds')
    output=args.output.absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('Symlink output refused')
    output.mkdir(parents=True,exist_ok=False)
    owner=FaultSession(tick_s=args.tick_s,history_budget_steps=8192)
    protocol=dict(source=signature(),duration_s=args.duration_s,tick_s=args.tick_s,
                  forecast_horizon_s=args.forecast_horizon_s,privileged=True,stationary_accepted=False)
    atomic_json(output/'protocol.json',seal(protocol))
    rows=[capture(owner)];failure=None
    try:
        for i in range(steps):
            owner.step(modes={k:'auto' for k in owner.control.loops} if i==0 else None)
            rows.append(capture(owner))
    except ValueError as error:
        failure=str(error)
    atomic_json(output/'trace.json',seal(dict(records=rows)))
    atomic_json(output/'checkpoint.json',owner.checkpoint())
    report=dict(completed=failure is None,failure=failure,completed_time_s=owner.control.plant.time,
                stationary_accepted=False,r1_accepted=False,source_unchanged=signature()==protocol['source'],
                summary=summarize(rows,requested_horizon_s=args.forecast_horizon_s,maximum_gap_s=args.tick_s) if len(rows)>1 else None)
    atomic_json(output/'report.json',seal(report))
    print('completed='+str(report['completed'])+'; time_s='+str(report['completed_time_s']))
    return 0 if report['completed'] and report['source_unchanged'] else 1


if __name__=='__main__':
    raise SystemExit(main())
