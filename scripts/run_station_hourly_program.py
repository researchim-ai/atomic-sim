"""Generate a local coarse reference, then refine/restart and audit it.

No private previous artifacts are required. Long computations are intentional;
none of the requested horizons are shortened when the state is not stationary.
"""
from argparse import ArgumentParser
from pathlib import Path
import json

from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_faults import FaultSession
from atomic_model_station_campaign.engine import signature
from scripts import qualify_station_nominal as nominal
from scripts import qualify_station_hourly_replay as replay


def generate_reference(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output refused')
    output.mkdir(parents=True,exist_ok=False)
    source=signature();driver=sha(__file__)
    protocol=dict(source=source,driver_sha256=driver,criteria=nominal.CRITERIA,
        criteria_worker_sha256=sha(nominal.__file__),duration_s=3600.,tick_s=5.,
        max_step_s=1.,rtol=1e-7,tracking_tau_s=10.,history_budget_steps=8192)
    atomic_json(output/'generation_protocol.json',seal(protocol))
    owner=FaultSession(tick_s=5.,history_budget_steps=8192);rows=[];failure=None
    for index in range(720):
        try:result=owner.step(modes={k:'auto' for k in owner.control.loops} if index==0 else None)
        except ValueError as error:failure=str(error);break
        record,_=nominal.row(owner,result);rows.append(record)
        if (index+1)%20==0:print('coarse',owner.control.plant.time,'/3600',flush=True)
        if (index+1)%120==0:
            atomic_json(output/(str(index+1)+'.json'),seal(dict(source=source,trace=rows,checkpoint=owner.checkpoint())))
            rows=[]
    complete=owner.control.plant.time==3600. and failure is None
    if not complete:
        atomic_json(output/'failure.json',seal(dict(completed=False,reason=failure,completed_time_s=owner.control.plant.time,
            accepted_partial_rows=rows,checkpoint=owner.checkpoint(),formal_acceptance=False)))
        raise ValueError('Coarse hour did not complete; partial evidence saved')
    files=[output/(str(n)+'.json') for n in nominal.BLOCKS]
    baseline=dict(completed=True,completed_duration_s=3600.,source=source,formal_acceptance=False,
        logical_input_keys=True,logical_input_prefix='artifacts/station-nominal-exploration-v1/',
        actual_generation_directory=str(output),generation_protocol=protocol,
        generation_protocol_sha256=sha(output/'generation_protocol.json'),
        source_unchanged=source==signature() and driver==sha(__file__),
        evidence_sha256={'artifacts/station-nominal-exploration-v1/'+p.name:sha(p) for p in files})
    if not baseline['source_unchanged']:raise ValueError('Source changed during coarse generation')
    atomic_json(output/'baseline.json',seal(baseline));return output/'baseline.json'


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.absolute()
    if any(q.is_symlink() for q in (out,*out.parents)):raise ValueError('Symlink output refused')
    out.mkdir(parents=True,exist_ok=False)
    baseline=generate_reference(out/'coarse')
    try:nominal.run(out/'candidate',coarse_evidence=out/'coarse',baseline=baseline)
    except ValueError as error:
        if str(error)!='Full-hour nominal qualification failed':raise
        print('Whole-station admission failed; preserving negative result and checking numerical replay separately',flush=True)
    report=replay.run(out/'numerical',candidate=out/'candidate')
    print(json.dumps(report,sort_keys=True))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
