"""Uninterrupted declared load cycle; incomplete or failed physics remains evidence."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_campaign.engine import signature as campaign_signature
from atomic_model_station_faults import FaultSession
from atomic_model_station_readiness import schema
from atomic_model_station_load import LoadCycle,assess,compare
from scripts.qualify_station_inventory import read


def identity():
    import atomic_model_station_load
    return dict(campaign=campaign_signature(),driver=sha(__file__),load={p.name:sha(p) for p in sorted(Path(atomic_model_station_load.__file__).parent.glob('*.py'))})


def run(output,cycle=None,*,max_step_s=1.,rtol=1e-7,reference=None,restart=False):
    cycle=cycle or LoadCycle();episode=cycle.episode(max_step_s=max_step_s,rtol=rtol)
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output refused')
    source=identity()
    if reference:
        previous=read(Path(reference)/'protocol.json')
        previous_cycle=LoadCycle(**previous['cycle'])
        if previous['source']!=source or previous_cycle!=cycle:raise ValueError('Load reference source or cycle differs')
        if restart and (previous['episode']['max_step_s']!=max_step_s or previous['episode']['rtol']!=rtol):
            raise ValueError('Native replay requires the original numerical settings')
    output.mkdir(parents=True,exist_ok=False)
    source=identity();protocol=dict(source=source,cycle=asdict(cycle),episode=episode.payload(),
        criteria=dict(power_relative=.02,pressure_relative=.02,speed_relative=.01,sg_level_m=.02,hotwell_level_m=.005,feed_level_m=.005,refinement_relative=.001),
        scope='All completed transitions; no state, time or PI reset; task acceptance requires full numerical and installed evidence')
    atomic_json(output/'protocol.json',seal(protocol))
    owner=FaultSession(tick_s=cycle.tick_s,history_budget_steps=max(256,cycle.ticks),
        equilibrium_start=True,inventory_deadband_m=.0002,max_step_s=max_step_s,rtol=rtol,
        tracking_tau_s=max(10.,cycle.tick_s))
    layout=schema(owner);commands={p['tick']:{k:v for k,v in p.items() if k!='tick'} for p in episode.commands}
    records=[];rejection=None;anchor=None
    atomic_json(output/'initial.json',seal(dict(checkpoint=owner.checkpoint(),layout=layout)))
    reference_rows=read(Path(reference)/'trajectory.json')['records'] if reference else None
    for tick in range(cycle.ticks):
        kwargs=dict(commands.get(tick,{}))
        if tick==0:kwargs['modes']={k:'auto' for k in owner.control.loops}
        try:result=owner.step(**kwargs)
        except ValueError as error:
            rejection=str(error);break
        snap=owner.control.plant.snapshot()
        records.append(dict(result=result,load_fraction=owner.control.plant.controls['load_fraction'],
            state=owner.control.plant.state.tolist(),physical={k:snap[k] for k in ('time_s','powers','energy_relative','mass_relative','boron_relative')}))
        if tick+1==cycle.ticks//2:
            anchor=owner.checkpoint();atomic_json(output/'midpoint.json',anchor)
            if restart:
                owner=FaultSession.restore(anchor)
                if owner.checkpoint()!=anchor:raise ValueError('Load native prefix differs')
        if (tick+1)%20==0:print('load',owner.control.plant.time,'/',cycle.duration_s,flush=True)
    assessment=assess(records,cycle)
    unchanged=identity()==source
    final=owner.checkpoint();atomic_json(output/'final.json',final)
    atomic_json(output/'trajectory.json',seal(dict(records=records)))
    comparison=compare(records,reference_rows,accounting_indices=layout['accounting_indices']) if reference_rows is not None else None
    exact=None
    if reference and restart:
        exact=final==json.loads((Path(reference)/'final.json').read_text()) and records==reference_rows
    report=dict(completed=len(records)==cycle.ticks,rejection=rejection,source_unchanged=unchanged,
        assessment=assessment,refinement=comparison,native_prefix_replayed=bool(restart and anchor),
        native_whole_trajectory_exact=exact,completed_time_s=owner.control.plant.time,
        passed=bool(assessment['passed'] and unchanged and (comparison is None or comparison['passed']) and exact is not False),
        closed_tasks=[],r1_accepted=False,validated=False,
        scope='Single declared trajectory; a passing run alone does not close NPP-11.02')
    atomic_json(output/'report.json',seal(report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-step-s',type=float,default=1.);p.add_argument('--rtol',type=float,default=1e-7)
    p.add_argument('--reference',type=Path);p.add_argument('--restart',action='store_true')
    a=p.parse_args();r=run(a.output,max_step_s=a.max_step_s,rtol=a.rtol,reference=a.reference,restart=a.restart)
    print(json.dumps(r,sort_keys=True));return 0 if r['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
