"""Record a complete shutdown or a fail-closed partial horizon with exact rollback."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
from copy import deepcopy
from core.status import DomainError,SolverError,ModelLimitError,classify_model_error
from atomic_model_station_faults import FaultSession
from atomic_model_station_shutdown import ShutdownCase,capture,assess
from atomic_model_station_readiness import schema
from atomic_model_station_campaign.engine import signature
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_inventory import read


def identity():
    import atomic_model_station_shutdown
    return dict(station=signature(),driver_sha256=sha(__file__),shutdown={p.name:sha(p) for p in sorted(Path(atomic_model_station_shutdown.__file__).parent.glob('*.py'))})


def run(output,case=None,*,max_step_s=1.,rtol=1e-7,restart=False,reference=None):
    case=case or ShutdownCase();out=Path(output).absolute();source=identity()
    if any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Symlink output refused')
    if reference:
        protocol=read(Path(reference)/'protocol.json')
        if protocol['source']!=source or ShutdownCase(**protocol['case'])!=case:raise ValueError('Shutdown reference identity')
        if restart and (protocol['max_step_s']!=max_step_s or protocol['rtol']!=rtol):raise ValueError('Native numerical settings differ')
    owner=FaultSession(tick_s=case.tick_s,history_budget_steps=max(256,case.ticks),equilibrium_start=True,
        inventory_deadband_m=.0002,max_step_s=max_step_s,rtol=rtol,tracking_tau_s=max(10.,case.tick_s),film_model=case.film_model)
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'protocol.json',seal(dict(source=source,case=asdict(case),max_step_s=max_step_s,rtol=rtol,
        criteria=dict(energy_relative=1e-6,mass_relative=1e-10,boron_relative=1e-10,fission_fraction=1e-3,primary_momentum_pa=1e-5,dhr_transfer_w=1e-5,dhr_entropy_min_w_k=-1e-6),closed_tasks=[])))
    initial=capture(owner);atomic_json(out/'initial.json',seal(dict(record=initial,checkpoint=owner.checkpoint(),layout=schema(owner))))
    records=[];failure=None;rollback=None;native=False
    for tick in range(case.ticks):
        before=None
        # Structural clone keeps rejected-step proof without serializing the whole prefix each tick.
        before=deepcopy(owner)
        try:result=owner.step(operator=case.command() if tick==0 else None,modes={k:'auto' for k in owner.control.loops} if tick==0 else None)
        except (ValueError,DomainError,SolverError) as error:
            failure=dict(kind=type(error).__name__,category=classify_model_error(error),message=str(error),attempted_start_s=owner.control.plant.time)
            if isinstance(error,ModelLimitError):failure['details']=error.details()
            if hasattr(error,'details') and isinstance(error,SolverError):failure['details']=error.details()
            rollback=owner.checkpoint()==before.checkpoint();break
        records.append(capture(owner,result))
        if tick+1==case.ticks//2:
            saved=owner.checkpoint();atomic_json(out/'midpoint.json',saved)
            if restart:
                owner=FaultSession.restore(saved);native=owner.checkpoint()==saved
                if not native:raise ValueError('Shutdown native prefix differs')
        if (tick+1)%4==0:print(case.family,owner.control.plant.time,'/',case.duration_s,flush=True)
    verdict=assess(initial,records,case);final=owner.checkpoint()
    atomic_json(out/'trajectory.json',seal(dict(records=records)));atomic_json(out/'final.json',final)
    exact=None
    if restart and reference:
        exact=(read(Path(reference)/'trajectory.json')['records']==records and json.loads((Path(reference)/'final.json').read_text())==final)
    report=dict(passed=bool(verdict['passed'] and source==identity() and failure is None and exact is not False),
        completed=len(records)==case.ticks,completed_time_s=owner.control.plant.time,source_unchanged=source==identity(),assessment=verdict,
        failure=failure,rollback_exact=rollback,native_prefix_exact=native,native_whole_path_exact=exact,
        minimum_accepted_reynolds=min(r['physical']['film']['reynolds'] for r in [initial,*records]),
        exact_zero_demand_steps=sum(any(m['kind']=='exact_closed_loop_exponential' for m in r['physical']['exact_mode_reduction']) for r in records),
        closed_tasks=[],npp_11_03_accepted=False,r1_accepted=False,validated=False)
    atomic_json(out/'report.json',seal(report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--family',choices=['scram','coastdown'],default='scram')
    p.add_argument('--duration-s',type=float);p.add_argument('--max-step-s',type=float,default=1.);p.add_argument('--rtol',type=float,default=1e-7)
    p.add_argument('--film-model',choices=['dittus_boelter','gnielinski'],default='dittus_boelter')
    p.add_argument('--restart',action='store_true');p.add_argument('--reference',type=Path)
    a=p.parse_args();r=run(a.output,ShutdownCase(a.family,duration_s=a.duration_s,film_model=a.film_model),max_step_s=a.max_step_s,rtol=a.rtol,restart=a.restart,reference=a.reference)
    print(json.dumps({k:r[k] for k in ('passed','completed','completed_time_s','failure')}));return 0 if r['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
