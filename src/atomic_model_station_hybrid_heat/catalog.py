"""Explicit pressure-cycle cases with full-denominator batch reporting."""
from copy import deepcopy
from dataclasses import dataclass,asdict
import re
from atomic_model_station import PWRBypassSession,BypassConfig
from .model import advance_hybrid_schedule,identity
from .scenarios import prepare_relief_cycle


@dataclass(frozen=True)
class CycleCase:
    name:str
    area_m2:float=.03
    stroke_time_s:float=.1

    def __post_init__(self):
        if type(self.name) is not str or re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',self.name) is None:
            raise ValueError('Safe nonempty case identifier required')
        BypassConfig(area_m2=self.area_m2,stroke_time_s=self.stroke_time_s,
            relief_open_pa=6368300.,relief_close_pa=6368290.)


def run_cycle_case(case,*,subdivisions=(8,16,32)):
    if type(case) is not CycleCase:raise ValueError('CycleCase required')
    source=identity();proof=None;checks={};failure=None;recipe=None
    try:
        _,schedule,reference=prepare_relief_cycle()
        config=BypassConfig(**dict(reference['bypass'],area_m2=case.area_m2,stroke_time_s=case.stroke_time_s))
        session=PWRBypassSession(bypass=config,rtol=reference['rtol'],max_step_s=reference['max_step_s'])
        before=session.checkpoint();native=deepcopy(session)
        recipe=dict(reference,bypass=asdict(config),case=asdict(case))
        for row in schedule:native.advance(row['duration_s'],row['patch'])
        proof=advance_hybrid_schedule(session,schedule,subdivisions=subdivisions)
        events=[e for w in proof['windows'] for packet in w['replays'][-1]['events'] for e in packet['journal_events']]
        pressure=[e for e in events if e['kind'] in ('bypass_relief_open','bypass_relief_close')]
        checks=dict(audit=proof['passed'],native_equivalence=session.checkpoint()==native.checkpoint(),
            pressure_order=[e['kind'] for e in pressure]==reference['expected_pressure_events'],
            pressure_residuals=all(abs(e['pressure_pa']-getattr(config,'relief_open_pa' if e['kind']=='bypass_relief_open' else 'relief_close_pa'))<=.05 for e in pressure),
            final_return=not session.relief_latched and float(session.state[session.bypass_index])==0.,
            return_event=any(e['kind']=='bypass_position_arrival' and e.get('target')==0. for e in events),
            steam_arrivals=sum(e['kind']=='steam_valve_arrived' for e in events)==2,
            complete_clock=abs(session.time-reference['duration_s'])<=1e-12)
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error),receipt=getattr(error,'receipt',None))
        if 'session' in locals() and 'before' in locals():failure['session_unchanged']=session.checkpoint()==before
    if source!=identity():raise ValueError('Cycle catalog source changed')
    return dict(source=source,case=asdict(case),recipe=recipe,proof=proof,checks=checks,failure=failure,
        passed=failure is None and bool(checks) and all(checks.values()),
        physical_range_qualified=False,industrial_qualification=False,closed_tasks=[])


def run_cycle_catalog(cases,*,subdivisions=(8,16,32)):
    if type(cases) not in (list,tuple) or not 1<=len(cases)<=8 or any(type(c) is not CycleCase for c in cases):
        raise ValueError('One to eight explicit CycleCase objects required')
    names=[c.name for c in cases]
    if len(set(names))!=len(names):raise ValueError('Duplicate cycle case identifier')
    variants=[(c.area_m2,c.stroke_time_s) for c in cases]
    if len(set(variants))!=len(variants):raise ValueError('Duplicate physical case under different identifiers')
    source=identity();rows=[]
    for case in cases:
        rows.append(run_cycle_case(case,subdivisions=subdivisions))
        if source!=identity():raise ValueError('Cycle catalog source changed')
    passed=sum(r['passed'] for r in rows)
    return dict(source=source,cases=[asdict(c) for c in cases],results=rows,total_cases=len(cases),
        passed_cases=passed,failed_cases=len(cases)-passed,success_fraction=passed/len(cases),passed=passed==len(cases),
        all_attempts_retained=True,physical_range_qualified=False,industrial_qualification=False,closed_tasks=[])
