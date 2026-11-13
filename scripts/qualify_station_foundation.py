"""Package A admission for declared index-1 R0 foundation, not NPP R1."""
from argparse import ArgumentParser
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station.model import StationConfig,signature
from atomic_model_station.mixed_session import MixedStationSession
from atomic_model_station.nonlinear import continuation
from atomic_model_station.foundation import passport,parameter_registry,validate_registry,reuse_audit,acceptance
from atomic_model_station.contracts import FluidState,fluid_connection,thermal_connection,mechanical_connection,electrical_connection

TASKS=[f'NPP-{b:02d}.{n:02d}' for b in range(3) for n in range(1,5)]


def reference_rhs(c,power,connected=True,load=1.):
    def f(t,x):
        tp,tw,ts,tc,k=x;q1=c.primary_ua_w_k*(tp-tw);q2=c.secondary_ua_w_k*(tw-ts)
        q=c.engine_ua_w_k*(ts-tc);work=c.engine_efficiency*(1-tc/ts)*q
        gen=(c.generator_drag_nm_s_rad*load if connected else 0.)*2*k/c.rotor_inertia_kg_m2
        bearing=c.bearing_drag_nm_s_rad*2*k/c.rotor_inertia_kg_m2;export=c.generator_efficiency*gen;cool=c.cooling_ua_w_k*(tc-c.ambient_temperature_k)
        return [(power-q1)/c.primary_capacity_j_k,(q1-q2)/c.wall_capacity_j_k,(q2-q)/c.secondary_capacity_j_k,
                (q-work+bearing+gen-export-cool)/c.cold_capacity_j_k,work-gen-bearing]
    return f


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    c=StationConfig();source=signature();driver=sha(__file__);registry=parameter_registry(c)
    audit=reuse_audit();levels=acceptance();validate_registry(registry)
    output.mkdir(parents=True,exist_ok=False)
    criteria=levels['levels']['R0']['targets']
    protocol=dict(source=source,driver_sha256=driver,passport=passport(),registry=registry,reuse_audit=audit,
                  acceptance=levels,closed_task_scope='Package A interfaces and minimal R0 assembly; no R1 equipment integration',
                  nonlinear_model='Five stocks, six algebraic fluxes, HeatExchangerWall adapter, Carnot-fraction ideal engine')
    atomic_json(output/'protocol.json',seal(protocol));print('protocol sealed',flush=True)
    roots=continuation(c);checks={task:True for task in TASKS}
    checks['NPP-00.01']=passport()['target_model_id']=='npp-pwr-research'
    checks['NPP-00.02']=all(r['source_sha256'] and r['states_and_units'] and r['applicability'] and r['status'] in ('ready','adapter','redesign') for r in audit)
    checks['NPP-00.03']=validate_registry(registry)
    checks['NPP-00.04']=set(levels['levels'])=={'R0','R1','R2','R3'}
    flows=[]
    for q in (2.,-2.,0.):
        a=FluidState(1e6,1e5,q,(('boron',.001),));b=FluidState(1e6,3e5,-q,(('boron',.002),))
        r=fluid_connection(a,b);flows.append(r)
    checks['NPP-01.01']=flows[0]['enthalpy_a_to_b_w']==2e5 and flows[1]['enthalpy_a_to_b_w']==-6e5 and flows[2]['enthalpy_a_to_b_w']==0.
    checks['NPP-01.02']=(thermal_connection(500.,10.,500.,-10.)['flow_residual_w']==0 and mechanical_connection(100.,2.,100.,-2.)['power_residual_w']==0 and electrical_connection(0.,False,0.,False)['power_residual_w']==0)
    s=MixedStationSession();manifest=s.system.manifest()
    checks['NPP-01.03']=len(manifest['variables'])==11 and len({v['owner'] for v in manifest['variables']})==2
    # Explicit rejected incompatible import and singular algebraic constraint.
    from atomic_model_station.mixed import Variable,ResidualComponent,MixedSystem
    rejected=[]
    for unit in ('W','K'):
        x=Variable('a','x','K',True,1.,-10.,10.);z=Variable('b','z','K',False,1.,-10.,10.)
        try:
            a=ResidualComponent('a',[x],{'b.z':unit},[1.],lambda t,y,d,u:[d['a.x']-y['b.z']])
            b=ResidualComponent('b',[z],{},[1.],lambda t,y,d,u:[0.])
            model=MixedSystem([a,b]);model.consistent(0.,np.array([1.,0.]),{})
        except ValueError:rejected.append(unit)
    checks['NPP-01.04']=len(rejected)==2
    checks['NPP-02.01']=s.initialization['rank']==11 and len(manifest['variables'])==11
    checks['NPP-02.02']=all(row['receipt']['rank']==11 and row['receipt']['residual']<criteria['initial_residual'] for row in roots)
    refinements=[]
    initial=s.y[:5].copy();reference=solve_ivp(reference_rhs(c,1.05e6),(0.,10.),initial,method='DOP853',rtol=1e-12,atol=1e-8)
    if not reference.success:raise ValueError('Independent ODE reference failed')
    scales=np.array([600.]*4+[6e7])
    for h in (.4,.2,.1):
        r=MixedStationSession(max_step_s=h);r.set_command({'source_power_w':1.05e6});r.advance(10.)
        error=float(np.max(abs(r.y[:5]-reference.y[:,-1])/scales));refinements.append(dict(max_step_s=h,error=error,ledger=r.ledger))
    if not (refinements[-1]['error']<criteria['reference_scaled_error'] and all(b['error']<=a['error'] for a,b in zip(refinements,refinements[1:]))):raise ValueError('Time refinement/reference failed')
    print('time refinement passed',flush=True)
    s=MixedStationSession();s.set_command({'source_power_w':1.1e6});threshold=float(s.y[0]+.01)
    s.guard('load-switch','stock.primary',threshold,1,{'generator_connected':False});s.advance(1.)
    root_events=[e for e in s.events if e['kind']=='root'];preserved=MixedStationSession();old=preserved.y[:5].copy();preserved.set_command({'generator_connected':False})
    trial=MixedStationSession();trial.schedule(.5,{'generator_connected':False});trial.history=[dict(op='observe',args={})]*256;before=trial.checkpoint()
    rollback=False
    try:trial.advance(1.)
    except ValueError:rollback=trial.checkpoint()==before
    checks['NPP-02.03']=bool(root_events) and s.revision==1 and np.array_equal(old,preserved.y[:5]) and rollback
    r=MixedStationSession(seed=73);m=r.observe();r.control(606.,m['wall_temperature_k'],.1)
    r.schedule(1.,{'generator_connected':False});r.guard('late','stock.primary',850.,1,{'source_power_w':0.});r.advance(.4)
    saved=json.loads(json.dumps(r.checkpoint()));restored=MixedStationSession.restore(saved)
    exact=restored.observe()==r.observe() and restored.advance(1.)==r.advance(1.)
    checks['NPP-02.04']=exact and not r.queue and r.revision==1
    atomic_json(output/'runtime.json',seal(dict(checkpoint=saved,post_restart=r.snapshot(),root_events=root_events,rollback_exact=rollback)))
    nominal=MixedStationSession(max_step_s=1.);initial=nominal.y.copy()
    for i in range(60):
        nominal.advance(60.)
        if i%10==0:print('nominal',i+1,'/60',flush=True)
    nominal_error=float(np.max(abs(nominal.y-initial)/nominal.system.scales))
    if nominal_error>1e-6:raise ValueError('Nominal drift')
    atomic_json(output/'nominal.json',seal(dict(checkpoint=nominal.checkpoint(),drift=nominal_error)))
    atomic_json(output/'continuation.json',seal(dict(rows=roots)))
    unchanged=source==signature() and driver==sha(__file__)
    passed=unchanged and all(checks.values())
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,package_a_complete=passed,
        task_checks=checks,closed_tasks=TASKS if passed else [],refinements=refinements,
        nominal_duration_s=nominal.time,nominal_scaled_drift=nominal_error,nominal_ledger=nominal.ledger,
        restart_exact=exact,root_events=root_events,rollback_exact=rollback,
        scope='All package-A tasks admitted for bounded index-1 R0 foundation; R1/R2/R3 not accepted',
        reactor_connected=False,steam_cycle_connected=False,validated=False,plant_referenced=False,
        limitations=['Implicit Euler first order; accepted finite-step program, not arbitrary stiffness guarantees',
                     'One crossing per guard per step; localization bracket tolerance is not physical event-time accuracy',
                     'Fluid connection contracts implemented; full fluid inventory/momentum/EOS station awaits package B',
                     'PI/sensor/RNG are foundation examples, not plant-control or protection qualification'])
    atomic_json(output/'report.json',seal(report));return report


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.output)
