"""Research station instrumentation: causal clocks, lag, noise and paired restart."""
from argparse import ArgumentParser
from decimal import Decimal,localcontext
import json
from pathlib import Path

import numpy as np
from atomic_model_station import PWRBypassSession
from atomic_model_station.bypass import bypass_signature
from atomic_model_station_signals import Sensor,StationSignals,signature
from scripts import qualify_station_actuators as evidence
from atomic_model_multiphysics.provenance import sha

TASKS=['NPP-09.01']
CRITERIA=dict(lag_scaled_absolute=1e-12,noise_mean_absolute=.15,noise_std_min=1.85,
    noise_std_max=2.15,station_energy_relative=1e-6,station_mass_relative=1e-10)


def signals_signature():return dict(instruments=signature(),station=bypass_signature())


def patch(tick):
    return dict(steam_admission=.99) if tick==6 else dict(steam_admission=1.) if tick==16 else None


def run(output,reference=None):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output refused')
    output.mkdir(parents=True,exist_ok=False)
    source=signals_signature();driver=sha(__file__);serializer=sha(evidence.__file__)
    specs=[Sensor('power','fission_power_w',delay_ticks=1,lag_s=.5,sigma=2e5),
           Sensor('pressure','steam_pressure_pa',period_ticks=2,delay_ticks=3,lag_s=2.,sigma=100.),
           Sensor('flow','steam_flow_kg_s',delay_ticks=1,lag_s=1.,sigma=.2),
           Sensor('speed','turbine_speed_rad_s',lag_s=.2,sigma=.01),
           Sensor('hotwell','hotwell_mass_kg',period_ticks=5,delay_ticks=2,lag_s=5.,sigma=1.)]
    bank=StationSignals(specs,seed=42);plant=PWRBypassSession(max_step_s=.5)
    evidence.save(output/'protocol.json',dict(source=source,driver_sha256=driver,
        serializer_sha256=serializer,criteria=CRITERIA,candidate_tasks=TASKS,
        duration_s=30.,checkpoint_at_s=10.,tick_s=1.,seed=42,
        initial_instruments=bank.checkpoint(),initial_station=plant.checkpoint(),
        commands=[dict(time_s=5.,steam_admission=.99),dict(time_s=15.,steam_admission=1.)],
        scope='Passive hot-state research instruments; linear-between-truth-ticks input, not hardware calibration',
        agent_contract='Only operator_frame is public; truth, analog state, pending queue and RNG checkpoints are privileged'))
    lag_cases=[]
    for tau in (.01,.1,1.,1000.):
        owner=StationSignals([Sensor('ramp','fission_power_w',lag_s=tau)],tick_s=.1)
        errors=[]
        with localcontext() as context:
            context.prec=60
            for tick in range(101):
                time=Decimal(tick)/10;constant=Decimal(str(tau))
                expected=float(time-constant+constant*(-time/constant).exp())
                value=owner.ingest(tick,{'fission_power_w':float(time)})['measurements']['ramp']['value']
                errors.append(abs(value-expected)/max(abs(expected),1.))
        lag_cases.append(dict(lag_s=tau,scaled_absolute_error=max(errors)))
    delayed=StationSignals([Sensor('p','fission_power_w',period_ticks=3,delay_ticks=5)])
    delays=[]
    for tick in range(31):
        value=delayed.ingest(tick,{'fission_power_w':float(tick)})['measurements']['p']
        expected=None if tick<5 else float(((tick-5)//3)*3)
        delays.append(dict(tick=tick,reading=value,expected_value=expected,passed=value['value']==expected))
    noise=StationSignals([Sensor('n','fission_power_w',sigma=2.)],seed=17)
    draws=np.array([noise.ingest(t,{'fission_power_w':0.})['measurements']['n']['value'] for t in range(2000)])
    noise_reference=dict(count=len(draws),mean=float(draws.mean()),std=float(draws.std()),minimum=float(draws.min()),maximum=float(draws.max()))
    evidence.save(output/'independent.json',dict(linear_ramp=lag_cases,delays=delays,noise=noise_reference))
    snapshots=[plant.snapshot()];frames=[bank.ingest_station(snapshots[-1])];passive=True
    saved_plant=saved_bank=None
    for tick in range(1,31):
        snap=plant.advance(1.,patch(tick));state=plant.state.copy()
        frames.append(bank.ingest_station(snap));snapshots.append(snap)
        passive=passive and np.array_equal(state,plant.state)
        if tick==10:
            saved_plant=json.loads(json.dumps(plant.checkpoint()));saved_bank=json.loads(json.dumps(bank.checkpoint()))
        if tick%10==0:print('station instruments',tick,'/30',flush=True)
    evidence.save(output/'station.json',dict(operator_frames=frames,trusted_station_trace=snapshots,
        paired_checkpoint=dict(station=saved_plant,instruments=saved_bank),
        final_station=plant.checkpoint(),final_instruments=bank.checkpoint(),passive=bool(passive)))
    restored_plant=PWRBypassSession.restore(saved_plant);restored_bank=StationSignals.restore(saved_bank)
    exact=True
    for tick in range(11,31):
        snap=restored_plant.advance(1.,patch(tick))
        exact=exact and restored_bank.ingest_station(snap)==frames[tick] and snap==snapshots[tick]
    exact=exact and restored_plant.checkpoint()==plant.checkpoint() and restored_bank.checkpoint()==bank.checkpoint()
    evidence.save(output/'restart.json',dict(exact=bool(exact),final_station=restored_plant.checkpoint(),final_instruments=restored_bank.checkpoint()))
    before=bank.checkpoint();rejected=False
    values={s.quantity:0. for s in specs};values['steam_pressure_pa']=float('nan')
    try:bank.ingest(31,values)
    except ValueError:rejected=bank.checkpoint()==before
    evidence.save(output/'rollback.json',dict(invalid_truth_rejected_atomically=rejected))
    allowed={'value','unit','sampled_at_s','available_at_s','age_s','quality'}
    private_free=all(set(f)=={'time_s','measurements'} and all(set(r)==allowed for r in f['measurements'].values()) for f in frames)
    causal=all(r['available_at_s'] is None or r['available_at_s']<=f['time_s'] for f in frames for r in f['measurements'].values())
    unchanged=source==signals_signature() and driver==sha(__file__) and serializer==sha(evidence.__file__)
    passed=bool(unchanged and exact and rejected and passive and private_free and causal
        and max(c['scaled_absolute_error'] for c in lag_cases)<=CRITERIA['lag_scaled_absolute']
        and all(c['passed'] for c in delays)
        and abs(noise_reference['mean'])<CRITERIA['noise_mean_absolute']
        and CRITERIA['noise_std_min']<noise_reference['std']<CRITERIA['noise_std_max']
        and max(s['energy_relative'] for s in snapshots)<=CRITERIA['station_energy_relative']
        and max(s['mass_relative'] for s in snapshots)<=CRITERIA['station_mass_relative'])
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,closed_tasks=TASKS if passed else [],
        restart_exact=bool(exact),passive=bool(passive),operator_payload_private_free=private_free,causal_delivery=causal,
        invalid_truth_rejected_atomically=rejected,station_duration_s=30.,
        maximum_lag_scaled_absolute_error=max(c['scaled_absolute_error'] for c in lag_cases),noise=noise_reference,
        max_station_energy_relative=max(s['energy_relative'] for s in snapshots),
        max_station_mass_relative=max(s['mass_relative'] for s in snapshots),
        complete_package_c=False,r1_accepted=False,validated=False,
        limitations=['Research instrument parameters and hot-state initialization, not calibrated plant instruments',
            'Consecutive truth frames required; exact lag response applies to linear input between supplied ticks',
            'Operator payload separation is a data contract, not an arbitrary-Python security sandbox',
            'Passive observer only; no controllers, protection arbitration, sensor fault catalog or full R1 duration acceptance'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        reference=Path(reference);files={p.relative_to(output) for p in output.rglob('*.json')}
        if files!={p.relative_to(reference) for p in reference.rglob('*.json')}:raise ValueError('Instrument evidence sets differ')
        for name in sorted(files):
            if evidence.read(output/name)!=evidence.read(reference/name):raise ValueError('Installed instrument evidence differs:'+str(name))
    if not passed:raise ValueError('Station instrumentation qualification failed')
    print(json.dumps(dict(passed=True,closed_tasks=TASKS)),flush=True)
    return report


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reference',type=Path)
    args=parser.parse_args();run(args.output,args.reference)
