"""Do not mistake sampled thermal quadrature or transient energy ratios for exact heat counters."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import time
from atomic_model_station_env import EnvConfig,StationEnv,ActionError
from atomic_model_station_energy_eval import read_energy,assess_energy
from atomic_model_station_energy_eval.model import identity as energy_identity

FLUXES=('nuclear_heat_w','primary_pump_loss_out_w','external_cooling_heat_out_w','external_net_w')


def identity():
    return dict(energy=energy_identity(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob('*.py'))})


def finite(x):return type(x) in (int,float) and math.isfinite(x)


def read_heat(env):
    """Privileged read-only evaluator; no additional observations are sent to agents."""
    energy=read_energy(env)
    plant=deepcopy(env._session.control.plant);snapshot=plant.snapshot()
    return dict(energy=energy,stored_j=float(plant._stored()),boundary_j=float(plant.state[plant.ledger_index]),
        fluxes={key:float(snapshot['powers'][key]) for key in FLUXES})


def assess_heat(samples):
    if type(samples) not in (list,tuple) or not samples:raise ValueError('Nonempty heat samples required')
    for s in samples:
        if type(s) is not dict or set(s)!={'energy','stored_j','boundary_j','fluxes'}:
            raise ValueError('Complete heat sample required')
        if not finite(s['stored_j']) or not finite(s['boundary_j']):raise ValueError('Finite storage and boundary ledger required')
        if type(s['fluxes']) is not dict or set(s['fluxes'])!=set(FLUXES) or any(not finite(v) for v in s['fluxes'].values()):
            raise ValueError('Complete finite heat fluxes required')
    electric=assess_energy([s['energy'] for s in samples]);reasons=set(electric['reasons'])
    integrals={key:0. for key in FLUXES};maximum_instant=0.
    for i,s in enumerate(samples):
        f=s['fluxes'];export=s['energy']['powers']['net_export_w']
        if any(f[k]<0 for k in FLUXES if k!='external_net_w'):reasons.add('negative_nonreversing_heat_flux')
        terms=[f['nuclear_heat_w'],-f['primary_pump_loss_out_w'],-f['external_cooling_heat_out_w'],-export]
        residual=f['external_net_w']-math.fsum(terms)
        maximum_instant=max(maximum_instant,abs(residual))
        if abs(residual)>1e-5+1e-10*max(1.,sum(abs(x) for x in terms)):
            reasons.add('instantaneous_boundary_identity')
        if i:
            before=samples[i-1];dt=s['energy']['time_s']-before['energy']['time_s']
            for key in FLUXES:integrals[key]+=dt*(before['fluxes'][key]+f[key])/2
    first,last=samples[0],samples[-1]
    stored=last['stored_j']-first['stored_j'];boundary=last['boundary_j']-first['boundary_j']
    export=electric['energy_delta_j']['net_exported_j'];heat=integrals['nuclear_heat_w']
    losses=integrals['primary_pump_loss_out_w']+integrals['external_cooling_heat_out_w']
    balance_heat=boundary+losses+export
    scale=max(1.,abs(heat)+abs(losses)+abs(export))
    closure=heat-balance_heat;storage_defect=stored-boundary
    boundary_quad=integrals['external_net_w']-boundary
    if abs(closure)/scale>1e-6:reasons.add('sampled_heat_closure')
    if abs(storage_defect)/scale>1e-6:reasons.add('window_storage_balance')
    if abs(boundary_quad)/scale>1e-6:reasons.add('boundary_quadrature')
    if electric['duration_s']<=0:reasons.add('empty_window')
    power_values=[s['fluxes']['nuclear_heat_w'] for s in samples]
    variation=(max(power_values)-min(power_values))/max(max(abs(v) for v in power_values),1.)
    positive=heat>0
    ratios_allowed=positive and not reasons
    return dict(passed=not reasons,reasons=sorted(reasons),electrical=electric,
        sampled_integrals_j=integrals,nuclear_heat_from_boundary_j=balance_heat,
        stored_energy_change_j=stored,integrated_boundary_change_j=boundary,
        heat_closure_j=closure,heat_closure_relative=abs(closure)/scale,
        window_storage_defect_j=storage_defect,boundary_quadrature_defect_j=boundary_quad,
        maximum_instantaneous_boundary_defect_w=maximum_instant,
        finite_window_net_electric_to_nuclear_heat_ratio=export/heat if ratios_allowed else None,
        finite_window_gross_electric_to_nuclear_heat_ratio=electric['energy_delta_j']['gross_generated_j']/heat if ratios_allowed else None,
        storage_change_to_nuclear_heat=stored/heat if positive else None,
        sampled_nuclear_power_variation=variation,
        quadrature='Endpoint trapezoids;estimate,not backend-integrated heat counter',
        independent_refinement_required=True,stationary_efficiency_certified=False)


def run_heat_window(config):
    """Unforced manual-control reference; changing tick also changes solver restart cadence."""
    if type(config) is not EnvConfig or config.automatic or config.react_on_alarm or config.faults:
        raise ValueError('Unforced manual configuration without scheduled faults required')
    source=identity();started=time.perf_counter();samples=[];failure=None
    try:
        env=StationEnv(config);env.reset();samples.append(read_heat(env))
        for _ in range(config.max_steps):
            _,_,terminated,truncated,_=env.step({});samples.append(read_heat(env))
            if terminated or truncated:break
    except Exception as error:
        failure=dict(type=type(error).__name__,code=error.code if isinstance(error,ActionError) else 'execution_error',message=str(error))
    accounting=assess_heat(samples) if samples else None
    complete=failure is None and len(samples)==config.max_steps+1
    if source!=identity():raise ValueError('Heat evaluator/backend changed during execution')
    return dict(config=asdict(config),source=source,samples=samples,accounting=accounting,failure=failure,
        complete=complete,passed=bool(complete and accounting['passed']),wall_s=time.perf_counter()-started,
        heat_quadrature_only_refinement=False,industrial_thermal_efficiency_claimed=False)
