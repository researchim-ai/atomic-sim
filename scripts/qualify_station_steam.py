"""Finite-header/manual-admission research qualification; no complete steam train."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from atomic_model_station import PWRSteamSession, SteamConfig
from atomic_model_station.steam import steam_signature, actual_expansion
from atomic_model_plant_extension import water
from atomic_model_multiphysics.provenance import sha
from physics.materials import iapws
from physics.power.steam_turbine_train import expansion
from scripts import qualify_station_actuators as evidence
from scripts.qualify_station_primary import vector as primary_vector

TASKS=['NPP-05.01']
SETTINGS=[(1.,1e-7),(.5,2e-8),(.2,2e-9)]
CRITERIA=dict(energy_relative=1e-6,mass_relative=1e-10,boron_relative=1e-10,
    instantaneous_relative=1e-8,temporal_relative=1e-4,nominal_tail_relative=1e-5,
    routing_energy_w=.01,condenser_energy_w=.01,entropy_roundoff_w_k=.001,
    pressure_reference_pa=.05,quality_reference=1e-8,expansion_reference_relative=1e-10)


def vector(r):return np.r_[primary_vector(r),r['header_inventory']]


def admit(r):
    h=r['steam_header'];b=r['condenser_balance']
    return bool(r['energy_relative']<=CRITERIA['energy_relative'] and r['mass_relative']<=CRITERIA['mass_relative']
        and r['boron_relative']<=CRITERIA['boron_relative']
        and abs(r['instantaneous_energy_defect_w'])/max(abs(r['powers']['nuclear_heat_w']),1.)<=CRITERIA['instantaneous_relative']
        and abs(h['routing_energy_residual_w'])<=CRITERIA['routing_energy_w']
        and abs(b['energy_residual_w'])<=CRITERIA['condenser_energy_w'] and abs(b['mass_residual_kg_s'])<1e-8
        and min(h[k] for k in ('inlet_valve_entropy_w_k','mixing_entropy_w_k','heat_entropy_w_k','expansion_entropy_w_k'))>=-CRITERIA['entropy_roundoff_w_k']
        and h['outlet_pressure_ratio']<=.01 and h['quality']>=r['steam_config']['minimum_quality']
        and h['inlet_flow_kg_s']>=0 and h['turbine_flow_kg_s']>=0)


def inventory_reference():
    """Analytic M/U path at frozen ports + independent scalar mixture inversion."""
    s=PWRSteamSession();h=s.header;initial=h.thermo;rows=[];mass0=h.mass_kg;energy0=h.internal_energy_j
    for t in (0.,.005,.01,.02):
        mass=mass0+100*t;energy=energy0+100*initial.h_j_kg*t
        rho=mass/h.volume_m3;u=energy/mass
        def mixture(p):
            f,g=water.saturation(p)
            x=(1/rho-1/f.density_kg_m3)/(1/g.density_kg_m3-1/f.density_kg_m3)
            return (1-x)*f.internal_energy_j_kg+x*g.internal_energy_j_kg-u,x
        pressure=brentq(lambda p:mixture(p)[0],6e6,6.6e6,xtol=1e-6)
        quality=mixture(pressure)[1]
        if not 0<=quality<=1:raise ValueError('Frozen scalar reference left physical mixture')
        y=s.state.copy();y[s.steam_index]=mass;y[s.steam_index+1]=energy
        point=s._sync_header(y)
        rows.append(dict(time_s=t,mass_kg=mass,energy_j=energy,scalar_pressure_pa=pressure,
            scalar_quality=quality,newton_pressure_pa=float(point.p_pa),newton_quality=float(point.quality),
            pressure_error_pa=abs(float(point.p_pa)-pressure),quality_error=abs(float(point.quality)-quality)))
    pin,pout,eta=6e6,8000.,.85;hin=iapws.saturation_properties(pin)[1].h_j_kg
    actual=actual_expansion(pin,hin,pout,eta)
    prior=expansion(dict(inlet_pressure_pa=pin,outlet_pressure_pa=pout,isentropic_efficiency=eta))
    error=max(abs(actual[k]-prior[k])/max(abs(prior[k]),1.) for k in prior)
    return dict(path=rows,dry_limit_relative_error=float(error),
        scope='Frozen prescribed fluxes qin1000/qout900kg/s, equal initial port h, no heat; analytic stocks and independent scalar root, not station dynamics')


def run(output,reference=None):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    output.mkdir(parents=True,exist_ok=False)
    source=steam_signature();driver=sha(__file__);serializer=sha(evidence.__file__)
    s=PWRSteamSession(max_step_s=1.);initial=s.snapshot()
    evidence.save(output/'protocol.json',dict(source=source,driver_sha256=driver,serializer_sha256=serializer,
        criteria=CRITERIA,settings=SETTINGS,initial=initial,parameters=s.parameter_registry(),candidate_tasks=TASKS,
        nominal_duration_s=3600.,trip_duration_s=60.,
        scope='Single finite well-mixed near-dry header, manual admission, small-drop one-way inlet, no pipe waves or manufacturer maps',
        valve='Actual finite linear strokes, analytical arrival events; fluxes derived algebraically, no ghost pipe kinetic stock',
        charging='Added header M/U from explicit initial SG tap enthalpy and pressure loss; pre-existing stocks unchanged'))
    print('steam protocol sealed',flush=True)
    nominal=[]
    for i in range(60):
        nominal.append(s.advance(60.))
        if i%5==0:print('steam nominal',i+1,'/60',flush=True)
    tail=float(max(abs(vector(nominal[-1])-vector(nominal[-3]))/np.maximum(abs(vector(nominal[-3])),1.)))
    evidence.save(output/'nominal.json',dict(trace=nominal,tail_relative_drift=tail,checkpoint=s.checkpoint()))
    s=PWRSteamSession(max_step_s=.5);regulated=[]
    for patch in [dict(steam_admission=.98),dict(steam_admission=1.)]:
        regulated.append(s.advance(10.,patch))
    evidence.save(output/'regulation.json',dict(trace=regulated,checkpoint=s.checkpoint()))
    s=PWRSteamSession(max_step_s=.5);trip=[]
    for i in range(6):
        trip.append(s.advance(10.,dict(scram=True,turbine_trip=True) if i==0 else None))
        print('steam trip',i+1,'/6',flush=True)
    evidence.save(output/'trip.json',dict(trace=trip,checkpoint=s.checkpoint()))
    s=PWRSteamSession(steam=SteamConfig(heat_loss_ua_w_k=1000.),max_step_s=.5)
    warm=s.advance(10.)
    evidence.save(output/'heat_loss.json',warm)
    s=PWRSteamSession(max_step_s=.2);isolated=s.advance(.01,dict(steam_inlet_isolated=True))
    reopened=s.advance(.02,dict(steam_inlet_isolated=False))
    evidence.save(output/'isolation.json',dict(early=isolated,reopened=reopened))
    trials=[]
    for step,rtol in SETTINGS:
        s=PWRSteamSession(max_step_s=step,rtol=rtol)
        moving=s.advance(.1,dict(scram=True,turbine_trip=True))
        saved=json.loads(json.dumps(s.checkpoint()));t=PWRSteamSession.restore(saved)
        final=s.advance(4.9);exact=t.advance(4.9)==final
        trials.append(dict(step_s=step,rtol=rtol,moving=moving,checkpoint=saved,final=final,restart_exact=exact))
        print('steam refinement',step,flush=True)
    fine=vector(trials[-1]['final']);errors=[float(max(abs(vector(t['final'])-fine)/np.maximum(abs(fine),1.))) for t in trials]
    refs=inventory_reference()
    evidence.save(output/'references.json',dict(trials=trials,refinement_errors=errors,independent=refs))
    before=s.checkpoint();rollback=False
    try:s.advance(.1,dict(scram=True,steam_admission=2.))
    except ValueError:rollback=s.checkpoint()==before
    evidence.save(output/'rollback.json',dict(exact=rollback,checkpoint=before))
    observations=[initial,*nominal,*regulated,*trip,warm,isolated,reopened,*[x[k] for x in trials for k in ('moving','final')]]
    unchanged=source==steam_signature() and driver==sha(__file__) and serializer==sha(evidence.__file__)
    held_mass=max(abs(r['steam_header']['mass_kg']-trip[0]['steam_header']['mass_kg']) for r in trip)
    closed=all(r['steam_header']['inlet_flow_kg_s']==0. and r['steam_header']['turbine_flow_kg_s']==0. for r in trip)
    pressure_error=max(x['pressure_error_pa'] for x in refs['path']);quality_error=max(x['quality_error'] for x in refs['path'])
    passed=bool(unchanged and all(admit(r) for r in observations) and closed and held_mass<1e-6 and rollback
        and all(t['restart_exact'] for t in trials) and max(errors)<=CRITERIA['temporal_relative']
        and tail<=CRITERIA['nominal_tail_relative'] and pressure_error<=CRITERIA['pressure_reference_pa']
        and quality_error<=CRITERIA['quality_reference'] and refs['dry_limit_relative_error']<=CRITERIA['expansion_reference_relative']
        and warm['steam_header']['heat_to_cooling_w']>0
        and regulated[0]['steam_header']['mass_kg']!=initial['steam_header']['mass_kg'])
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,closed_tasks=TASKS if passed else [],
        task_checks={'NPP-05.01':passed},nominal_duration_s=3600.,trip_duration_s=60.,
        nominal_tail_relative_drift=tail,refinement_errors=errors,restart_exact=all(t['restart_exact'] for t in trials),rollback=rollback,
        closed_header_mass_drift_kg=held_mass,closed_flows_exact=closed,pressure_reference_error_pa=pressure_error,
        quality_reference_error=quality_error,dry_expansion_reference_relative_error=refs['dry_limit_relative_error'],
        max_energy_relative=max(r['energy_relative'] for r in observations),max_mass_relative=max(r['mass_relative'] for r in observations),
        max_boron_relative=max(r['boron_relative'] for r in observations),
        max_routing_energy_residual_w=max(abs(r['steam_header']['routing_energy_residual_w']) for r in observations),
        complete_package_b=False,complete_package_c=False,r1_accepted=False,validated=False,plant_referenced=False,full_long_checkpoint_replay=False,
        limitations=['Single equilibrium header, no axial wave/inertia/condensate-film or water-hammer model',
            'Valve coefficients sized at one declared point, linear aggregate admission and constant expansion efficiency; no hardware/choked-flow map',
            'Manual admission only, automatic governor unavailable in this backend',
            'Near-dry header quality>=0.99, pressure5..8MPa, flowing inlet loss<=5%, outlet pressure ratio<=0.01',
            'No separator/MSR or moisture-erosion qualification, no header relief/bypass reconfiguration;NPP05.03/04 remain open',
            'Legacy SG dump still a separate direct condenser branch; no whole-stationR1 or plant validation'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        reference=Path(reference);files={f.relative_to(output) for f in output.rglob('*.json')}
        if files!={f.relative_to(reference) for f in reference.rglob('*.json')}:raise ValueError('Evidence sets differ')
        for name in sorted(files):
            if evidence.read(output/name)!=evidence.read(reference/name):raise ValueError('Installed steam evidence differs:'+str(name))
        print('All installed finite steam header evidence matches exactly',flush=True)
    if not passed:raise ValueError('Steam header qualification failed; no task accepted')
    print(json.dumps(dict(passed=True,closed_tasks=TASKS)),flush=True)
    return report


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path);args=parser.parse_args();run(args.output,args.reference)
