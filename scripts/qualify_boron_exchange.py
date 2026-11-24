"""Matched finite dilution/boration transients with thermal inputs held equal."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from coupling.pwr_boron import BoronExchangeConfig,BoronExchangePWRSession,identity
from coupling import refinement
from coupling.refinement import coupled_pwr_mesh
from scripts import qualify_pwr_distributed_fuel as budgets

PRESETS=dict(coarse=dict(max_step_s=.1,rtol=2e-8),fine=dict(max_step_s=.05,rtol=2e-9))
STEPS=[dict(duration_s=.1,patch={'boron_flow_target_kg_s':50.}),dict(duration_s=.4,patch=None),
       dict(duration_s=.5,patch=None),dict(duration_s=1.,patch=None),dict(duration_s=.5,patch={'boron_flow_target_kg_s':0.})]
CRITERIA=dict(refinement_relative=1e-4,mass_integral_abs_kg=1e-5,transfer_relative=1e-10,
    minimum_primary_boron_change_kg=.001,minimum_power_effect_relative=1e-4,numerical_margin=20.)


def source():
    return dict(model=identity(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helpers_sha256={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (refinement,budgets)})


def transferred_mass(time,flow=50.,tau=.2,close=2.):
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (time,flow,tau,close)) or not 0<=time<=3600 or flow<0 or tau<=0 or close<0:
        raise ValueError('Finite nonnegative actuator timing and positive tau required')
    t=min(time,close);mass=flow*(t-tau*(1-math.exp(-t/tau)))
    if time>close:mass+=flow*(1-math.exp(-close/tau))*tau*(1-math.exp(-(time-close)/tau))
    return mass


def audit(row,initial,*,expected_transferred_mass=None):
    x=row['finite_boron_exchange'];tanks=x['tanks'];q=x['flow_kg_s'];rho=initial['rho'];c=tanks['supply']['boron_fraction']
    cold=row['coupled_state'][initial['cold_index']:initial['cold_index']+3];cold_c=cold[2]/cold[0]
    water=np.asarray(x['tank_water_energy_rates_w']);dm=np.asarray(x['tank_mass_rates_kg_s']);db=np.asarray(x['tank_boron_rates_kg_s'])
    gas=np.array([tanks[n]['pressure_pa'] for n in ('supply','receiver')])*dm/rho
    residual=x['primary_energy_rate_w']+sum(water+gas)-x['electrical_power_w']
    mass=transferred_mass(row['time_s']) if expected_transferred_mass is None else expected_transferred_mass
    if type(mass) not in (int,float) or not math.isfinite(mass) or mass<0:raise ValueError('Finite nonnegative reference transferred mass required')
    return dict(energy=abs(residual)/max(abs(x['primary_energy_rate_w']),1.)<=CRITERIA['transfer_relative'],
        boron=bool(max(abs(db-np.array([-q*c,q*cold_c])))<=1e-12 and abs(x['primary_boron_rate_kg_s']+sum(db))<=1e-12),
        mass_integral=abs(tanks['supply']['mass_kg']-(initial['supply_mass']-mass))<=CRITERIA['mass_integral_abs_kg'] and abs(tanks['receiver']['mass_kg']-(initial['receiver_mass']+mass))<=CRITERIA['mass_integral_abs_kg'],
        finite_reserve=all(t['mass_kg']>=20. for t in tanks.values()),
        source_concentration=abs(c-initial['supply_fraction'])<=1e-12,
        electrical=0<=x['hydraulic_power_w']<=x['electrical_power_w'] and x['integrated_pump_electric_j']>=0.)


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);frozen=source();trajectories={}
    for case,ppm in (('dilution',0.),('boration',2000.)):
        for preset,options in PRESETS.items():
            print(case,preset,flush=True)
            s=BoronExchangePWRSession(**coupled_pwr_mesh(2),boron=BoronExchangeConfig(supply_boron_ppm=ppm),**options)
            first=s.snapshot();rows=[];saved=None
            meta=dict(cold_index=13+3*s.plant.network.volume_names().index('cold_leg'),rho=s.boron_rho,
                supply_mass=s.boron_config.supply_mass_kg,receiver_mass=s.boron_config.receiver_mass_kg,supply_fraction=ppm*1e-6,base_size=s.boron_index)
            for i,step in enumerate(STEPS):
                rows.append(s.advance(**step));print(case,preset,s.time,flush=True)
                if i==1:saved=json.loads(json.dumps(s.checkpoint()))
            restart=None
            if preset=='fine':
                restored=BoronExchangePWRSession.restore(saved)
                restart=[restored.advance(**step) for step in STEPS[2:]]==rows[2:] and restored.checkpoint()==s.checkpoint()
            before=s.checkpoint()
            try:s.advance(.01,{'boron_flow_target_kg_s':1000.})
            except ValueError:rollback=s.checkpoint()==before
            else:rollback=False
            trajectories[f'{case}_{preset}']=dict(initial=first,snapshots=rows,metadata=meta,restart_exact=restart,rollback=rollback,checkpoint=before)
            atomic_json(out/f'{case}_{preset}.json',seal(trajectories[f'{case}_{preset}']))
    assessment=assess(trajectories)
    report=dict(source=frozen,criteria=CRITERIA,assessment=assessment,passed=assessment['passed'] and source()==frozen,
        trajectories=4,restart_suffixes=2,horizon_s=2.5,external_boron_source=False,finite_tanks=True,
        experimental_validation=False,long_horizon_accepted=False,station_env_connected=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,trajectories=trajectories)))
    if not report['passed']:raise ValueError('Finite boron qualification failed; attempts retained')
    return dict(passed=True,power_effect=assessment['power_effect_relative'])


def primary_boron(row):
    t=row['finite_boron_exchange'];return t['total_boron_kg']-sum(v['boron_kg'] for v in t['tanks'].values())


def assess(trajectories):
    if set(trajectories)!={f'{c}_{p}' for c in ('dilution','boration') for p in PRESETS}:raise ValueError('Complete dilution/boration temporal matrix required')
    gates=dict(budgets=True,transfer=True,rollback=True,timeline=True,restart=True,direction=True,reference_frozen=True)
    errors=[];power={};changes={}
    for key,t in trajectories.items():
        if len(t['snapshots'])!=5:raise ValueError('Complete finite exchange timeline required')
        for row,time in zip(t['snapshots'],(.1,.5,1.,2.,2.5)):
            gates['timeline'] &= row['time_s']==time
            gates['budgets'] &= all(budgets.checks(row).values())
            gates['transfer'] &= all(audit(row,t['metadata']).values())
            gates['reference_frozen'] &= row['local_moderator']['reference']==t['initial']['local_moderator']['reference']
        gates['rollback'] &= t['rollback'] is True
        if key.endswith('_fine'):gates['restart'] &= t['restart_exact'] is True
        power[key]=np.array([r['powers']['nuclear_fission_w'] for r in t['snapshots']])
        changes[key]=primary_boron(t['snapshots'][-1])-primary_boron(t['initial'])
        gates['direction'] &= changes[key]<-CRITERIA['minimum_primary_boron_change_kg'] if key.startswith('dilution') else changes[key]>CRITERIA['minimum_primary_boron_change_kg']
    for case in ('dilution','boration'):
        errors.extend(budgets.relative(a['coupled_state'],np.asarray(b['coupled_state'])) for a,b in zip(trajectories[case+'_coarse']['snapshots'],trajectories[case+'_fine']['snapshots']))
    a,b=(trajectories[c+'_fine'] for c in ('dilution','boration'));n=a['metadata']['base_size']
    gates['matched_initial_primary']=a['initial']['coupled_state'][:n]==b['initial']['coupled_state'][:n]
    ca=dict(a['checkpoint']['payload']['config']);cb=dict(b['checkpoint']['payload']['config'])
    ca.pop('supply_boron_ppm');cb.pop('supply_boron_ppm');gates['matched_equipment']=ca==cb
    p0=float(a['initial']['powers']['nuclear_fission_w'])
    effect=float(max(abs(power['dilution_fine']-power['boration_fine']))/p0)
    noise=float(max(max(abs(power[c+'_coarse']-power[c+'_fine'])) for c in ('dilution','boration'))/p0)
    gates.update(refined=max(errors)<=CRITERIA['refinement_relative'],effect_resolved=effect>=CRITERIA['minimum_power_effect_relative'] and effect>=CRITERIA['numerical_margin']*noise)
    return dict(passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},refinement_relative=max(errors),
        primary_boron_changes_kg=changes,power_effect_relative=effect,numerical_power_change_relative=noise)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
