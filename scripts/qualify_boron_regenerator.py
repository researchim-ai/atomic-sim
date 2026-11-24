"""Finite PWR exchange with optional steady counterflow heat recovery.

Local numerical acceptance only; not plant CVCS or experimental qualification.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_bvp
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from coupling.pwr_boron import BoronExchangeConfig,BoronExchangePWRSession,identity
from coupling import refinement
from scripts import qualify_boron_exchange as chemical

CASES=dict(dilution=0.,boration=2000.)
MODES=dict(bypass=0.,recovery=200000.)
CRITERIA=dict(temporal_relative=1e-4,heat_relative=1e-8,port_relative=1e-10,
    minimum_receiver_cooling_k=.01,minimum_peak_recovery_w=1e6)


def source():
    return dict(model=identity(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helpers_sha256={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        for m in (refinement,chemical,chemical.budgets)})


def capacity_reference(x,flow,*,max_nodes=2000):
    """Independent differential boundary problem; no epsilon-NTU formula."""
    if type(max_nodes) is not int or not 8<=max_nodes<=10000:raise ValueError('Reference node budget in 8..10000 required')
    hot=x['hot_inlet_temperature_k'];cold=x['cold_inlet_temperature_k'];ua=x['ua_w_k']
    if flow==0 or ua==0:return dict(heat_w=0.,hot_outlet_k=hot,cold_outlet_k=cold)
    ch=flow*x['hot_cp_j_kg_k'];cc=flow*x['cold_cp_j_kg_k']
    def rhs(z,y):
        gap=y[0]-y[1];return np.vstack((-ua/ch*gap,-ua/cc*gap))
    def boundary(a,b):return np.array([a[0]-hot,b[1]-cold])
    z=np.linspace(0.,1.,8);guess=np.vstack((np.full(8,hot),np.full(8,cold)))
    solution=solve_bvp(rhs,boundary,z,guess,tol=1e-9,max_nodes=max_nodes)
    if not solution.success:raise ValueError('Counterflow reference boundary solve did not converge')
    return dict(heat_w=float(ch*(hot-solution.y[0,-1])),hot_outlet_k=float(solution.y[0,-1]),cold_outlet_k=float(solution.y[1,0]))


def close(a,b,tol=CRITERIA['port_relative']):
    return bool(abs(a-b)<=tol*max(abs(a),abs(b),1.))


def audit(row,metadata,*,reference_max_nodes=2000,reference_solver=None,expected_transferred_mass=None):
    x=row['finite_boron_exchange'];g=x['regenerator'];ports=x['exchange_ports'];q=x['flow_kg_s'];heat=g['heat_w']
    solver=capacity_reference if reference_solver is None else reference_solver
    if not callable(solver):raise ValueError('Callable independent reference required')
    reference=solver(g,q,max_nodes=reference_max_nodes);tanks=x['tanks'];rho=metadata['rho']
    primary=q*(ports['supply_enthalpy_j_kg']-ports['primary_donor_enthalpy_j_kg'])+x['electrical_power_w']+heat
    receiver=q*ports['primary_donor_enthalpy_j_kg']-heat-tanks['receiver']['pressure_pa']*q/rho
    supply=-q*ports['supply_enthalpy_j_kg']+tanks['supply']['pressure_pa']*q/rho
    rise=(ports['pump_specific_work_j_kg']-ports['pump_head_pa']/rho)/4180.
    entropy=g['hot_capacity_w_k']*math.log(g['hot_outlet_capacity_temperature_k']/g['hot_inlet_temperature_k'])+g['cold_capacity_w_k']*math.log(g['cold_outlet_capacity_temperature_k']/g['cold_inlet_temperature_k'])
    return dict(reference_heat=close(heat,reference['heat_w'],CRITERIA['heat_relative']),
        reference_outlets=close(g['hot_outlet_capacity_temperature_k'],reference['hot_outlet_k'],1e-9) and close(g['cold_outlet_capacity_temperature_k'],reference['cold_outlet_k'],1e-9),
        capacities=close(g['hot_capacity_w_k'],q*g['hot_cp_j_kg_k']) and close(g['cold_capacity_w_k'],q*4180.) and g['cold_cp_j_kg_k']==4180.,
        capacity_entropy=g['capacity_entropy_generation_w_k']>=-1e-6 and close(g['capacity_entropy_generation_w_k'],entropy,1e-8),
        primary_port=close(x['primary_energy_rate_w'],primary),
        tank_ports=close(x['tank_water_energy_rates_w'][0],supply) and close(x['tank_water_energy_rates_w'][1],receiver),
        pump=close(x['electrical_power_w'],q*ports['pump_specific_work_j_kg']) and close(x['electrical_power_w'],row['powers']['boron_pump_electric_w']) and close(ports['pump_specific_work_j_kg'],ports['pump_head_pa']/(rho*ports['pump_efficiency'])),
        pump_temperature=close(ports['pump_sensible_temperature_rise_k'],rise) and close(g['cold_inlet_temperature_k'],tanks['supply']['temperature_k']+rise),
        conductance=g['ua_w_k']==metadata['ua_w_k'],
        transfer=all(chemical.audit(row,metadata,expected_transferred_mass=expected_transferred_mass).values()))


def assess(cases):
    if set(cases)!=set(CASES):raise ValueError('Both chemical cases required')
    expected={f'{mode}_{preset}' for mode in MODES for preset in chemical.PRESETS}
    if any(set(t)!=expected for t in cases.values()):raise ValueError('Complete bypass/recovery temporal matrix required')
    gates=dict(configuration=True,timeline=True,commands=True,budgets=True,exchange=True,reference_frozen=True,
        restart=True,rollback=True,matched_initial=True,cooling=True,recovery=True)
    errors=[];benefits={};maximum_heat=0.
    for case,trajectories in cases.items():
        for key,t in trajectories.items():
            mode,preset=key.split('_');ua=MODES[mode]
            if len(t['snapshots'])!=len(chemical.STEPS):raise ValueError('Complete exchange timeline required')
            gates['configuration'] &= t['checkpoint']['payload']['config']==asdict(BoronExchangeConfig(supply_boron_ppm=CASES[case],regenerator_ua_w_k=ua)) and t['metadata']['ua_w_k']==ua
            gates['commands'] &= t['steps']==chemical.STEPS and t['checkpoint']['payload']['history']==chemical.STEPS
            gates['timeline'] &= [r['time_s'] for r in t['snapshots']]==[.1,.5,1.,2.,2.5]
            for row in [t['initial'],*t['snapshots']]:
                gates['budgets'] &= all(chemical.budgets.checks(row).values())
                gates['exchange'] &= all(audit(row,t['metadata']).values())
                gates['reference_frozen'] &= row['local_moderator']['reference']==t['initial']['local_moderator']['reference']
                maximum_heat=max(maximum_heat,row['finite_boron_exchange']['regenerator']['heat_w'])
            gates['rollback'] &= t['rollback'] is True
            if key=='recovery_fine':gates['restart'] &= t['restart_exact'] is True
            if preset=='coarse':
                fine=trajectories[mode+'_fine']
                errors.extend(chemical.budgets.relative(a['coupled_state'],np.asarray(b['coupled_state'])) for a,b in zip([t['initial'],*t['snapshots']],[fine['initial'],*fine['snapshots']]))
        left,right=(trajectories[m+'_fine'] for m in MODES)
        gates['matched_initial'] &= left['initial']['coupled_state']==right['initial']['coupled_state']
        a=left['snapshots'][-1]['finite_boron_exchange']['tanks']['receiver'];b=right['snapshots'][-1]['finite_boron_exchange']['tanks']['receiver']
        benefits[case]=dict(receiver_cooling_k=a['temperature_k']-b['temperature_k'],receiver_water_energy_reduction_j=a['water_energy_j']-b['water_energy_j'])
        gates['cooling'] &= benefits[case]['receiver_cooling_k']>=CRITERIA['minimum_receiver_cooling_k'] and benefits[case]['receiver_water_energy_reduction_j']>0
        gates['recovery'] &= max(r['finite_boron_exchange']['regenerator']['heat_w'] for r in right['snapshots'])>=CRITERIA['minimum_peak_recovery_w']
    gates['temporal']=max(errors)<=CRITERIA['temporal_relative']
    return dict(passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},temporal_maximum_relative=max(errors),benefits=benefits,maximum_recovered_heat_w=maximum_heat)


def negative_controls(cases):
    checks={}
    def reject(name,modify,gate):
        copied=deepcopy(cases);modify(copied['dilution']['recovery_fine']);result=assess(copied)
        checks[name]=not result['passed'] and not result['gates'][gate]
    def final(t):return t['snapshots'][-1]['finite_boron_exchange']
    reject('wrong_recovered_heat',lambda t:final(t)['regenerator'].__setitem__('heat_w',0.),'exchange')
    reject('missing_receiver_cooling',lambda t:final(t)['tank_water_energy_rates_w'].__setitem__(1,final(t)['tank_water_energy_rates_w'][1]+final(t)['regenerator']['heat_w']),'exchange')
    reject('missing_primary_heat',lambda t:final(t).__setitem__('primary_energy_rate_w',final(t)['primary_energy_rate_w']-final(t)['regenerator']['heat_w']),'exchange')
    reject('negative_capacity_entropy',lambda t:final(t)['regenerator'].__setitem__('capacity_entropy_generation_w_k',-1e6),'exchange')
    reject('wrong_conductance',lambda t:t['metadata'].__setitem__('ua_w_k',0.),'configuration')
    reject('invented_pump_power',lambda t:final(t).__setitem__('electrical_power_w',final(t)['electrical_power_w']+1e6),'exchange')
    reject('failed_restart',lambda t:t.__setitem__('restart_exact',False),'restart')
    return dict(passed=all(checks.values()),checks=checks,scope='Mutated copies only; native trajectories unchanged')


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);frozen=source();cases={}
    for case,ppm in CASES.items():
        trajectories={}
        for mode,ua in MODES.items():
            for preset,options in chemical.PRESETS.items():
                key=f'{mode}_{preset}';print(case,key,flush=True)
                s=BoronExchangePWRSession(**refinement.coupled_pwr_mesh(2),boron=BoronExchangeConfig(supply_boron_ppm=ppm,regenerator_ua_w_k=ua),**options)
                first=s.snapshot();rows=[];saved=None
                meta=dict(cold_index=13+3*s.plant.network.volume_names().index('cold_leg'),rho=s.boron_rho,
                    supply_mass=s.boron_config.supply_mass_kg,receiver_mass=s.boron_config.receiver_mass_kg,supply_fraction=ppm*1e-6,ua_w_k=ua)
                for i,step in enumerate(chemical.STEPS):
                    rows.append(s.advance(**step));print(case,key,s.time,flush=True)
                    if i==1:saved=json.loads(json.dumps(s.checkpoint()))
                restart=None
                if key=='recovery_fine':
                    restored=BoronExchangePWRSession.restore(saved)
                    restart=[restored.advance(**step) for step in chemical.STEPS[2:]]==rows[2:] and restored.checkpoint()==s.checkpoint()
                before=s.checkpoint()
                try:s.advance(.01,{'boron_flow_target_kg_s':1000.})
                except ValueError:rollback=s.checkpoint()==before
                else:rollback=False
                trajectories[key]=dict(initial=first,snapshots=rows,metadata=meta,steps=deepcopy(chemical.STEPS),restart_exact=restart,rollback=rollback,checkpoint=before)
                atomic_json(out/f'{case}_{key}.json',seal(trajectories[key]))
        cases[case]=trajectories
    assessment=assess(cases);negative=negative_controls(cases)
    report=dict(source=frozen,criteria=CRITERIA,assessment=assessment,negative_controls=negative,
        passed=assessment['passed'] and negative['passed'] and source()==frozen,trajectories=8,restart_suffixes=2,horizon_s=2.5,
        fluid_cells=2,thermal_cells=4,neutron_cells=2,experimental_validation=False,long_horizon_accepted=False,
        spatial_qualification=False,station_env_connected=False,closed_tasks=[],
        closure='Steady zero-storage regenerator; inlet-frozen heat capacities, explicit pump pressure/sensible work split; no outlet EOS flash')
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,cases=cases)))
    if not report['passed']:raise ValueError('Regenerative exchange qualification failed; attempts retained')
    return dict(passed=True,assessment=assessment)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
