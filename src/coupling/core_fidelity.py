"""Bounded two-resolution PWR comparison and a protocol-locked approximation.

The four-node calculation is a numerical reference in the same synthetic
model, not experimental truth. No running state is projected between meshes.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_station import CoreConfig
from . import mesh_comparison
from .pwr_moderator import LocalModeratorPWRSession,identity as model_identity

CASES=('insertion','return')
MESHES=(2,4)
PRESETS=dict(coarse=dict(max_step_s=.1,rtol=2e-8),fine=dict(max_step_s=.05,rtol=2e-9))
LIMITS=dict(profile_l2=.05,power_relative=.001,temperature_k=5.,hydraulic_relative=.001,temporal_relative=1e-4)


def identity():
    return dict(model=model_identity(),policy_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        comparison_sha256=hashlib.sha256(Path(mesh_comparison.__file__).read_bytes()).hexdigest())


def protocol(case,position):
    if type(case) is not str or case not in CASES:raise ValueError('Only qualified insertion/return protocols supported')
    if type(position) not in (float,int) or not np.isfinite(position) or not 0<=position<=.99:raise ValueError('Finite initial rod trim required')
    steps=[dict(duration_s=.1,patch={'rod_target':position+.01}),dict(duration_s=.4,patch=None),dict(duration_s=.5,patch=None),
        dict(duration_s=1.,patch=None if case=='insertion' else {'rod_target':position})]
    if case=='return':steps.append(dict(duration_s=1.,patch=None))
    return steps


def session(cells,preset='fine'):
    if type(cells) is not int or cells not in MESHES or type(preset) is not str or preset not in PRESETS:
        raise ValueError('Qualified 2/4 nodes and numerical preset required')
    return LocalModeratorPWRSession(core=CoreConfig(kind='diffusion',cells=cells),**PRESETS[preset])


def discrepancy(a,b,reference):
    """Same thermal/fluid grids; exact overlap errors for neutron fields."""
    configs=[r['core_config'] for r in (a,b)]
    if any(c['kind']!='diffusion' for c in configs) or {k:v for k,v in configs[0].items() if k!='cells'}!={k:v for k,v in configs[1].items() if k!='cells'}:
        raise ValueError('Only neutron resolution may differ')
    if a['fluid_grid']!=b['fluid_grid'] or a['fuel_grid']!=b['fuel_grid']:raise ValueError('Fixed thermal/fluid grids required')
    faces=[np.linspace(0.,c['length_m'],c['cells']+1) for c in configs];p0=reference['powers']['nuclear_fission_w']
    profiles=dict(fission=mesh_comparison.profile_error(faces[0],a['core']['fission_by_cell_w'],faces[1],b['core']['fission_by_cell_w'],
        kind='cell_integral',scale=p0/reference['core_config']['length_m']))
    for i in range(6):
        profiles['precursor'+str(i)]=mesh_comparison.profile_error(faces[0],a['core']['local_precursors'][i],faces[1],b['core']['local_precursors'][i],
            kind='cell_average',scale=float(np.mean(reference['core']['local_precursors'][i])))
    temperature=max(float(np.max(abs(np.asarray(a[k]['temperature_k'])-np.asarray(b[k]['temperature_k']))))
        for k in ('distributed_fuel','distributed_cladding','distributed_fluid'))
    hydraulic=max(float(np.max(abs(np.asarray(a['distributed_fluid'][k])-np.asarray(b['distributed_fluid'][k]))/np.maximum(abs(np.asarray(b['distributed_fluid'][k])),1.)))
        for k in ('pressure_pa','face_flow_kg_s'))
    powers=max(abs(a['powers'][k]-b['powers'][k])/p0 for k in ('nuclear_fission_w','net_export_w'))
    gates=dict(profiles=all(p['normalized_l2']<=LIMITS['profile_l2'] for p in profiles.values()),
        temperature=temperature<=LIMITS['temperature_k'],hydraulic=hydraulic<=LIMITS['hydraulic_relative'],powers=powers<=LIMITS['power_relative'])
    return dict(passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},profiles=profiles,
        maximum_temperature_difference_k=temperature,maximum_hydraulic_relative=hydraulic,maximum_power_relative=powers)


def cost_decision(cost,case):
    if type(cost) is not dict or cost.get('format')!='core-fidelity-cost/v1' or cost.get('source')!=identity():
        raise ValueError('Current source-bound fidelity cost required')
    if cost.get('construction_timed') is not True or type(cost.get('repeats')) is not int or cost['repeats']!=3:
        raise ValueError('Complete-episode construction and three repeats required')
    if type(case) is not str or case not in CASES:raise ValueError('Qualified case required')
    measurements=cost.get('measurements')
    if type(measurements) is not dict or set(measurements)!=set(CASES) or any(type(v) is not dict or set(v)!={'2','4'} for v in measurements.values()):
        raise ValueError('Complete fidelity timing matrix required')
    rows=measurements[case]
    for key in ('2','4'):
        values=rows[key]
        if type(values) is not list or len(values)!=3 or any(type(v) not in (float,int) or not np.isfinite(v) or v<=0 for v in values):
            raise ValueError('Three complete positive timing repeats required')
    cheap=statistics.median(rows['2']);reference=statistics.median(rows['4'])
    return dict(cheap_eligible=cheap<reference,speed_ratio=reference/cheap,
        median_cheap_s=cheap,median_reference_s=reference,scope='Local full-episode medians; no universal speed claim')


def _budget_ok(r):
    f=r['distributed_fuel'];power=max(f['fission_w'],f['thermal_w'],1.)
    return (r['energy_relative']<=1e-9 and abs(r['energy_residual_j'])<=.1 and r['mass_relative']<=1e-10 and r['boron_relative']<=1e-10
        and abs(r['instantaneous_energy_defect_w'])/power<=1e-10 and abs(f['nuclear_storage_identity_residual_w'])/power<=1e-12
        and abs(f['fuel_transfer_residual_w'])/power<=1e-12 and abs(f['decay_exchange_residual_j'])<=.1
        and r['core']['neutron_precursor_inventory']['relative_defect']<=1e-10
        and max(f['exchange']['power_relative_defect'],f['exchange']['pairing_relative_defect'])<=1e-12
        and min(np.ravel(f['decay_stored_j']))>=0 and min(r['core']['owned_state'])>=0
        and all(300<=v<=2800 for v in f['temperature_k']))


def assess(cases):
    expected={f'{n}_{p}' for n in MESHES for p in PRESETS}
    if type(cases) is not dict or set(cases)!=set(CASES) or any(set(t)!=expected for t in cases.values()):
        raise ValueError('Complete fidelity scenario/resolution/time matrix required')
    gates=dict(budgets=True,commands=True,timeline=True,stationary=True,reference_frozen=True,temporal=True,
        profiles=True,temperature=True,hydraulic=True,powers=True,restart=True,rollback=True,configuration=True)
    metrics={};time_error=0.
    for case,trajectories in cases.items():
        reference=trajectories['4_fine']['initial'];profiles=[]
        for key,t in trajectories.items():
            n,preset=key.split('_');initial=t['initial'];steps=protocol(case,initial['actuator_status']['rod_position'])
            if len(t['snapshots'])!=len(steps):raise ValueError('Complete fidelity observations required')
            gates['commands'] &= t['steps']==steps and t['checkpoint']['payload']['history']==steps
            gates['timeline'] &= np.allclose([r['time_s'] for r in t['snapshots']],np.cumsum([s['duration_s'] for s in steps]),rtol=0,atol=1e-12)
            gates['configuration'] &= initial['core_config']['cells']==int(n) and initial['core_config']==dict(reference['core_config'],cells=int(n))
            gates['configuration'] &= initial['fluid_grid']==reference['fluid_grid'] and initial['fuel_grid']==reference['fuel_grid']
            gates['configuration'] &= abs(initial['powers']['nuclear_fission_w']-reference['powers']['nuclear_fission_w'])/reference['powers']['nuclear_fission_w']<=1e-8
            stationary=initial['local_moderator_stationarity']
            gates['stationary'] &= stationary['constrained_rank']==stationary['constrained_unknowns'] and stationary['max_selected_tolerance_ratio']<=1.
            gates['rollback'] &= t['rollback'] is True
            if preset=='fine':gates['restart'] &= t['restart_exact'] is True
            for r in [initial,*t['snapshots']]:
                gates['budgets'] &= _budget_ok(r)
                gates['reference_frozen'] &= r['local_moderator']['reference']==initial['local_moderator']['reference']
            if preset=='coarse':
                fine=trajectories[n+'_fine']
                for a,b in zip(t['snapshots'],fine['snapshots']):
                    time_error=max(time_error,float(np.max(abs(np.asarray(a['coupled_state'])-np.asarray(b['coupled_state']))/np.maximum(abs(np.asarray(b['coupled_state'])),1.))))
        left,right=trajectories['2_fine'],trajectories['4_fine']
        for a,b in zip([left['initial'],*left['snapshots']],[right['initial'],*right['snapshots']]):
            comparison=discrepancy(a,b,reference);profiles.append(comparison)
            for key in ('profiles','temperature','hydraulic','powers'):gates[key] &= comparison['gates'][key]
        metrics[case]=dict(comparisons=profiles,maximum_profile_l2=max(p['normalized_l2'] for x in profiles for p in x['profiles'].values()),
            maximum_power_relative=max(x['maximum_power_relative'] for x in profiles),maximum_temperature_difference_k=max(x['maximum_temperature_difference_k'] for x in profiles),
            maximum_hydraulic_relative=max(x['maximum_hydraulic_relative'] for x in profiles))
    gates['temporal'] &= time_error<=LIMITS['temporal_relative']
    return dict(passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},temporal_maximum_relative=time_error,cases=metrics)


def validate_science(science,*,with_seeds=False):
    p=unseal(deepcopy(science))
    if type(p) is not dict or set(p)!={'policy_source','cases','report'} or p['policy_source']!=identity():
        raise ValueError('Current source-bound core fidelity evidence required')
    report=p['report'];expected_controls={'power_bias','sixth_precursor_bias','energy_failure','restart_failure','temperature_bias','rollback_failure'}
    if (type(report) is not dict or report.get('passed') is not True or type(report.get('negative_controls')) is not dict
            or report['negative_controls'].get('passed') is not True
            or type(report['negative_controls'].get('checks')) is not dict
            or set(report['negative_controls']['checks'])!=expected_controls
            or any(v is not True for v in report['negative_controls']['checks'].values())):
        raise ValueError('Accepted report and complete negative controls required')
    try:result=assess(p['cases'])
    except (KeyError,TypeError,IndexError) as exc:raise ValueError('Incomplete core fidelity evidence') from exc
    if not result['passed'] or result!=p['report']['assessment']:raise ValueError('Core fidelity evidence did not pass reassessment')
    seeds={}
    for n in MESHES:
        seeds[n]=session(n);anchor=seeds[n].snapshot()
        if any(p['cases'][case][str(n)+'_fine']['initial']!=anchor for case in CASES):
            raise ValueError('Evidence does not match the current default recipe anchor')
    return (p,seeds) if with_seeds else p


class QualifiedCoreRun:
    """Only a verified fine-preset protocol; no arbitrary commands or remapping."""
    def __init__(self,science,cost,*,case):
        _,seeds=validate_science(science,with_seeds=True)
        decision=cost_decision(unseal(deepcopy(cost)),case)
        self.case=case;self.cells=2 if decision['cheap_eligible'] else 4
        self.decision=decision;self._evidence_sha256=hashlib.sha256(json.dumps([science,cost],sort_keys=True,allow_nan=False).encode()).hexdigest()
        self._session=seeds[self.cells];self._steps=protocol(case,self._session.snapshot()['actuator_status']['rod_position']);self._next=0

    def observation(self):
        return dict(snapshot=self._session.snapshot(),resolution=dict(cells=self.cells,reference_cells=4,
            case=self.case,next_step=self._next,decision=deepcopy(self.decision),arbitrary_commands_allowed=False,
            source=identity(),experimental_validation=False))

    def advance(self,duration_s,patch=None):
        if self._next>=len(self._steps):raise ValueError('Qualified protocol is complete')
        expected=self._steps[self._next]
        if type(duration_s) not in (float,int) or not np.isfinite(duration_s) or duration_s!=expected['duration_s'] or patch!=expected['patch']:
            raise ValueError('Outside qualified protocol; use a fresh reference run for another command history')
        self._session.advance(**deepcopy(expected));self._next+=1
        return self.observation()

    def checkpoint(self):
        return seal(dict(format='qualified-core-run/v1',source=identity(),case=self.case,next_step=self._next,
            evidence_sha256=self._evidence_sha256,cells=self.cells,session=self._session.checkpoint()))

    @classmethod
    def restore(cls,saved,science,cost):
        p=unseal(deepcopy(saved))
        if (type(p) is not dict or set(p)!={'format','source','case','next_step','evidence_sha256','cells','session'}
                or p['format']!='qualified-core-run/v1' or p['source']!=identity()):raise ValueError('Qualified core checkpoint source/format')
        if type(p['next_step']) is not int or not 0<=p['next_step']<=5:raise ValueError('Qualified core replay budget')
        obj=cls(science,cost,case=p['case'])
        if obj.cells!=p['cells'] or obj._evidence_sha256!=p['evidence_sha256']:raise ValueError('Qualified core evidence changed')
        for step in obj._steps[:p['next_step']]:obj.advance(**step)
        if obj.checkpoint()!=saved:raise ValueError('Qualified core replay differs')
        return obj
