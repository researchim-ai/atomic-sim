"""Bounded independent action trials, explainable metrics and optimistic commit.

Constraints are checked at observation boundaries, not certified between steps.
No hardware safety decision or uncertainty/experimental validation is implied.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from atomic_model_hem_heated import HeatedHEMNetwork
from atomic_model_hem_heated.model import canonical,signature as physics_signature
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_workspace.common import keys

MAX_CANDIDATES=16
MAX_STEPS=8
MAX_TOTAL_HORIZON_S=.02


def signature():
    return dict(physics=physics_signature(),decision={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob('*.py'))})


def number(value,low,high):
    if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
        raise ValueError('Finite decision number within declared bounds required')
    return float(value)


def admit(model,candidates,target,limits):
    if type(model) is not HeatedHEMNetwork:raise ValueError('HeatedHEMNetwork required')
    if not isinstance(candidates,list) or not 1<=len(candidates)<=MAX_CANDIDATES:raise ValueError('Candidate budget1..16')
    keys(target,('wall','temperature_k'));keys(limits,('wall_temperature_max_k','fluid_temperature_max_k','pressure_max_pa'))
    if target['wall'] not in model.wall_ids:raise ValueError('Unknown target wall')
    number(target['temperature_k'],300,900)
    number(limits['wall_temperature_max_k'],300,900);number(limits['fluid_temperature_max_k'],300,623.15)
    number(limits['pressure_max_pa'],1e5,16e6)
    names=set();total=0.
    for candidate in candidates:
        keys(candidate,('id','steps'));name=candidate['id']
        if not isinstance(name,str) or not name.isascii() or not name.replace('_','').isalnum() or len(name)>48 or name in names:
            raise ValueError('Unique ASCII candidate id required')
        names.add(name);steps=candidate['steps']
        if not isinstance(steps,list) or not 1<=len(steps)<=MAX_STEPS:raise ValueError('Decision step budget1..8')
        horizon=0.
        for step in steps:
            keys(step,('duration_s','controls'))
            horizon+=number(step['duration_s'],1e-9,.005)
            # Preflight schema and physical input power budget; numerical domain
            # failures during integration remain per-candidate outcomes.
            model._mask(step['controls'])
        if horizon>.005:raise ValueError('Candidate horizon exceeds5ms')
        total+=horizon
    if total>MAX_TOTAL_HORIZON_S:raise ValueError('Combined simulated horizon exceeds20ms')
    # Ranking horizons must be equal: otherwise doing less could look better.
    horizons=[math.fsum(s['duration_s'] for s in c['steps']) for c in candidates]
    if any(h!=horizons[0] for h in horizons):raise ValueError('Equal candidate horizons required')
    grids=[tuple(s['duration_s'] for s in c['steps']) for c in candidates]
    if any(g!=grids[0] for g in grids):raise ValueError('Identical decision grids required')
    if len(canonical(dict(candidates=candidates,target=target,limits=limits)).encode())>65536:raise ValueError('Decision input size budget')


def violations(observation,limits):
    result=[]
    for wall in observation['walls']:
        if wall['temperature_k']>limits['wall_temperature_max_k']:
            result.append(dict(kind='wall_temperature',id=wall['id'],value=wall['temperature_k'],limit=limits['wall_temperature_max_k'],unit='K',time_s=observation['time_s']))
    for cell in observation['cells']:
        for key,limit,unit in [('temperature_k','fluid_temperature_max_k','K'),('pressure_pa','pressure_max_pa','Pa')]:
            if cell[key]>limits[limit]:result.append(dict(kind=key,id=cell['id'],value=cell[key],limit=limits[limit],unit=unit,time_s=observation['time_s']))
    return result


def branch(request,scheme,checkpoint,candidate,target,limits):
    model=HeatedHEMNetwork(request,scheme=scheme);model.restore(checkpoint)
    points=[model.snapshot()];failure=None;tracking=effort=0.
    for index,step in enumerate(candidate['steps']):
        try:
            obs=model.step(step['controls'],step['duration_s'],expected_revision=points[-1]['revision'],
                           controller_state={'decision_candidate':candidate['id'],'decision_step':index})
        except ValueError as error:
            failure=dict(step=index,reason=str(error));break
        points.append(obs)
        tw=next(w['temperature_k'] for w in obs['walls'] if w['id']==target['wall'])
        tracking+=abs(tw-target['temperature_k'])*step['duration_s']
        effort+=step['duration_s']*math.fsum(abs(v) for part in step['controls'].values() for v in part.values())
    breaches=[v for p in points for v in violations(p,limits)]
    mass=max(abs(p['global_ledger']['mass_residual_kg']) for p in points)
    energy=max(abs(p['global_ledger']['energy_residual_j']) for p in points)
    entropy=min(p['global_ledger']['numerical_entropy_residual_j_k'] for p in points)
    balances=mass<=1e-9 and energy<=1e-5 and entropy>=-2e-5
    status='model_rejected' if failure else 'balance_rejected' if not balances else 'constraint_violated' if breaches else 'eligible'
    return dict(id=candidate['id'],status=status,failure=failure,violations=breaches,
        metrics=dict(tracking_error_k_s=tracking,absolute_commanded_energy_j=effort,mass_residual_kg=mass,
                     energy_residual_j=energy,numerical_entropy_min_j_k=entropy),
        observations=points,checkpoint=model.checkpoint())


def evaluate(model,candidates,*,target,limits):
    """Return a sealed replayable decision packet, without advancing live model."""
    if type(model) is not HeatedHEMNetwork:raise ValueError('HeatedHEMNetwork required')
    candidates,target,limits=deepcopy((candidates,target,limits))
    with model._lock:
        admit(model,candidates,target,limits)
        checkpoint=model.checkpoint();request=deepcopy(model.request);scheme=model.scheme;source=signature()
    results=[branch(request,scheme,checkpoint,c,target,limits) for c in candidates]
    if signature()!=source:raise ValueError('Decision source changed')
    eligible=[r for r in results if r['status']=='eligible']
    selected=min(eligible,key=lambda r:(r['metrics']['tracking_error_k_s'],r['metrics']['absolute_commanded_energy_j'],r['id']))['id'] if eligible else None
    payload=dict(schema_version=1,source=source,base_checkpoint=checkpoint,candidates=candidates,target=target,limits=limits,
        results=results,selected_id=selected,ranking='Eligible only; endpoint-quadrature wall IAE, absolute commanded energy, id',
        constraint_scope='Observation boundaries only; no between-step guarantee',validated=False,plant_referenced=False)
    if len(canonical(payload).encode())>64*1024**2:raise ValueError('Decision output size budget')
    return seal(payload)


def commit(model,decision):
    """Replay the entire decision, reject stale/tampered packets, then commit.

    Re-evaluation deliberately favors auditability over planner throughput.
    The final optimistic check and model restore share its transaction lock.
    """
    if type(model) is not HeatedHEMNetwork:raise ValueError('HeatedHEMNetwork required')
    payload=unseal(deepcopy(decision))
    with model._lock:
        if payload['source']!=signature() or canonical(model.checkpoint())!=canonical(payload['base_checkpoint']):
            raise ValueError('Stale decision source or live checkpoint')
    replay=evaluate(model,payload['candidates'],target=payload['target'],limits=payload['limits'])
    if canonical(replay)!=canonical(decision):raise ValueError('Decision replay differs')
    if payload['selected_id'] is None:raise ValueError('No eligible candidate')
    chosen=next(r for r in payload['results'] if r['id']==payload['selected_id'])
    with model._lock:
        if canonical(model.checkpoint())!=canonical(payload['base_checkpoint']):raise ValueError('Live model advanced during planning')
        model.restore(chosen['checkpoint'])
        return model.snapshot()
