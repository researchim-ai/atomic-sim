"""Uniform finite-capacity wall/fluid feedback; exact zero-flow graph reduction."""
from copy import deepcopy
from atomic_model_hem_qualified import eos
from atomic_model_hem_qualified.model import QualifiedHEMNetwork
from atomic_model_hem_qualified.qualification import CRITERIA
from scripts.verify_phase_cycles import CASES,request_for

WALL_SPECIFIC_CAPACITY=1e4
WALL_SPECIFIC_CONDUCTANCE=1e3
DURATIONS=(.0025,)*4


def wall_request(case, *, cells=2, cap=2e-5, diameter_m=.05):
    if case not in CASES:raise ValueError('Declared wall cycle only')
    if diameter_m not in (.05,.08):raise ValueError('Declared wall geometry only')
    request=request_for(case,cells,cap)
    for edge in request['model']['edges']:edge['diameter_m']=diameter_m
    # The largest node volume is needed to keep all vapor-wall capacities
    # inside the existing100..1e7 J/K scalar parser domain.
    for node in request['model']['nodes']:node['volume_m3']=.02
    initial_model=QualifiedHEMNetwork(request)
    cells=initial_model.snapshot()['cells']
    _,offset=CASES[case]
    request['model']['walls']=[]
    for index,cell in enumerate(cells):
        tw=cell['temperature_k']+(10. if offset<0 else -10.)
        request['model']['walls'].append(dict(id=f'wall{index}',cell=cell['id'],
            capacity_j_k=cell['mass_kg']*WALL_SPECIFIC_CAPACITY,
            conductance_w_k=cell['mass_kg']*WALL_SPECIFIC_CONDUCTANCE,
            initial_temperature_k=tw,ambient_temperature_k=tw,ambient_ua_w_k=0.,heat_power_w=0.))
    return request


def execute(case, *, cells=2, cap=2e-5, diameter_m=.05):
    request=wall_request(case,cells=cells,cap=cap,diameter_m=diameter_m)
    model=QualifiedHEMNetwork(request)
    observations=[model.snapshot()];checkpoint=None
    for i,duration in enumerate(DURATIONS):
        observations.append(model.advance(duration))
        if i==1:checkpoint=model.checkpoint()
    restored=QualifiedHEMNetwork(request);restored.restore(checkpoint)
    for duration in DURATIONS[2:]:last=restored.advance(duration)
    before,after=observations[0],observations[-1]
    checks=dict(energy_balance=all(abs(s['global_ledger']['energy_residual_j'])<=CRITERIA['energy_balance_j'] for s in observations),
                mass_balance=all(abs(s['global_ledger']['mass_residual_kg'])<=CRITERIA['mass_balance_kg'] for s in observations),
                closed=all(s['global_ledger'][k]==0. for s in observations for k in ('external_fluid_heat_j','external_wall_heat_j','external_entropy_j_k')),
                entropy=all(s['global_ledger']['numerical_entropy_residual_j_k']>=CRITERIA['numerical_entropy_lower_j_k'] for s in observations),
                thermal_production=all(s['global_ledger']['thermal_entropy_production_j_k']>=0. for s in observations),
                phase_crossing=all(a['phase']!=b['phase'] for a,b in zip(before['cells'],after['cells'],strict=True)),
                wall_moves_towards_fluid=all(abs(w['temperature_k']-c['temperature_k'])<abs(old_w['temperature_k']-old_c['temperature_k']) for w,c,old_w,old_c in zip(after['walls'],after['cells'],before['walls'],before['cells'],strict=True)),
                restart_exact=last==after and restored.checkpoint()==model.checkpoint())
    return dict(request=request,observations=observations,checkpoint=checkpoint,
                record=dict(case=case,passed=all(checks.values()),checks=checks))
