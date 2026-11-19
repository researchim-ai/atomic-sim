"""Constrained dimensional steady equations for the current point-core topology.

The solve changes declared time-zero states and actuator settings, not equipment
coefficients. It is never an operating-state projection or a warmup shortcut.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import numpy as np
from core.status import DomainError, SolverError
from atomic_model_multiphysics.provenance import sha


def signature():
    return dict(files={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},numpy=np.__version__)


@dataclass(frozen=True)
class EquilibriumConfig:
    power_w:float=3.15e9
    pressure_pa:float=6.3e6
    speed_rad_s:float=157.

    def __post_init__(self):
        for k,low,high in [('power_w',1.6e9,3.3e9),('pressure_pa',5.3e6,6.8e6),('speed_rad_s',150.,160.)]:
            x=getattr(self,k)
            if type(x) not in (int,float) or not math.isfinite(x) or not low<=x<=high:
                raise ValueError('Equilibrium target domain: '+k)


def solve(owner,config=None):
    from atomic_model_station.faults import PWRFaultSession
    c=config if config is not None else EquilibriumConfig()
    if type(c) is not EquilibriumConfig or type(owner) is not PWRFaultSession:
        raise ValueError('Point-station equilibrium requires EquilibriumConfig/PWRFaultSession')
    if owner.time!=0 or len(owner.history) or owner.events or owner.active_faults or owner.scram_latched or owner.turbine_tripped or getattr(owner,'station_equilibrium',None) is not None:
        raise ValueError('Equilibrium initialization requires a fresh healthy operating epoch')
    if owner.core_config.kind!='point' or owner.base_size!=43 or len(owner.state)!=78:
        raise ValueError('Equilibrium supports only the declared 78-coordinate point-core topology')
    original=owner.checkpoint();work=deepcopy(owner);y0=work.state.copy();work._rhs(0.,y0)
    if any(y0[i]!=0. for i in [*range(work.ledger_index,work.ledger_index+4),*range(work.power_index+1,work.power_index+4)]):
        raise ValueError('Operating ledgers cannot be reinitialized')
    source=signature();parameters=deepcopy(work.parameter_registry());initial_energy=work._stored()
    names=work.plant.network.volume_names();m={n:13+3*i for i,n in enumerate(names)}
    primary=['core','hot_leg','sg_primary','cold_leg','pressurizer'];loop=primary[:-1]
    primary_mass=sum(y0[m[n]] for n in primary);boron=sum(y0[m[n]+2] for n in loop)
    press=work.plant.network.volumes['pressurizer'];entropy=press.mass_kg*press.thermo.s_j_kg_k
    secondary=[m['sg_secondary'],work.base_size-3,work.feed_index,work.feed_index+2,work.feed_index+4,work.steam_index]
    secondary_mass=sum(y0[i] for i in secondary)
    q0=13+3*len(names);q={n:q0+i for i,n in enumerate(work.plant.network.junction_names())}
    fixed={*range(7),11,q['surge'],q['steam'],q['spray'],q['dump'],work.primary_index+1,
           work.steam_index+2,work.steam_index+3,*range(work.feed_index+2,work.feed_index+6),work.power_index,
           m['pressurizer']+2,m['sg_secondary']+2,*range(work.ledger_index,work.ledger_index+4),
           *range(work.power_index+1,work.power_index+4)}
    retained=np.array([i for i in range(len(y0)) if i not in fixed]);equations=retained.copy()
    equations[list(equations).index(m['pressurizer'])]=q['surge']
    if len(retained)!=49:raise ValueError('Unexpected steady topology dimension')
    scale=np.maximum(abs(y0),1.);tolerance=np.full(len(y0),1e-8)
    tolerance[7:10]=1.
    for i in m.values():tolerance[i:i+3]=[1e-6,1.,1e-10]
    tolerance[work.base_size-3:work.base_size-1]=[1e-6,1.]
    tolerance[10]=1/work.plant.fuel_capacity;tolerance[work.base_size-1]=1/work.plant.sg_wall_capacity
    tolerance[work.base_size]=1.;tolerance[work.base_size+1]=1/work.config.clad_capacity_j_k
    for i in (work.cooling_index,work.cooling_index+1,work.feed_index+1,work.feed_index+3,work.feed_index+5,work.steam_index+1):tolerance[i]=1.
    for i in (work.cooling_index+2,work.feed_index,work.feed_index+2,work.feed_index+4,work.feed_index+6,work.steam_index):tolerance[i]=1e-6
    for name,index in q.items():
        edge=work.plant.network.junctions[name]
        tolerance[index]=.01*edge.area_m2/edge.length_m if edge.flow_law=='momentum' else 1e-6
    relative_power=c.power_w/work.plant.kinetics.power_nominal_w
    if not .5<=relative_power<=1.1:raise ValueError('Steady power outside declared relative-power scope')
    calls=0;domain_rejections=0
    domain_errors=(DomainError,SolverError,ValueError,ArithmeticError)

    def state(z):
        y=y0.copy();y[retained]=z*scale[retained]
        return work._reconstruct_state(y)

    def residual(z):
        nonlocal calls
        calls+=1
        if calls>10000:raise SolverError('Steady initialization evaluation budget')
        y=state(z)
        if any(not 0<=y[i]<=1 for i in (12,work.steam_index+4,work.bypass_index)):
            raise DomainError('Steady actuator stroke')
        work.controls.update(rod_target=float(y[12]),steam_admission=float(y[work.steam_index+4]),steam_bypass=float(y[work.bypass_index]))
        rate=work._rhs(0.,y);work._admit(y);r=rate/tolerance
        r[12]=(y[work.core_index]-relative_power)/1e-10
        r[m['core']]=(sum(y[m[n]] for n in primary)-primary_mass)/max(1e-6,primary_mass*1e-10)
        r[m['core']+2]=(sum(y[m[n]+2] for n in loop)-boron)/max(1e-10,boron*1e-10)
        press=work.plant.network.volumes['pressurizer']
        r[m['pressurizer']+1]=(press.mass_kg*press.thermo.s_j_kg_k-entropy)/(abs(entropy)*1e-10)
        r[work.base_size-3]=(sum(y[i] for i in secondary)-secondary_mass)/max(1e-6,secondary_mass*1e-10)
        target_energy=.5*work.config.inertia_kg_m2*c.speed_rad_s**2
        r[work.steam_index+4]=(y[work.base_size]-target_energy)/max(target_energy*1e-10,1.)
        r[work.bypass_index]=(work.header.thermo.p_pa-c.pressure_pa)/1.
        value=r[equations]
        if not np.isfinite(value).all():raise DomainError('Nonfinite equilibrium residual')
        return value

    def jacobian(z,r):
        nonlocal domain_rejections
        columns=[]
        for j in range(len(z)):
            h=1e-6*max(abs(z[j]),1.);delta=np.zeros_like(z);delta[j]=h
            try:col=(residual(z+delta)-r)/h
            except domain_errors:
                domain_rejections+=1
                col=(r-residual(z-delta))/h
            columns.append(col)
        return np.column_stack(columns)

    seed=y0.copy();seed[work.bypass_index]=.5;seed[work.steam_index+4]=.95
    z=seed[retained]/scale[retained];history=[]
    for iteration in range(60):
        r=residual(z);defect=float(max(abs(r)));history.append(defect)
        if defect<=1.:break
        matrix=jacobian(z,r);rows=np.maximum(np.linalg.norm(matrix,axis=1),1e-30)
        a=matrix/rows[:,None];cols=np.maximum(np.linalg.norm(a,axis=0),1e-30)
        delta=np.linalg.lstsq(a/cols,-r/rows,rcond=1e-12)[0]/cols
        maximum=float(max(abs(delta)))
        if maximum==0:raise SolverError('Steady Newton direction vanished')
        alpha=min(1.,.15/maximum);old=float(np.linalg.norm(r/rows));accepted=False
        for _ in range(22):
            try:
                trial=residual(z+alpha*delta)
                if np.linalg.norm(trial/rows)<old:
                    z+=alpha*delta;accepted=True;break
            except domain_errors:domain_rejections+=1
            alpha*=.5
        if not accepted:raise SolverError('No domain-admitted decreasing steady Newton step')
    else:raise SolverError('Steady initialization iteration budget')
    r=residual(z);matrix=jacobian(z,r);rows=np.maximum(np.linalg.norm(matrix,axis=1),1e-30)
    matrix=matrix/rows[:,None];cols=np.maximum(np.linalg.norm(matrix,axis=0),1e-30)
    singular=np.linalg.svd(matrix/cols,compute_uv=False);rank=int(np.count_nonzero(singular>singular[0]*1e-9))
    if rank!=len(retained):raise SolverError('Steady constrained Jacobian is rank deficient')
    final=state(z);residual(z);work.state=final.copy();rate=work._rhs(0.,final);work._admit(final)
    if abs(float(work.plant.network.junctions['spray'].flow_target_kg_s))>1e-9:
        raise DomainError('Steady zero-spray branch is incompatible with pressurizer demand')
    # All independent dimensional equations, including those replaced by anchors,
    # are accepted again. No optimizer success flag substitutes for this audit.
    excluded={*range(work.ledger_index,work.ledger_index+4),*range(work.power_index+1,work.power_index+4),0,1,2,3,4,5,6,11}
    full=float(max(abs(rate[i])/tolerance[i] for i in range(len(final)) if i not in excluded))
    if full>1. or max(abs(r))>1. or abs(float(work.powers['external_net_w']))>50.:
        violations={i:float(rate[i]/tolerance[i]) for i in range(len(final)) if i not in excluded and abs(rate[i]/tolerance[i])>1.}
        raise SolverError('Whole-station dimensional steady residual failed: '+str(dict(violations=violations,constrained=float(max(abs(r))),external_net_w=float(work.powers['external_net_w']))))
    work.origin_energy=work._stored();snap=work.snapshot()
    if snap['mass_relative']>1e-10 or snap['boron_relative']>1e-10 or snap['energy_relative']>1e-10:
        raise SolverError('Steady conserved inventory audit failed')
    if work.parameter_registry()!=parameters or signature()!=source or owner.checkpoint()!=original:
        raise SolverError('Steady initialization changed parameters/source/caller')
    report=dict(source=source,config=asdict(c),time_s=0.,warmup_steps=0,iterations=len(history)-1,evaluations=calls,
        domain_rejections=domain_rejections,scaled_defect=float(max(abs(r))),full_dimensional_ratio=full,
        constrained_unknowns=len(retained),constrained_rank=rank,equilibrated_singular_values=singular.tolist(),
        convergence_history=history,external_net_w=float(work.powers['external_net_w']),
        initial_energy_change_j=float(work._stored()-initial_energy),parameters_changed=False,
        operating_state_projected=False,declared_initial_state_solved=True,
        constraints=['primary total mass','secondary/header/tank total mass','circulating boron',
            'pressurizer/secondary boron and isolated reserves fixed','pressurizer entropy',
            'fission power','header pressure','shaft speed','inactive spray/dump and zero surge branch'],
        solved_controls={k:float(work.controls[k]) for k in ('rod_target','steam_admission','steam_bypass')},
        validated=False,plant_referenced=False,stationary_hold_accepted=False)
    work.station_equilibrium=dict(config=asdict(c),source=source,report=report)
    return work,report
