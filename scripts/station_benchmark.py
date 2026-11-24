"""Public, finite R0 thermal/shaft benchmark; never a full-PWR validation.

A port assembly/exact stepper is compared with separately written component
balances and three IVP algorithms. The independent equations share declared
physical parameters; they are not an independent plant dataset.
"""
import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from atomic_model_station import StationConfig, StationResidual, AssembledStationSession
from atomic_model_station.model import STATE_NAMES, signature
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json, read_json
from atomic_model_workspace.common import digest

METHODS = ('assembled-exact', 'Radau', 'BDF', 'DOP853')
SCALES = np.array([1.,1.,1.,1.,1e6,1e6,1e6,1e6])


def contract():
    config = asdict(StationConfig())
    cases = [dict(name='nominal', segments=[dict(end_s=10., power_w=1e6, connected=True)]),
        dict(name='source-step', segments=[dict(end_s=2.,power_w=1e6,connected=True),
            dict(end_s=6.,power_w=8e5,connected=True),dict(end_s=10.,power_w=1e6,connected=True)]),
        dict(name='load-disconnect',segments=[dict(end_s=2.,power_w=1e6,connected=True),
            dict(end_s=6.,power_w=1e6,connected=False),dict(end_s=10.,power_w=1e6,connected=True)])]
    value = dict(format='station-r0-benchmark/v1',model_scope='synthetic-thermal-shaft-subsystem',
        config=config, cases=cases, methods=list(METHODS), sample_spacing_s=.25,
        initial_values=[*StationResidual(StationConfig(**config)).steady()[0].tolist(),0.,0.,0.],
        sample_times_s=np.linspace(0.,10.,41).tolist(),
        state_names=[*STATE_NAMES,'source_energy_j','electrical_export_j','cooling_rejection_j'],
        units=['K','K','K','K','J','J','J','J'],normalization_scales=SCALES.tolist(),
        rtol=1e-8, normalized_atol=1e-9, maximum_step_s=.25,
        maximum_rhs_calls=20000, maximum_wall_s_per_attempt=60.,
        maximum_normalized_error=1e-4, maximum_relative_energy_defect=1e-8,
        error_definition='max over every sample and output of abs(candidate-reference)/normalization_scale',
        conservation_definition='abs(delta stored-source+export+cooling)/max(1,abs(source)+abs(export)+abs(cooling))',
        command_convention='piecewise constant; switch at exact segment boundary; no state reset',
        boundary_conditions='prescribed source; fixed ambient; controlled resistive generator load',
        physical_domain='existing R0 sampled temperature/rotor/Carnot checks at every output',
        source=signature(), recipe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        solver_documentation='https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html',
        plant_validation=False, full_station_comparison=False, agent_comparison=False)
    return dict(value, sha256=digest(value))


def component_rates(config, state, power, connected):
    """Independent explicit power balances, including three boundary integrals."""
    c=config; p,w,s,cold,k=state[:5]
    qpw=c.primary_ua_w_k*(p-w)
    qws=c.secondary_ua_w_k*(w-s)
    qeng=c.engine_ua_w_k*(s-cold)
    shaft=c.engine_efficiency*qeng
    omega2=2*k/c.rotor_inertia_kg_m2
    bearing=c.bearing_drag_nm_s_rad*omega2
    generator=(c.generator_drag_nm_s_rad if connected else 0.)*omega2
    export=c.generator_efficiency*generator
    rejected=c.cooling_ua_w_k*(cold-c.ambient_temperature_k)
    return np.array([(power-qpw)/c.primary_capacity_j_k,
        (qpw-qws)/c.wall_capacity_j_k,(qws-qeng)/c.secondary_capacity_j_k,
        (qeng-shaft+bearing+generator-export-rejected)/c.cold_capacity_j_k,
        shaft-bearing-generator,power,export,rejected])


def _matrix(c,power,connected):
    a,b=StationResidual(c).operators(power,connected)
    m=np.zeros((9,9));m[:5,:5]=a;m[:5,8]=b
    m[5,8]=power
    m[6,4]=c.generator_efficiency*2*(c.generator_drag_nm_s_rad if connected else 0.)/c.rotor_inertia_kg_m2
    m[7,3]=c.cooling_ua_w_k;m[7,8]=-c.cooling_ua_w_k*c.ambient_temperature_k
    return m


def _grid(start,end,spacing):
    n=round((end-start)/spacing)
    return np.linspace(start,end,n+1)[1:]


def _sample_native(session):
    s=session.snapshot()
    return np.array([*s['state'],s['source_energy_j'],s['electrical_export_j'],s['cooling_rejection_j']])


def _reference(c,case,spacing):
    origin=np.r_[StationResidual(c).steady()[0],0.,0.,0.]
    rows=[origin.tolist()];times=[0.];start=0.
    for segment in case['segments']:
        matrix=_matrix(c,segment['power_w'],segment['connected'])
        # Center all states at this segment's start to avoid subtracting large
        # equilibrium inventories during matrix-exponential propagation.
        shifted=matrix.copy();shifted[:8,8]=matrix[:8,:8]@origin+matrix[:8,8]
        for t in _grid(start,segment['end_s'],spacing):
            delta=(expm(shifted*(t-start))@np.r_[np.zeros(8),1.])[:8]
            rows.append((origin+delta).tolist());times.append(float(t))
        origin=np.asarray(rows[-1]);start=segment['end_s']
    return np.asarray(times),np.asarray(rows)


def assess_submission(contract_sha256, case_name, times_s, values):
    """Score external SI-unit samples; execution cost/provenance is not certified.

    A prefix is retained and scored but cannot pass the complete-grid gate.
    Missing, reordered or shifted samples cannot be interpolated into success.
    """
    spec=contract()
    if type(contract_sha256) is not str or contract_sha256!=spec['sha256']:
        raise ValueError('Exact current benchmark contract required')
    case=next((c for c in spec['cases'] if c['name']==case_name),None)
    if case is None:raise ValueError('Declared benchmark case required')
    if type(times_s) is not list or type(values) is not list or not times_s or len(times_s)!=len(values):
        raise ValueError('Nonempty aligned benchmark samples required')
    if len(times_s)>41 or any(type(t) not in (int,float) or not np.isfinite(t) for t in times_s):
        raise ValueError('Bounded finite sample epochs required')
    if any(type(row) is not list or len(row)!=8 or
           any(type(v) not in (int,float) or not np.isfinite(v) for v in row) for row in values):
        raise ValueError('Eight finite SI output quantities required')
    c=StationConfig(**spec['config']);model=StationResidual(c)
    times,reference=_reference(c,case,spec['sample_spacing_s'])
    if times_s!=times[:len(times_s)].tolist():raise ValueError('Exact consecutive benchmark grid required')
    actual=np.asarray(values,dtype=float)
    for row in actual:model.validate(row[:5])
    if any(actual[0,5:]!=0):raise ValueError('Zero initial boundary integrals required')
    error=float(np.max(abs((actual-reference[:len(values)])/SCALES)))
    weights=np.array([c.primary_capacity_j_k,c.wall_capacity_j_k,c.secondary_capacity_j_k,c.cold_capacity_j_k,1.])
    delta=(actual[:,:5]-reference[0,:5])@weights
    residual=delta-actual[:,5]+actual[:,6]+actual[:,7]
    scale=np.maximum(1.,np.sum(abs(actual[:,5:]),axis=1))
    defect=float(np.max(abs(residual)/scale))
    return dict(full_grid=len(values)==len(reference),
        state_and_integrals=error<=spec['maximum_normalized_error'],
        energy_budget=defect<=spec['maximum_relative_energy_defect'],
        maximum_normalized_error=error,maximum_relative_energy_defect=defect,
        external_execution_cost_verified=False)


def run_attempt(spec,case,method):
    if method not in METHODS:raise ValueError('Declared benchmark method required')
    c=StationConfig(**spec['config']);model=StationResidual(c)
    _,reference=_reference(c,case,spec['sample_spacing_s'])
    started=time.perf_counter();rows=[reference[0].tolist()];epochs=[0.]
    calls=nfev=njev=nlu=0;failure=None;start=0.;session=None
    if method=='assembled-exact':session=AssembledStationSession(c)
    try:
        for segment in case['segments']:
            grid=_grid(start,segment['end_s'],spec['sample_spacing_s'])
            power,connected=segment['power_w'],segment['connected']
            if session is not None:
                for t in grid:
                    if time.perf_counter()-started>spec['maximum_wall_s_per_attempt']:
                        raise RuntimeError('wall_budget')
                    session.advance(float(t-epochs[-1]),source_power_w=power,generator_connected=connected)
                    rows.append(_sample_native(session).tolist());epochs.append(float(t))
            else:
                def rhs(t,z):
                    nonlocal calls
                    calls+=1
                    if calls>spec['maximum_rhs_calls']:raise RuntimeError('rhs_budget')
                    if time.perf_counter()-started>spec['maximum_wall_s_per_attempt']:raise RuntimeError('wall_budget')
                    return component_rates(c,z*SCALES,power,connected)/SCALES
                solution=solve_ivp(rhs,(start,segment['end_s']),np.asarray(rows[-1])/SCALES,
                    method=method,t_eval=grid,rtol=spec['rtol'],atol=spec['normalized_atol'],
                    max_step=spec['maximum_step_s'])
                nfev+=solution.nfev;njev+=solution.njev;nlu+=solution.nlu
                for t,z in zip(solution.t,solution.y.T):
                    model.validate(z[:5]*SCALES[:5])
                    rows.append((z*SCALES).tolist());epochs.append(float(t))
                if not solution.success:raise RuntimeError(solution.message)
            start=segment['end_s']
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error))
    wall=time.perf_counter()-started
    assessed=assess_submission(spec['sha256'],case['name'],epochs,rows)
    checks=dict(full_grid=failure is None and assessed['full_grid'],
        state_and_integrals=assessed['state_and_integrals'],energy_budget=assessed['energy_budget'],
        work_budget=calls<=spec['maximum_rhs_calls'],wall_budget=wall<=spec['maximum_wall_s_per_attempt'])
    return dict(case=case['name'],method=method,checks=checks,passed=all(checks.values()),
        failure=failure,times_s=epochs,values=rows,maximum_normalized_error=assessed['maximum_normalized_error'],
        maximum_relative_energy_defect=assessed['maximum_relative_energy_defect'],rhs_calls=calls,nfev=nfev,njev=njev,nlu=nlu,
        wall_s=wall,wall_per_simulated_s=wall/epochs[-1] if epochs[-1] else None)


def comparison_matrix(spec,attempts):
    """Evidence by capability; no unsupported whole-simulator ranking."""
    expected={(c['name'],m) for c in spec['cases'] for m in spec['methods']}
    actual=[(a['case'],a['method']) for a in attempts]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('Every case/method attempt must be retained exactly once')
    methods={m:dict(attempted=sum(a['method']==m for a in attempts),
        passed=sum(a['passed'] for a in attempts if a['method']==m),
        maximum_normalized_error=max(a['maximum_normalized_error'] for a in attempts if a['method']==m),
        total_wall_s=sum(a['wall_s'] for a in attempts if a['method']==m),
        total_rhs_calls=sum(a['rhs_calls'] for a in attempts if a['method']==m)) for m in spec['methods']}
    return dict(scope=spec['model_scope'],contract_sha256=spec['sha256'],methods=methods,
        dimensions=dict(accuracy='measured against affine matrix exponential on all eight outputs',
            conservation='integrated source/export/cooling and stored thermal/rotor energy',
            robustness='finite three-case matrix only; all rejected attempts retained',
            speed='single shared-host measurements; setup/reference/scoring excluded; no universal speed claim',
            control='prescribed event schedule only; no optimized closed-loop policy comparison',
            diagnosis='not exercised by this subsystem benchmark',
            portability='requires separate installed-wheel replay',
            provenance='synthetic parameters, two equation implementations, shared SciPy dependency'),
        gaps=['full-station independent model','real measurements and uncertainties',
              'long coastdown and phase transitions','agent industrial utility','external simulator timings'],
        whole_simulator_superiority_established=False)


def assess_file(submission,output):
    """Public interchange without trusting a submitted success flag or timing."""
    data=read_json(submission)
    keys={'format','contract_sha256','case','simulator','times_s','values'}
    if type(data) is not dict or set(data)!=keys or data['format']!='station-r0-submission/v1':
        raise ValueError('Exact station-r0-submission/v1 schema required')
    if type(data['simulator']) is not str or not data['simulator'].strip() or len(data['simulator'])>128:
        raise ValueError('Bounded declared simulator identity required')
    assessment=assess_submission(data['contract_sha256'],data['case'],data['times_s'],data['values'])
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True)
    report=dict(contract_sha256=data['contract_sha256'],case=data['case'],simulator=data['simulator'],
        simulator_identity_verified=False,submission_sha256=hashlib.sha256(Path(submission).read_bytes()).hexdigest(),
        assessment=assessment,passed=all(assessment[k] for k in ('full_grid','state_and_integrals','energy_budget')))
    atomic_json(out/'assessment.json',seal(report))
    return report


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);spec=contract()
    atomic_json(out/'contract.json',seal(spec))
    attempts=[]
    for case in spec['cases']:
        for method in spec['methods']:
            attempts.append(run_attempt(spec,case,method))
            atomic_json(out/'attempts.json',seal(dict(contract_sha256=spec['sha256'],attempts=attempts)))
    report=dict(contract_sha256=spec['sha256'],source_unchanged=spec==contract(),
        attempted=len(attempts),passed_attempts=sum(a['passed'] for a in attempts),
        failures=[dict(case=a['case'],method=a['method'],failure=a['failure'],checks=a['checks']) for a in attempts if not a['passed']],
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            machine=platform.machine(),threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
            exclusive_host=False),full_station_comparison=False,experimental_validation=False,
        independent_equation_implementation=True,independent_physical_data=False)
    report['passed']=report['source_unchanged'] and report['passed_attempts']==report['attempted']==12
    atomic_json(out/'report.json',seal(report))
    atomic_json(out/'comparison.json',seal(comparison_matrix(spec,attempts)))
    scientific=[{k:v for k,v in a.items() if k not in ('wall_s','wall_per_simulated_s')} for a in attempts]
    atomic_json(out/'scientific.json',seal(dict(contract=spec,attempts=scientific)))
    if not report['passed']:raise ValueError('Benchmark failed; all attempts retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--submission',type=Path)
    args=parser.parse_args()
    result=assess_file(args.submission,args.output) if args.submission is not None else run(args.output)
    print(result)
    if not result['passed']:raise SystemExit(1)
