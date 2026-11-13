"""R0 scaffold verification: no fluid/steam/NPP validation claim."""
import argparse
from dataclasses import asdict
from pathlib import Path
import math

import numpy as np
from scipy.integrate import solve_ivp
from atomic_model_station import StationConfig,StationSession,passport,signature
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json

CASES={'nominal':[(60.,1e6,True)]*60,
       'heat_change':[(30.,1e6,True),(60.,1.05e6,True),(60.,1e6,True)],
       'load_trip':[(10.,1e6,True),(5.,1e6,False),(30.,1e6,True)]}


def reference_rhs(c,power,connected):
    # Independent power equations; does not read StationResidual operators.
    def rhs(t,z):
        p,w,s,cold,k=z[:5]
        q1=c.primary_ua_w_k*(p-w);q2=c.secondary_ua_w_k*(w-s)
        heat=c.engine_ua_w_k*(s-cold);mechanical=c.engine_efficiency*heat
        omega2=2*k/c.rotor_inertia_kg_m2
        generator=(c.generator_drag_nm_s_rad if connected else 0.)*omega2
        bearing=c.bearing_drag_nm_s_rad*omega2;export=c.generator_efficiency*generator
        cooling=c.cooling_ua_w_k*(cold-c.ambient_temperature_k)
        return [(power-q1)/c.primary_capacity_j_k,(q1-q2)/c.wall_capacity_j_k,
                (q2-heat)/c.secondary_capacity_j_k,
                ((1-c.engine_efficiency)*heat+bearing+generator-export-cooling)/c.cold_capacity_j_k,
                mechanical-generator-bearing,export,cooling]
    return rhs


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    source=signature();driver=sha(__file__);config=StationConfig()
    output.mkdir(parents=True,exist_ok=False)
    criteria=dict(reference_relative=1e-9,energy_relative=1e-10,energy_absolute_j=1e-5)
    atomic_json(output/'protocol.json',seal(dict(source=source,driver_sha256=driver,passport=passport(),
        config=asdict(config),parameters=config.registry(),cases=CASES,criteria=criteria,
        scope='R0 thermal-mechanical prototype, not station R1 or fluid/steam assembly')))
    rows=[]
    for name,commands in CASES.items():
        session=StationSession();ref=np.r_[session.snapshot()['state'],0.,0.];trace=[];errors=[]
        before_last=None
        for index,(duration,power,connected) in enumerate(commands):
            if index==len(commands)-1:before_last=session.checkpoint()
            solution=solve_ivp(reference_rhs(config,power,connected),(0.,duration),ref,method='DOP853',rtol=1e-12,atol=1e-8)
            if not solution.success:raise ValueError('Independent reference failed')
            ref=solution.y[:,-1];actual=session.advance(duration,source_power_w=power,generator_connected=connected)
            values=np.r_[actual['state'],actual['electrical_export_j'],actual['cooling_rejection_j']]
            errors.append(float(np.max(abs(values-ref)/np.maximum(abs(ref),1.))))
            trace.append(actual)
        saved=session.checkpoint();restored=StationSession.restore(before_last)
        duration,power,connected=commands[-1]
        exact=(restored.advance(duration,source_power_w=power,generator_connected=connected)==trace[-1]
               and StationSession.restore(saved).snapshot()==session.snapshot())
        row=dict(case=name,trace=trace,max_reference_relative=max(errors),restart_exact=exact,
                 completed=len(trace)==len(commands),checkpoint=saved)
        atomic_json(output/f'{name}.json',seal(row));rows.append(row)
    unchanged=source==signature() and driver==sha(__file__)
    report=dict(completed=True,passed=unchanged and all(r['completed'] and r['restart_exact'] and r['max_reference_relative']<criteria['reference_relative'] for r in rows),
        source_unchanged=unchanged,cases=[dict(case=r['case'],duration_s=r['trace'][-1]['time_s'],
            max_reference_relative=r['max_reference_relative'],max_energy_defect_j=max(abs(x['step_energy_residual_j']) for x in r['trace']),restart_exact=r['restart_exact']) for r in rows],
        package_a_complete=False,reactor_connected=False,steam_cycle_connected=False,
        validated=False,plant_referenced=False,limitations=passport()['unsupported'])
    atomic_json(output/'report.json',seal(report));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.output)
