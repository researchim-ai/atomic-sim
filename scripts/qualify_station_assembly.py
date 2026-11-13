"""Composed R0 graph, independent ODE verification and scheduled command replay."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from atomic_model_station import StationConfig,AssembledStationSession,build_r0,signature
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import qualify_station_r0 as reference


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    c=StationConfig();source=signature()
    drivers={str(Path(p).absolute()):sha(p) for p in (__file__,reference.__file__)}
    output.mkdir(parents=True,exist_ok=False)
    criteria=dict(reference_relative=1e-9)
    atomic_json(output/'protocol.json',seal(dict(source=source,drivers=drivers,config=c.registry(),cases=reference.CASES,
        initial_graph=build_r0(c,c.nominal_source_power_w,True).manifest(),criteria=criteria,
        scope='Affine energetic R0 assembly; not nonlinear DAE, fluid ports or NPP R1')))
    cases=[]
    for name,commands in reference.CASES.items():
        s=AssembledStationSession();ref=np.r_[s.snapshot()['state'],0.,0.];trace=[];errors=[];before_last=None
        for index,(duration,power,connected) in enumerate(commands):
            if index==len(commands)-1:before_last=s.checkpoint()
            solved=solve_ivp(reference.reference_rhs(c,power,connected),(0.,duration),ref,method='DOP853',rtol=1e-12,atol=1e-8)
            if not solved.success:raise ValueError('Reference integration failed')
            ref=solved.y[:,-1];row=s.advance(duration,source_power_w=power,generator_connected=connected)
            values=np.r_[row['state'],row['electrical_export_j'],row['cooling_rejection_j']]
            errors.append(float(np.max(abs(values-ref)/np.maximum(abs(ref),1.))));trace.append(row)
        restored=AssembledStationSession.restore(json.loads(json.dumps(before_last)))
        duration,power,connected=commands[-1]
        exact=restored.advance(duration,source_power_w=power,generator_connected=connected)==trace[-1]
        atomic_json(output/f'{name}.json',seal(dict(trace=trace,checkpoint=s.checkpoint(),reference_errors=errors,restart_exact=exact)))
        cases.append(dict(name=name,duration_s=trace[-1]['time_s'],max_reference_relative=max(errors),restart_exact=exact,
                          max_energy_defect_j=max(abs(x['step_energy_residual_j']) for x in trace)))
    events=[dict(at_s=0.,commands={'source_power_w':1.05e6}),dict(at_s=3.,commands={'generator_connected':False}),dict(at_s=5.,commands={'generator_connected':True})]
    scheduled=AssembledStationSession();receipt=scheduled.advance_schedule(10.,events)
    manual=AssembledStationSession();manual.advance(3.,source_power_w=1.05e6);manual.advance(2.,generator_connected=False);manual.advance(5.,generator_connected=True)
    schedule_exact=manual.checkpoint()==scheduled.checkpoint()
    atomic_json(output/'schedule.json',seal(dict(receipt=receipt,checkpoint=scheduled.checkpoint(),manual_exact=schedule_exact)))
    unchanged=source==signature() and all(sha(p)==h for p,h in drivers.items())
    report=dict(completed=True,passed=unchanged and schedule_exact and all(x['restart_exact'] and x['max_reference_relative']<criteria['reference_relative'] for x in cases),
        source_unchanged=unchanged,cases=cases,schedule_exact=schedule_exact,inventory_count=5,power_block_count=7,connection_count=16,
        package_a_complete=False,nonlinear_dae_supported=False,fluid_enthalpy_supported=False,reactor_connected=False,
        validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report));return report


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.output)
