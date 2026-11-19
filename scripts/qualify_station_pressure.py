"""Manufactured-solution audit separating native FD and compatible pressure maps."""
import argparse,json
from pathlib import Path
from importlib.resources import files
import numpy as np
from atomic_model_station_pressure import pressure_operators,solve_pressure,manufactured_pressure
from atomic_model_station_pressure import operators
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import atomic_json
from scripts import replay_station_pressure_operators as producer


def identity():
    return dict(driver=sha(__file__),operators=sha(operators.__file__),producer=sha(producer.__file__),
        resource=sha(files('atomic_model_station_pressure').joinpath('data/native_operators.json')))


def native_matrices(row,m):
    r=np.asarray(row['radius']);n=len(r);first=np.zeros((n,n));second=np.zeros((n,n));seen=set()
    expected={(i,j) for i in range(1,n+1) for j in range(max(-3,i-4),min(n,i+4)+1)}
    for i,j,a,b in row['bands']:
        if i!=int(i) or j!=int(j) or (i,j) not in expected or (i,j) in seen or not np.isfinite(a+b):raise ValueError('Complete unique finite native bands required')
        i,j=int(i),int(j);seen.add((i,j));column=j-1 if j>0 else -j;sign=1 if j>0 else (-1)**m
        first[i-1,column]+=sign*a;second[i-1,column]+=sign*b
    if seen!=expected:raise ValueError('Incomplete native operator export')
    return first,second


def run(output,reference=None):
    out=Path(output).absolute();source=identity()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    fixture=unseal(json.loads(files('atomic_model_station_pressure').joinpath('data/native_operators.json').read_text()))
    if fixture['harness_sha256']!=source['producer'] or set(fixture['records'])!={'32','64','96'}:raise ValueError('Pinned complete native operator export required')
    criteria=dict(relative_native_coefficients=1e-9,relative_compatible_interior_residual=1e-9,pressure_recovery=1e-7)
    out.mkdir(parents=True);rows={};passed=True
    for n,row in fixture['records'].items():
        for m in (0,1,2,4):
            k=0 if m==0 else 3;op=pressure_operators(row['radius'],k,m,.628319)
            first,second=native_matrices(row,m)
            errors=[float(np.max(abs(a-b))/max(1,float(np.max(abs(b))))) for a,b in [(op.radial_first,first),(op.radial_second,second)]]
            p,dr,exact=manufactured_pressure(row['radius'],m,k,.628319)
            rhs=op.laplacian_composed@p;boundary=dict(wall_derivative=float((op.radial_first@p)[-1])) if k or m else dict(wall_gauge=float(p[-1]))
            old,old_audit=solve_pressure(op,rhs,composed=False,**boundary)
            new,new_audit=solve_pressure(op,rhs,composed=True,**boundary)
            scale=max(1,float(np.max(abs(rhs))));compatible_error=new_audit['interior_div_grad_residual']/scale
            recovery=float(np.max(abs(new-p)));native_ok=max(errors)<criteria['relative_native_coefficients']
            case_ok=bool(native_ok and compatible_error<criteria['relative_compatible_interior_residual'] and recovery<criteria['pressure_recovery']);passed=passed and case_ok
            rows[f'n{n}-m{m}']=dict(passed=case_ok,n=int(n),m=m,k=k,relative_native_coefficient_errors=errors,
                fd_laplacian_error=float(np.max(abs(op.laplacian_fd@p-exact))),composed_laplacian_error=float(np.max(abs(rhs-exact))),
                operator_mismatch_action=float(np.max(abs((op.laplacian_composed-op.laplacian_fd)@p))),
                original_fd=old_audit,compatible=new_audit,relative_compatible_interior_residual=compatible_error,
                original_pressure_error=float(np.max(abs(old-p))),compatible_pressure_error=recovery,
                maximum_compatible_wall_gradient=float(np.max(abs((op.gradient@new).reshape(3,-1)[:,-1]))))
    report=dict(passed=bool(passed),source=source,criteria=criteria,cases=rows,native_meshes=3,manufactured_cases=len(rows),
        maximum_relative_native_coefficient_error=max(max(v['relative_native_coefficient_errors']) for v in rows.values()),
        maximum_relative_compatible_interior_residual=max(v['relative_compatible_interior_residual'] for v in rows.values()),
        maximum_pressure_recovery_error=max(v['compatible_pressure_error'] for v in rows.values()),
        native_fd_equals_composed=False,external_time_integrator_modified=False,no_slip_preservation_qualified=False,
        axis_regularity_qualified=False,time_integrator_qualified=False,dynamic_continuity_qualified=False,
        station_usable=False,validated=False,closed_tasks=[],
        scope='Native coefficient audit and manufactured scalar pressure solves only; boundary row and full velocity dynamics remain separate')
    if source!=identity():raise ValueError('Pressure audit source changed')
    atomic_json(out/'report.json',seal(report))
    if not passed:raise ValueError('Pressure manufactured qualification failed; evidence retained')
    if reference and (Path(reference)/'report.json').read_bytes()!=(out/'report.json').read_bytes():raise ValueError('Pressure installed repeat differs')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();print(run(a.output,a.reference)['passed'])

if __name__=='__main__':main()
