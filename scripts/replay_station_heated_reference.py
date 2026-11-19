"""Pinned external Openpipeflow mean-only replay; no simulator physics imported.

Archive is supplied locally, never downloaded/executed implicitly by library
imports. Only parameters.f90 is changed. Original solver source is not bundled.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import numpy as np
from scipy.io import netcdf_file
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts.station_process import execute

ARCHIVE_SHA='4652325064ce4ecb2b351875d17bf1b3baa58f2f89529c365191bc9ad98bc269'
SOURCE='openpipeflow-hp-1.21/'


def inventory(archive):
    if sha(archive)!=ARCHIVE_SHA:raise ValueError('Pinned external archive checksum differs')
    result={}
    with tarfile.open(archive) as stream:
        members=stream.getmembers()
        if len(members)>500:raise ValueError('External member budget')
        total=0
        for m in members:
            name=Path(m.name)
            if m.name.rstrip('/')==SOURCE.rstrip('/') and m.isdir():continue
            if name.is_absolute() or '..' in name.parts or not m.name.startswith(SOURCE):raise ValueError('Unsafe external member')
            if m.isdir():continue
            if not m.isfile() or m.size>2000000:raise ValueError('Ordinary bounded external files required')
            total+=m.size
            if total>10000000:raise ValueError('External expansion budget')
            result[str(name.relative_to(SOURCE))]=stream.extractfile(m).read()
    if 'program/parameters.f90' not in result or '00COPYING.txt' not in result:raise ValueError('Incomplete external archive')
    return result


def parameters(raw,*,cells,c,boundary,step):
    if cells not in (32,64) or c not in (0,2,5) or boundary not in ('delta_temperature','heat_flux') or step not in (.1,.05,.01,.005):
        raise ValueError('Declared external replay case required')
    values=dict(i_N=str(cells),d_C=str(c)+'d0',b_const_temp='.true.' if boundary=='delta_temperature' else '.false.',
        d_minE3d='-1d0',i_save_rate1=str(round(2000/step)),i_save_rate2='10000',i_maxtstep=str(round(8000/step)),
        d_maxt='-1d0',d_timestep=str(step)+'d0',d_maxdt=str(step)+'d0',d_dterr='1d-11')
    text=raw.decode()
    for key,value in values.items():
        pattern=r'(?m)^(\s*[^!\n]*::\s*'+re.escape(key)+r'\s*=)[^\n]*$'
        text,count=re.subn(pattern,lambda m:m[1]+' '+value,text)
        if count!=1:raise ValueError('External parameter anchor differs: '+key)
    return text.encode(),values


def initial(path,n,c,step):
    with netcdf_file(path,'w',version=1) as f:
        for key,v in dict(t=0.,Re=5300.,Pr=.7,C=float(c),alpha=.628319).items():setattr(f,key,np.float64(v))
        f.createDimension('r',n);f.createDimension('H',1);f.createDimension('ReIm',2)
        f.createVariable('r','d',('r',))[:]=np.linspace(.001,1.,n)
        f.createVariable('dt','d',())[...]=step
        for name in ('Ur','Ut','Uz','T'):
            v=f.createVariable(name,'d',('ReIm','H','r'));v.K=np.int32(1);v.M=np.int32(1);v.Mp=np.int32(1);v[:]=0.


def read_state(path,n,c):
    with netcdf_file(path,'r',mmap=False) as f:
        r=f.variables['r'][:].copy();u=f.variables['Uz'][:].copy();t=f.variables['T'][:].copy()
        if u.shape!=(2,1,n) or t.shape!=u.shape:raise ValueError('Mean-only external state required')
        if f.C!=c or f.Re!=5300 or abs(f.Pr-.7)>1e-14:raise ValueError('External parameters differ')
        if np.any(u[1]) or np.any(t[1]):raise ValueError('Mean imaginary modes not zero')
        result=dict(time=float(f.t),radius=r.tolist(),velocity=(1-r*r+u[0,0]).tolist(),temperature=(r*r+t[0,0]).tolist())
    if not np.all(np.isfinite([result['velocity'],result['temperature']])) or abs(r[-1]-1)>5e-14 or np.any(np.diff(r)<=0):
        raise ValueError('Malformed external profile')
    return result


def run(output,archive,dependencies):
    out=Path(output).absolute();dependencies=Path(dependencies).absolute();archive=Path(archive).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    original=inventory(archive);source_hash=sha(__file__);dep_lib=dependencies/'usr/lib/x86_64-linux-gnu'
    module=dependencies/'usr/include/netcdf.mod';lib=dep_lib/'libnetcdff.so'
    if not module.is_file() or not lib.is_file():raise ValueError('Local netCDF Fortran module and library required')
    out.mkdir(parents=True);records={};work=[]
    # Flux control differentiates the temperature field at the wall and needs
    # a smaller time step. The refused dt0.1 run is not accepted evidence.
    cases=[(bc,c,n,.1 if bc=='delta_temperature' else .01) for bc in ('delta_temperature','heat_flux') for c in (0,2,5) for n in (32,64)]
    cases += [('delta_temperature',2,64,.05),('heat_flux',2,64,.005)]
    for bc,c,n,step in cases:
        name=f'{bc}-c{c}-n{n}-dt{step}';directory=out/name;directory.mkdir()
        for name2,raw in original.items():
            target=directory/name2;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(raw)
        changed,values=parameters(original['program/parameters.f90'],cells=n,c=c,boundary=bc,step=step)
        (directory/'program/parameters.f90').write_bytes(changed)
        expected={name2:hashlib.sha256(raw).hexdigest() for name2,raw in original.items()};expected['program/parameters.f90']=sha(directory/'program/parameters.f90')
        flags='-ffree-line-length-none -x f95-cpp-input -c -O2 -I/usr/include -I'+str(module.parent)
        libs=f'-Wl,-rpath,{dep_lib} -L{dep_lib} cheby.o -l:libfftw3.so.3 -llapack -lnetcdff -lnetcdf'
        execute(['make','-j1','all','COMPFLAGS='+flags,'LIBS='+libs],cwd=directory,log=directory/'build.log',timeout_s=120,idle_s=60)
        initial(directory/'state.cdf.in',n,c,step)
        execute([str(directory/'main.out')],cwd=directory,log=directory/'run.log',timeout_s=180,idle_s=60)
        log=(directory/'run.log').read_text()
        if 'maxtstep reached!' not in log or '...done!' not in log:raise ValueError('External solver stopped before declared horizon')
        paths=sorted(directory.glob('state*.cdf.dat'));final=read_state(paths[-1],n,c);late=read_state(paths[-2],n,c)
        if abs(final['time']-8000)>1e-6 or abs(late['time']-6000)>1e-6:raise ValueError('Incomplete external final/late state')
        settling=float(np.max(np.abs(np.array([final['velocity'],final['temperature']])-np.array([late['velocity'],late['temperature']]))))
        if settling>1e-7:raise ValueError('External profile not settled; no steady claim')
        if expected!={name2:sha(directory/name2) for name2 in original}:raise ValueError('External source changed during build/run')
        records[name]=dict(boundary=bc,c=c,cells=n,step=step,final=final,late=late,late_maximum=settling,
            final_raw_sha256=sha(paths[-1]),late_raw_sha256=sha(paths[-2]),parameters=values,parameters_sha256=sha(directory/'program/parameters.f90'))
        work.append(dict(case=name,binary_sha256=sha(directory/'main.out'),initial_sha256=sha(directory/'state.cdf.in')))
        print('external heated mean',name,'settled',settling,flush=True)
    if source_hash!=sha(__file__):raise ValueError('External harness changed')
    fixture=dict(format='openpipeflow-hp-mean-only/v1',paper_doi='10.3390/math13020293',external_archive_sha256=ARCHIVE_SHA,
        external_archive_url='https://www.openpipeflow.org/images/5/53/Openpipeflow-hp-1.21.tgz',
        corrected_pdf_sha256='e56ad687c36cd8c34f45dce72a29891b2d13590d422ef6ec98dda185910999d0',
        harness_sha256=source_hash,unmodified_source_sha256={k:hashlib.sha256(v).hexdigest() for k,v in original.items()},
        modifications='Only listed parameters.f90 values; no solver or IO source modified',
        input='Zero velocity and temperature perturbations around author bases1-r^2,r^2; no Atomic Sim profile used',
        subspace=dict(axial_modes=1,azimuthal_modes=1,turbulent=False),time_unit='R/(2Ub)',records=records,
        note='Independent locally executed numerical software, not measurements or turbulent DNS; code is not redistributed')
    atomic_json(out/'fixture.json',seal(fixture))
    atomic_json(out/'audit.json',seal(dict(passed=True,harness_sha256=source_hash,archive_sha256=ARCHIVE_SHA,
        compiler=subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0],
        netcdf_module_sha256=sha(module),netcdf_library_sha256=sha(lib),work=work,fixture_sha256=sha(out/'fixture.json'))))
    return fixture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','archive','dependencies'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.output,a.archive,a.dependencies)

if __name__=='__main__':main()
