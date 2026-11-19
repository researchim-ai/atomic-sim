"""Explicitly patched external reference; original v67 evidence is retained.

Every case starts from a fresh pinned archive, not copied calculation outputs.
Only parameters and the declared quadrature-read patch modify author source.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import numpy as np
from scipy.io import netcdf_file
from atomic_model_station_quadrature import correct_quadrature,POLICY
from atomic_model_station_quadrature import model as math_model
from atomic_model_station_heated.native_reference import load_quadrature
from scripts import replay_station_heated_reference as original_harness
from scripts.station_process import execute
from atomic_model_multiphysics.provenance import sha,seal
from atomic_model_training.spec import atomic_json

PATCH='''
      ! Explicit numerical variant: corrected integration weights, no flow fit.
      open(97,file='corrected_weights.dat',status='old',iostat=N_)
      if(N_/=0) stop 'corrected quadrature: missing file'
      do n=1,i_N
         read(97,*,iostat=N_) dr,mes_D%intrdr(n)
         if(N_/=0) stop 'corrected quadrature: incomplete file'
         if(abs(dr-mes_D%r(n,1))>5d-14) stop 'corrected quadrature: mesh mismatch'
         if(mes_D%intrdr(n)<=0d0) stop 'corrected quadrature: nonpositive weight'
      end do
      close(97)
'''
ANCHOR='      call mes_rdom_init(mes_D)'


def cases():
    rows=[(bc,c,n,.1 if bc=='delta_temperature' else .01) for bc in ('delta_temperature','heat_flux') for c in (0,2,5) for n in (32,64)]
    return rows+[('delta_temperature',2,64,.05),('heat_flux',2,64,.005)]


def patch_mesh(raw):
    text=raw.decode()
    if text.count(ANCHOR)!=1:raise ValueError('Pinned mesh-precompute anchor differs')
    return text.replace(ANCHOR,ANCHOR+PATCH,1).encode()


def checked_state(path,n,c,step=None):
    with netcdf_file(path,'r',mmap=False) as state:
        for name,expected in dict(Re=5300.,Pr=.7,C=float(c),alpha=.628319).items():
            value=np.asarray(getattr(state,name))
            if value.ndim or not np.isfinite(value) or abs(float(value)-expected)>1e-13:
                raise ValueError('External finite scalar metadata differs: '+name)
        if not np.isfinite(state.t):raise ValueError('External time must be finite')
        dt=float(state.variables['dt'][...])
        if not np.isfinite(dt) or dt<=0 or (step is not None and abs(dt-step)>1e-14):raise ValueError('External time-step metadata differs')
        for name in ('Ur','Ut','Uz','T'):
            variable=state.variables[name];values=variable[:]
            if values.shape!=(2,1,n) or not np.all(np.isfinite(values)):raise ValueError('Complete finite mean field required')
            if any(np.asarray(getattr(variable,key)).ndim or getattr(variable,key)!=1 for key in ('K','M','Mp')):
                raise ValueError('Explicit mean-only mode metadata required')
            if name in ('Ur','Ut') and np.max(np.abs(values))>1e-13:raise ValueError('Radial/swirl field outside selected subspace')
    return original_harness.read_state(path,n,c)


def completed_states(directory,n,c,step=None):
    directory=Path(directory);paths=sorted(directory.glob('state*.cdf.dat'))
    if len(paths)!=5:raise ValueError('Complete five-state external output required')
    final=checked_state(paths[-1],n,c,step);late=checked_state(paths[-2],n,c,step)
    if abs(final['time']-8000)>1e-6 or abs(late['time']-6000)>1e-6:raise ValueError('Corrected external horizon incomplete')
    settling=float(np.max(np.abs(np.array([final['velocity'],final['temperature']])-np.array([late['velocity'],late['temperature']]))))
    if settling>1e-7:raise ValueError('Corrected external profile not settled')
    return paths,final,late,settling


def run(output,archive,dependencies):
    out=Path(output).absolute();archive=Path(archive).absolute();deps=Path(dependencies).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    originals=original_harness.inventory(archive);native=load_quadrature()
    snapshots={str(p):sha(p) for p in (Path(__file__),Path(math_model.__file__),Path(original_harness.__file__))}
    module=deps/'usr/include/netcdf.mod';lib=deps/'usr/lib/x86_64-linux-gnu'
    if not module.is_file() or not (lib/'libnetcdff.so').is_file():raise ValueError('Explicit local Fortran NetCDF dependency required')
    rules={n:correct_quadrature(p['radius'],p['weights_rdr']) for n,p in native['records'].items()}
    out.mkdir(parents=True);records={};work=[]
    for bc,c,n,step in cases():
        name=f'{bc}-c{c}-n{n}-dt{step}';directory=out/name;directory.mkdir()
        for filename,raw in originals.items():
            target=directory/filename;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        # No inherited run products may masquerade as a fresh final state.
        if list(directory.glob('state*.cdf.dat')):raise ValueError('Archive unexpectedly contains calculation outputs')
        params,values=original_harness.parameters(originals['program/parameters.f90'],cells=n,c=c,boundary=bc,step=step)
        (directory/'program/parameters.f90').write_bytes(params)
        (directory/'program/meshs.f90').write_bytes(patch_mesh(originals['program/meshs.f90']))
        rule=rules[str(n)];weight_file=directory/'corrected_weights.dat'
        np.savetxt(weight_file,np.column_stack([rule['radius'],rule['corrected_weights']]),fmt='%.17e')
        expected={filename:sha(directory/filename) for filename in originals};weight_hash=sha(weight_file)
        flags=f'-ffree-line-length-none -x f95-cpp-input -c -O2 -I/usr/include -I{module.parent}'
        libs=f'-Wl,-rpath,{lib} -L{lib} cheby.o -l:libfftw3.so.3 -llapack -lnetcdff -lnetcdf'
        execute(['make','-j1','all','COMPFLAGS='+flags,'LIBS='+libs],cwd=directory,log=directory/'build.log',timeout_s=120,idle_s=60)
        original_harness.initial(directory/'state.cdf.in',n,c,step);initial_hash=sha(directory/'state.cdf.in')
        execute([str(directory/'main.out')],cwd=directory,log=directory/'run.log',timeout_s=180,idle_s=60)
        log=(directory/'run.log').read_text()
        if 'maxtstep reached!' not in log or '...done!' not in log:raise ValueError('Corrected solver did not complete declared step budget')
        paths,final,late,settling=completed_states(directory,n,c,step)
        if not np.allclose(final['radius'],rule['radius'],rtol=0,atol=5e-14):raise ValueError('Native mesh differs from correction input')
        if expected!={filename:sha(directory/filename) for filename in originals} or weight_hash!=sha(weight_file) or initial_hash!=sha(directory/'state.cdf.in'):
            raise ValueError('Corrected source, weights or initial state changed during execution')
        changes={filename:dict(original=hashlib.sha256(raw).hexdigest(),modified=expected[filename]) for filename,raw in originals.items() if hashlib.sha256(raw).hexdigest()!=expected[filename]}
        if set(changes)!={'program/parameters.f90','program/meshs.f90'}:raise ValueError('Unexpected external source changes')
        records[name]=dict(boundary=bc,c=c,cells=n,step=step,final=final,late=late,late_maximum=settling,
            final_raw_sha256=sha(paths[-1]),late_raw_sha256=sha(paths[-2]),weights_sha256=weight_hash,
            initial_sha256=initial_hash,source_changes=changes,parameters=values)
        work.append(dict(case=name,binary_sha256=sha(directory/'main.out'),initial_sha256=initial_hash))
        print('corrected external mean',name,'complete; late difference',settling,flush=True)
    if snapshots!={name:sha(name) for name in snapshots}:raise ValueError('Corrected replay sources changed')
    fixture=dict(format='openpipeflow-corrected-radial-quadrature/v1',quadrature_policy=POLICY,
        archive_sha256=original_harness.ARCHIVE_SHA,archive_url='https://www.openpipeflow.org/images/5/53/Openpipeflow-hp-1.21.tgz',
        paper_doi='10.3390/math13020293',harness_sha256=sha(__file__),math_source_sha256=sha(math_model.__file__),
        original_harness_sha256=sha(original_harness.__file__),patch=PATCH,rules=rules,records=records,
        original_source_sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in originals.items()},
        subspace=dict(axial_modes=1,azimuthal_modes=1,turbulent=False),time_unit='R/(2Ub)',
        source_variant='Explicit corrected numerical quadrature; not the unmodified author code',
        inputs='Zero perturbations of author bases1-r^2,r^2; no Atomic Sim profile used',
        physical_model_coefficients_changed=False,original_v67_fixture_updated=False)
    atomic_json(out/'fixture.json',seal(fixture))
    atomic_json(out/'audit.json',seal(dict(passed=True,harness_sha256=sha(__file__),math_source_sha256=sha(math_model.__file__),
        fixture_sha256=sha(out/'fixture.json'),compiler=subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0],
        netcdf_module_sha256=sha(module),netcdf_library_sha256=sha(lib/'libnetcdff.so'),work=work)))
    return fixture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','archive','dependencies'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.output,a.archive,a.dependencies)

if __name__=='__main__':main()
