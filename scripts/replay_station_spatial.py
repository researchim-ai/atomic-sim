"""Bounded real Fourier-mode heated-pipe experiments, not a statistical DNS certificate."""
import argparse
import hashlib
from pathlib import Path
import re
import shutil
import tarfile
import numpy as np
from scipy.io import netcdf_file
from atomic_model_station_spectral import audit_state,compare_states
from atomic_model_station_spectral import io as audit_module
from atomic_model_station_spectral import modes as mode_module
from atomic_model_station_quadrature import correct_quadrature
from scripts import replay_station_heated_reference as base
from scripts import replay_station_corrected_quadrature as corrected
from scripts.audit_station_heated_quadrature import PROGRAM as WEIGHT_PROGRAM
from scripts.station_process import execute
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json

SEED_SHA='35169ca439d2328ab0613169470d8638564997ce53b9a5a19d17a3868d10b88a'
SEED_MEMBER='Re5300.Retau180.5D/state0005.cdf.dat'
NATIVE_AUDIT='''program audit_native
use io
implicit none
type(coll) dv
double precision e,ek(0:i_K1),em(0:i_M1),total,thermal,divnorm
character(200) statefile
call get_command_argument(1,statefile)
call mpi_precompute()
call par_precompute()
call mes_precompute()
call var_precompute()
call vel_precompute()
call temp_precompute()
call io_precompute()
io_statefile=statefile
call io_load_state()
call var_coll_norm(vel_ur,e,ek,em)
total=e
call var_coll_norm(vel_ut,e,ek,em)
total=total+e
call var_coll_norm(vel_uz,e,ek,em)
total=total+e
call var_coll_norm(temp_tau,thermal,ek,em)
call var_coll_div(vel_ur,vel_ut,vel_uz,dv)
call var_coll_norm(dv,divnorm,ek,em)
write(*,'(a,3es26.17)') 'AUDIT_VALUES ',total,thermal,divnorm
end program audit_native
'''


def parameter_source(raw,n,k,m,step,steps,restart=False):
    if n not in (64,96) or (k,m) not in ((80,76),(96,96)) or step not in (.01,.005) or steps not in (10,20,40):raise ValueError('Declared spatial probe required')
    values=dict(i_N=str(n),i_K=str(k),i_M=str(m),d_C='0d0',i_maxtstep=str(steps),i_save_rate1=str(steps//2),i_save_rate2='1',
        d_maxt='-1d0',d_minE3d='-1d0',d_timestep=str(step)+'d0',d_maxdt=str(step)+'d0',d_dterr='1d-7',d_time='-1d0' if restart else '0d0')
    text=raw.decode()
    for key,value in values.items():
        text,count=re.subn(r'(?m)^(\s*[^!\n]*::\s*'+key+r'\s*=)[^\n]*$',lambda q:q[1]+' '+value,text)
        if count!=1:raise ValueError('Spatial source anchor differs: '+key)
    return text.encode(),values


def initial(seed,target):
    shutil.copyfile(seed,target)
    def hashes():
        with netcdf_file(target,'r',mmap=False) as f:return {name:hashlib.sha256(f.variables[name][:].tobytes()).hexdigest() for name in ('Ur','Ut','Uz','r')}
    before=hashes()
    with netcdf_file(target,'a',mmap=False) as f:
        f.t=np.float64(0.);f.C=np.float64(0.);f.Pr=np.float64(.7)
        temp=f.createVariable('T','d',('ReIm','H','r'));temp.K=np.int32(96);temp.M=np.int32(96);temp.Mp=np.int32(1);temp[:]=0.
    if before!=hashes():raise ValueError('Upstream velocity coefficients changed during thermal initialization')
    return before


def write_sources(directory,originals,params):
    directory.mkdir()
    for name,raw in originals.items():
        p=directory/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(raw)
    if list(directory.glob('state*.cdf.dat')):raise ValueError('Inherited calculation outputs refused')
    (directory/'program/parameters.f90').write_bytes(params)


def build_flags(deps):
    include=deps/'usr/include';lib=deps/'usr/lib/x86_64-linux-gnu'
    # Make expands variable contents as code, so external dependency paths are restricted.
    if not re.fullmatch(r'[A-Za-z0-9_./-]+',str(deps)):raise ValueError('Simple local dependency path required for legacy Makefile')
    if not (include/'netcdf.mod').is_file() or not (lib/'libnetcdff.so').is_file():raise ValueError('Local Fortran NetCDF dependency required')
    return f'-ffree-line-length-none -x f95-cpp-input -c -O2 -I/usr/include -I{include}',f'-Wl,-rpath,{lib} -L{lib} cheby.o -l:libfftw3.so.3 -llapack -lnetcdff -lnetcdf'


def mesh_rule(out,originals,n,flags,libs):
    directory=out/f'mesh{n}';params,_=parameter_source(originals['program/parameters.f90'],n,80,76,.01,20)
    # Only geometry is evaluated; do not advance or allocate the flow fields.
    write_sources(directory,originals,params)
    execute(['make','-j1','meshs.o','COMPILER=gfortran -mcmodel=medium','COMPFLAGS='+flags,'LIBS='+libs],cwd=directory,log=directory/'build.log',timeout_s=120,idle_s=60)
    (directory/'weights.f90').write_text(WEIGHT_PROGRAM)
    execute(['gfortran','-mcmodel=medium','-I',str(directory),'-o',str(directory/'weights.out'),str(directory/'weights.f90'),
             *[str(directory/x) for x in ('meshs.o','mpi.o','parameters.o','cheby.o')],'-llapack'],cwd=directory,log=directory/'link.log',timeout_s=60,idle_s=30)
    execute([str(directory/'weights.out')],cwd=directory,log=directory/'weights.log',timeout_s=60,idle_s=30)
    data=np.loadtxt(directory/'native_weights.dat');return correct_quadrature(data[:,0],data[:,1])


def run(output,archive,seed_archive,dependencies):
    out=Path(output).absolute();deps=Path(dependencies).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    if sha(seed_archive)!=SEED_SHA:raise ValueError('Pinned public turbulent initial state required')
    originals=base.inventory(archive);flags,libs=build_flags(deps);own=sha(__file__);math_hash=sha(audit_module.__file__);modes_hash=sha(mode_module.__file__)
    out.mkdir(parents=True)
    with tarfile.open(seed_archive) as tar:
        member=tar.getmember(SEED_MEMBER)
        if not member.isfile() or member.size>100000000:raise ValueError('Bounded upstream checkpoint required')
        seed=out/'upstream.cdf';seed.write_bytes(tar.extractfile(member).read())
    rules={str(n):mesh_rule(out,originals,n,flags,libs) for n in (64,96)};records={};paths={}
    specs=[('whole',64,80,76,.01,20,None),('time_half',64,80,76,.005,40,None),
           ('fourier_fine',64,96,96,.01,20,None),('radial_fine',96,80,76,.01,20,None),('restart',64,80,76,.01,10,'whole')]
    for name,n,k,m,step,steps,resume in specs:
        directory=out/name;params,changes=parameter_source(originals['program/parameters.f90'],n,k,m,step,steps,bool(resume))
        write_sources(directory,originals,params);(directory/'program/meshs.f90').write_bytes(corrected.patch_mesh(originals['program/meshs.f90']))
        rule=rules[str(n)];weights=np.array(rule['corrected_weights']);np.savetxt(directory/'corrected_weights.dat',np.column_stack([rule['radius'],weights]),fmt='%.17e')
        expected={file:sha(directory/file) for file in originals};weight_hash=sha(directory/'corrected_weights.dat')
        if resume:
            shutil.copyfile(paths['whole'][1],directory/'state.cdf.in');initial_hashes=None;start=.1
        else:initial_hashes=initial(seed,directory/'state.cdf.in');start=0.
        input_sha=sha(directory/'state.cdf.in')
        execute(['make','-j1','all','COMPILER=gfortran -mcmodel=medium','COMPFLAGS='+flags,'LIBS='+libs],cwd=directory,log=directory/'build.log',timeout_s=180,idle_s=120)
        execute([str(directory/'main.out')],cwd=directory,log=directory/'run.log',timeout_s=900,idle_s=180)
        states=sorted(directory.glob('state*.cdf.dat'))
        if len(states)!=3:raise ValueError('Complete spatial initial/midpoint/final states required')
        snapshots=[audit_state(p,weights) for p in states]
        expected_times=[start,start+steps*step/2,start+steps*step]
        if max(abs(p['time']-t) for p,t in zip(snapshots,expected_times))>1e-12:raise ValueError('Spatial solver ended at an unexpected epoch')
        if any(p['axial_modes']!=k or p['azimuthal_modes']!=m or p['radial_nodes']!=n for p in snapshots):raise ValueError('Declared spatial resolution differs')
        energy=np.loadtxt(directory/'vel_energy.dat');thermal=np.loadtxt(directory/'temp_flux.dat');friction=np.loadtxt(directory/'vel_friction.dat')
        for table,columns in ((energy,5),(thermal,4),(friction,4)):
            if table.shape!=(steps,columns) or not np.all(np.isfinite(table)):raise ValueError('Complete finite diagnostic traces required')
            if not np.allclose(table[:,0],start+np.arange(steps)*step,rtol=0,atol=1e-11):raise ValueError('Diagnostic epochs differ')
        (directory/'audit_native.f90').write_text(NATIVE_AUDIT)
        objects=['io.o','meshs.o','mpi.o','nonlinear.o','parameters.o','temperature.o','timestep.o','transform.o','variables.o','velocity.o']
        execute(['gfortran','-mcmodel=medium','-I',str(directory),'-o',str(directory/'audit_native.out'),str(directory/'audit_native.f90'),
             *[str(directory/p) for p in objects],*libs.split()],cwd=directory,log=directory/'audit-build.log',timeout_s=60,idle_s=30)
        native=[]
        for index,state in enumerate(states):
            log=directory/f'audit{index}.log'
            execute([str(directory/'audit_native.out'),state.name],cwd=directory,log=log,timeout_s=120,idle_s=60)
            match=re.search(r'AUDIT_VALUES\s+(\S+)\s+(\S+)\s+(\S+)',log.read_text())
            if not match:raise ValueError('Independent native diagnostics missing')
            values=[float(x) for x in match.groups()]
            if not np.all(np.isfinite(values)):raise ValueError('Native audit is nonfinite')
            volume=2*np.pi**2/snapshots[index]['metadata']['alpha']
            native.append(dict(kinetic_integral=values[0],thermal_integral=values[1],divergence_l2=float(np.sqrt(2*values[2]/volume))))
        if expected!={file:sha(directory/file) for file in originals} or sha(directory/'corrected_weights.dat')!=weight_hash or sha(directory/'state.cdf.in')!=input_sha:
            raise ValueError('Spatial sources or input changed')
        records[name]=dict(spec=dict(n=n,k=k,m=m,step=step,steps=steps,start=start),snapshots=snapshots,native_audits=native,
            trace=dict(time=energy[:,0].tolist(),kinetic=energy[:,1].tolist(),axial_zero=energy[:,2].tolist(),azimuthal_zero=energy[:,3].tolist(),thermal_energy=energy[:,4].tolist(),
                axial_gradient=thermal[:,1].tolist(),wall_gradient=thermal[:,3].tolist(),friction_velocity=friction[:,3].tolist()),
            input_sha256=input_sha,upstream_velocity_sha256=initial_hashes,source_sha256=expected,weights_sha256=weight_hash,
            state_sha256={p.name:sha(p) for p in states},binary_sha256=sha(directory/'main.out'))
        paths[name]=states;print('spatial case complete',name,'time',snapshots[-1]['time'],flush=True)
    w=rules['64']['corrected_weights']
    comparison={name:compare_states(paths['whole'][-1],paths[name][-1],w) for name in ('time_half','fourier_fine','radial_fine','restart')}
    prefix=compare_states(paths['whole'][1],out/'restart/state.cdf.in',w)
    if own!=sha(__file__) or math_hash!=sha(audit_module.__file__) or modes_hash!=sha(mode_module.__file__):raise ValueError('Spatial replay source changed')
    fixture=dict(format='heated-spatial-integration-probe/v1',harness_sha256=own,audit_source_sha256=math_hash,
        modes_source_sha256=modes_hash,archive_sha256=base.ARCHIVE_SHA,seed_archive_sha256=SEED_SHA,seed_url='https://www.openpipeflow.org/images/2/2b/Re5300.Retau180.5D.tgz',
        upstream_checkpoint_sha256=sha(seed),rules=rules,records=records,comparisons=comparison,restart_prefix=prefix,
        source_variant='Author equations with explicit v68 corrected quadrature; parameters and mesh-weight input changed',
        thermal_initialization='Zero temperature perturbation on top of author r^2 base; velocity snapshot preserved; thermal spinup required',
        scope='Short three-dimensional integration experiments; no statistical steady state or DNS qualification')
    atomic_json(out/'fixture.json',seal(fixture));return fixture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','archive','seed-archive','dependencies'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.output,a.archive,a.seed_archive,a.dependencies)

if __name__=='__main__':main()
