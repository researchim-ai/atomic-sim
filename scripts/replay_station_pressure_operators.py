"""Export native radial FD coefficients from fresh pinned author-source builds."""
import argparse,re
from pathlib import Path
import numpy as np
from scripts import replay_station_heated_reference as base
from scripts import replay_station_spatial as spatial
from scripts.station_process import execute
from atomic_model_multiphysics.provenance import sha,seal
from atomic_model_training.spec import atomic_json

PROGRAM='''program export_pressure_mesh
use meshs
implicit none
integer n,j
call mpi_precompute()
call par_precompute()
call mes_precompute()
open(91,file='radius.dat',status='new')
do n=1,i_N
write(91,'(2es26.17)') mes_D%r(n,1),mes_D%intrdr(n)
end do
close(91)
open(91,file='bands.dat',status='new')
do n=1,i_N
do j=max(1-i_KL,n-i_KL),min(i_N,n+i_KL)
write(91,'(2i8,2es26.17)') n,j,mes_D%dr(1)%M(i_KL+1+n-j,j),mes_D%dr(2)%M(i_KL+1+n-j,j)
end do
end do
close(91)
end program export_pressure_mesh
'''


def run(output,archive,dependencies):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    originals=base.inventory(archive);flags,libs=spatial.build_flags(Path(dependencies).absolute());own=sha(__file__)
    out.mkdir(parents=True);records={}
    for n in (32,64,96):
        d=out/f'n{n}'
        params,_=base.parameters(originals['program/parameters.f90'],cells=min(n,64),c=0,boundary='delta_temperature',step=.01)
        if n==96:
            text,count=re.subn(r'(?m)^(\s*[^!\n]*::\s*i_N\s*=)[^\n]*$',r'\g<1> 96',params.decode());assert count==1;params=text.encode()
        spatial.write_sources(d,originals,params);source={name:sha(d/name) for name in originals}
        execute(['make','-j1','meshs.o','COMPILER=gfortran -mcmodel=medium','COMPFLAGS='+flags,'LIBS='+libs],cwd=d,log=d/'build.log',timeout_s=120,idle_s=60)
        (d/'export.f90').write_text(PROGRAM)
        execute(['gfortran','-mcmodel=medium','-I',str(d),'-o',str(d/'export.out'),str(d/'export.f90'),*[str(d/x) for x in ('meshs.o','mpi.o','parameters.o','cheby.o')],'-llapack'],cwd=d,log=d/'link.log',timeout_s=60,idle_s=30)
        execute([str(d/'export.out')],cwd=d,log=d/'export.log',timeout_s=60,idle_s=30)
        assert source=={name:sha(d/name) for name in originals}
        radius=np.loadtxt(d/'radius.dat');bands=np.loadtxt(d/'bands.dat')
        if radius.shape!=(n,2) or bands.ndim!=2 or bands.shape[1]!=4 or not np.all(np.isfinite(bands)):raise ValueError('Complete finite native coefficients required')
        records[str(n)]=dict(radius=radius[:,0].tolist(),native_weights=radius[:,1].tolist(),bands=bands.tolist(),source_sha256=source,binary_sha256=sha(d/'export.out'),radius_sha256=sha(d/'radius.dat'),bands_sha256=sha(d/'bands.dat'))
        print('Native pressure mesh exported',n,flush=True)
    if own!=sha(__file__):raise ValueError('Exporter source changed')
    result=dict(format='native-pressure-operator-bands/v1',archive_sha256=base.ARCHIVE_SHA,harness_sha256=own,program=PROGRAM,records=records,scope='Native unconstrained radial operators; no time evolution or pressure boundary-row export')
    atomic_json(out/'operators.json',seal(result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','archive','dependencies'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.output,a.archive,a.dependencies)

if __name__=='__main__':main()
