"""Read native mean weights from unmodified compiled external mesh routines."""
import argparse
from pathlib import Path
import numpy as np
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_inventory import read
from scripts.station_process import execute

PROGRAM='''program native_weights
 use meshs
 implicit none
 integer n
 call mpi_precompute()
 call par_precompute()
 call mes_precompute()
 open(91,file='native_weights.dat',status='replace')
 do n=1,i_N
  write(91,'(2es26.17)') mes_D%r(n,1),mes_D%intrdr(n)
 end do
 close(91)
end program native_weights
'''


def run(output,external):
    out=Path(output).absolute();external=Path(external).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    fixture=read(external/'fixture.json');audit=read(external/'audit.json')
    if not audit['passed'] or audit['fixture_sha256']!=sha(external/'fixture.json'):raise ValueError('Accepted external replay required')
    out.mkdir(parents=True);records={}
    for n in (32,64):
        src=external/f'delta_temperature-c0-n{n}-dt0.1';source=src/'native_weights.f90'
        if source.exists():raise ValueError('Existing diagnostic source refused')
        source.write_text(PROGRAM)
        execute(['gfortran','-I',str(src),'-o',str(src/'native_weights.out'),str(source),*[str(src/p) for p in ('meshs.o','mpi.o','parameters.o','cheby.o')],'-llapack'],
            cwd=src,log=out/f'build{n}.log',timeout_s=30,idle_s=15)
        execute([str(src/'native_weights.out')],cwd=src,log=out/f'weights{n}.log',timeout_s=30,idle_s=15)
        data=np.loadtxt(src/'native_weights.dat');row=fixture['records'][f'delta_temperature-c0-n{n}-dt0.1']
        if data.shape!=(n,2) or not np.allclose(data[:,0],row['final']['radius'],rtol=0,atol=1e-15):raise ValueError('Native mesh differs')
        moments={str(k):dict(native=float(data[:,1]@(data[:,0]**k)),exact=1/(k+2),error=float(data[:,1]@(data[:,0]**k)-1/(k+2))) for k in (0,2,4,6)}
        records[str(n)]=dict(radius=data[:,0].tolist(),weights_rdr=data[:,1].tolist(),moments=moments)
        source.unlink() # keep outputs; no source from external archive was altered
    result=dict(format='openpipeflow-native-quadrature/v1',external_fixture_sha256=sha(external/'fixture.json'),
        harness_sha256=sha(__file__),diagnostic_source=PROGRAM,records=records,
        scope='Native integration weights only; no solver source or physics modified')
    atomic_json(out/'quadrature.json',seal(result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--external',type=Path,required=True)
    a=p.parse_args();run(a.output,a.external)

if __name__=='__main__':main()
