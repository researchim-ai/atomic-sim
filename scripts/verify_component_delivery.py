"""Offline scoped source/sdist/wheel acceptance for registered physics components.

Does not replace the full library profile or qualify a real installation.
Scientific dependencies are reused from the invoking development environment.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv

from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_training.spec import atomic_json,read_json
from scripts.verify_local_library import source_manifest,evidence_manifest,evidence_unchanged,check_wheel

ROOT=Path(__file__).resolve().parents[1]
TESTS=('test_pwr_local_cladding.py','test_pwr_distributed_fuel.py','test_station_core.py',
    'test_station_power.py','test_conservative_exchange.py','test_core_channel_transient.py')
COLD_TESTS=TESTS[:4]
PROFILES={
    'numerical-reuse':dict(tests=('test_numerical_reuse.py','test_cached_exchanger.py','test_sparse_linear.py','test_sparse_steady_exchanger.py','test_pwr_sparse.py','test_component_delivery.py'),
        cold_tests=('test_numerical_reuse.py','test_cached_exchanger.py','test_sparse_linear.py','test_sparse_steady_exchanger.py','test_pwr_sparse.py','test_component_delivery.py'),module='scripts.qualify_numerical_reuse',
        imports=('physics.thermal.cached_exchanger','core.sparse_linear','coupling.pwr_sparse','scripts.qualify_numerical_reuse'),format='numerical-reuse-delivery/v1',scientific_limit_bytes=16*1024*1024),
    'core-fidelity':dict(tests=('test_core_fidelity.py','test_neutron_mesh.py','test_mesh_comparison.py','test_component_delivery.py'),
        cold_tests=('test_core_fidelity.py','test_mesh_comparison.py','test_component_delivery.py'),module='scripts.qualify_core_fidelity',
        imports=('coupling.core_fidelity','scripts.qualify_core_fidelity'),format='core-fidelity-delivery/v1',scientific_limit_bytes=16*1024*1024),
    'pwr-sparse':dict(tests=('test_pwr_sparse.py','test_component_delivery.py'),
        cold_tests=('test_pwr_sparse.py','test_component_delivery.py'),module='scripts.qualify_pwr_sparse',
        imports=('coupling.pwr_sparse','atomic_model_station.pwr_train','scripts.qualify_pwr_sparse'),format='pwr-sparse-delivery/v1',scientific_limit_bytes=16*1024*1024),
    'sparse-steady':dict(tests=('test_sparse_linear.py','test_sparse_steady_exchanger.py','test_two_stream_exchanger.py','test_component_delivery.py'),
        cold_tests=('test_sparse_linear.py','test_sparse_steady_exchanger.py','test_component_delivery.py'),module='scripts.qualify_sparse_steady',
        imports=('core.sparse_linear','physics.thermal.sparse_steady_exchanger','scripts.qualify_sparse_steady'),format='sparse-steady-delivery/v1',scientific_limit_bytes=8*1024*1024),
    'exchanger-cache':dict(tests=('test_cached_exchanger.py','test_two_stream_exchanger.py','test_component_delivery.py'),
        cold_tests=('test_cached_exchanger.py','test_component_delivery.py'),module='scripts.qualify_exchanger_cache',
        imports=('physics.thermal.cached_exchanger','scripts.qualify_exchanger_cache'),format='exchanger-cache-delivery/v1',scientific_limit_bytes=8*1024*1024),
    'boron-dosing':dict(tests=('test_boron_dosing.py','test_boron_closure.py','test_regenerator.py','test_pwr_boron.py','test_component_delivery.py'),
        cold_tests=('test_boron_dosing.py','test_boron_closure.py','test_pwr_boron.py','test_component_delivery.py'),module='scripts.qualify_boron_dosing',
        imports=('coupling.boron_dosing','coupling.pwr_boron','scripts.qualify_boron_dosing'),format='boron-dosing-delivery/v1',scientific_limit_bytes=8*1024*1024),
    'boron-closure':dict(tests=('test_boron_closure.py','test_boron_duration.py','test_regenerator.py','test_regenerator_qualification.py','test_pwr_boron.py','test_component_delivery.py'),
        cold_tests=('test_boron_closure.py','test_boron_duration.py','test_regenerator_qualification.py','test_pwr_boron.py','test_component_delivery.py'),module='scripts.qualify_boron_closure',
        imports=('coupling.pwr_boron','scripts.qualify_boron_closure'),format='boron-closure-delivery/v1',scientific_limit_bytes=8*1024*1024),
    'boron-duration':dict(tests=('test_boron_duration.py','test_regenerator.py','test_regenerator_qualification.py','test_pwr_boron.py','test_component_delivery.py'),
        cold_tests=('test_boron_duration.py','test_regenerator_qualification.py','test_pwr_boron.py','test_component_delivery.py'),module='scripts.qualify_boron_duration',
        imports=('coupling.pwr_boron','scripts.qualify_boron_duration'),format='boron-duration-delivery/v1',scientific_limit_bytes=8*1024*1024),
    'boron-ports':dict(tests=('test_boron_ports.py','test_regenerator.py','test_regenerator_qualification.py','test_pwr_boron.py','test_component_delivery.py'),
        cold_tests=('test_boron_ports.py','test_regenerator_qualification.py','test_pwr_boron.py','test_component_delivery.py'),module='scripts.qualify_boron_ports',
        imports=('coupling.boron_ports','coupling.pwr_boron','scripts.qualify_boron_ports'),
        format='boron-ports-delivery/v1',scientific_limit_bytes=8*1024*1024),
    'boron-regenerator':dict(tests=('test_regenerator.py','test_pwr_boron.py','test_boron_mesh.py','test_regenerator_qualification.py','test_component_delivery.py'),
        cold_tests=('test_regenerator.py','test_pwr_boron.py','test_regenerator_qualification.py','test_component_delivery.py'),
        module='scripts.qualify_boron_regenerator',imports=('coupling.regenerator','coupling.pwr_boron','scripts.qualify_boron_regenerator'),
        format='boron-regenerator-delivery/v1',scientific_limit_bytes=8*1024*1024),
    'boron-mesh':dict(tests=('test_boron_mesh.py','test_pwr_boron.py','test_coupled_mesh.py','test_hydraulic_geometry.py','test_component_delivery.py'),
        cold_tests=('test_boron_mesh.py','test_pwr_boron.py','test_component_delivery.py'),module='scripts.qualify_boron_mesh',
        imports=('coupling.pwr_boron','scripts.qualify_boron_mesh'),format='boron-mesh-delivery/v1',scientific_limit_bytes=16*1024*1024),
    'boron-exchange':dict(tests=('test_pwr_boron.py','test_pwr_moderator.py','test_pwr_channel.py','test_component_delivery.py'),
        cold_tests=('test_pwr_boron.py','test_component_delivery.py'),module='scripts.qualify_boron_exchange',
        imports=('coupling.pwr_boron','scripts.qualify_boron_exchange'),format='boron-exchange-delivery/v1',scientific_limit_bytes=8*1024*1024),
    'coupled-mesh':dict(tests=('test_coupled_mesh.py','test_thermal_mesh.py','test_hydraulic_geometry.py','test_mesh_comparison.py','test_neutron_mesh.py','test_pwr_channel.py','test_component_delivery.py'),
        cold_tests=('test_coupled_mesh.py','test_thermal_mesh.py','test_hydraulic_geometry.py','test_mesh_comparison.py','test_component_delivery.py'),
        module='scripts.qualify_coupled_mesh',imports=('coupling.refinement','coupling.pwr_moderator','scripts.qualify_coupled_mesh'),
        format='coupled-mesh-delivery/v1',scientific_limit_bytes=8*1024*1024),
    'fluid-geometry':dict(tests=('test_hydraulic_geometry.py','test_fluid_geometry_qualification.py','test_pwr_channel.py','test_pwr_stationary.py','test_pwr_moderator.py','test_component_delivery.py'),
        cold_tests=('test_hydraulic_geometry.py','test_fluid_geometry_qualification.py','test_pwr_channel.py','test_component_delivery.py'),
        module='scripts.qualify_fluid_geometry',imports=('coupling.hydraulic_geometry','coupling.pwr_channel','scripts.qualify_fluid_geometry'),
        format='fluid-geometry-delivery/v1',scientific_limit_bytes=8*1024*1024),
    'thermal-mesh':dict(tests=('test_thermal_mesh.py','test_mesh_comparison.py','test_conservative_exchange.py','test_pwr_local_cladding.py','test_pwr_channel.py','test_component_delivery.py'),
        cold_tests=('test_thermal_mesh.py','test_mesh_comparison.py','test_pwr_local_cladding.py','test_component_delivery.py'),
        module='scripts.qualify_thermal_mesh',imports=('coupling.thermal_grid','coupling.pwr_moderator','scripts.qualify_thermal_mesh'),
        format='thermal-mesh-delivery/v1',scientific_limit_bytes=8*1024*1024),
    'neutron-mesh':dict(tests=('test_mesh_comparison.py','test_neutron_mesh.py','test_conservative_exchange.py','test_pwr_moderator.py','test_station_core.py','test_component_delivery.py'),
        cold_tests=('test_mesh_comparison.py','test_neutron_mesh.py','test_pwr_moderator.py','test_component_delivery.py'),
        module='scripts.qualify_neutron_mesh',imports=('coupling.mesh_comparison','coupling.pwr_moderator','scripts.qualify_neutron_mesh'),
        format='neutron-mesh-delivery/v1',scientific_limit_bytes=8*1024*1024),
    'moderator-transients':dict(tests=('test_moderator_transients.py','test_pwr_moderator.py','test_station_actuators.py','test_pwr_channel.py',
            'test_station_piecewise_heat.py','test_station_event_heat.py','test_component_delivery.py'),
        cold_tests=('test_moderator_transients.py','test_pwr_moderator.py','test_station_actuators.py','test_station_piecewise_heat.py','test_station_event_heat.py','test_component_delivery.py'),
        module='scripts.qualify_moderator_transients',imports=('coupling.pwr_moderator','coupling.pwr_stationary','scripts.qualify_moderator_transients'),
        format='moderator-transients-delivery/v1',scientific_limit_bytes=8*1024*1024),
    'pwr-moderator':dict(tests=('test_pwr_moderator.py','test_pwr_stationary.py','test_pwr_channel.py','test_pwr_local_cladding.py',
            'test_pwr_distributed_fuel.py','test_conservative_exchange.py','test_eos_solver_precision.py','test_component_delivery.py',
            'test_station_core.py','test_station_power.py'),
        cold_tests=('test_pwr_moderator.py','test_pwr_stationary.py','test_pwr_channel.py','test_pwr_distributed_fuel.py','test_eos_solver_precision.py','test_component_delivery.py'),
        module='scripts.qualify_pwr_moderator',imports=('coupling.pwr_moderator','coupling.pwr_fuel','coupling.pwr_stationary'),
        format='pwr-moderator-delivery/v1',scientific_limit_bytes=8*1024*1024),
    'pwr-stationary':dict(tests=('test_pwr_stationary.py','test_pwr_channel.py','test_pwr_local_cladding.py','test_coolant_channel.py',
            'test_eos_solver_precision.py','test_component_delivery.py','test_station_core.py','test_station_power.py'),
        cold_tests=('test_pwr_stationary.py','test_pwr_channel.py','test_pwr_local_cladding.py','test_eos_solver_precision.py','test_component_delivery.py'),
        module='scripts.qualify_pwr_stationary',imports=('coupling.pwr_stationary','coupling.pwr_channel','coupling.pwr_cladding'),
        format='pwr-stationary-delivery/v1',scientific_limit_bytes=8*1024*1024),
    'pwr-channel':dict(tests=('test_pwr_channel.py','test_pwr_local_cladding.py','test_coolant_channel.py',
            'test_pwr_distributed_fuel.py','test_conservative_exchange.py','test_eos_solver_precision.py',
            'test_component_delivery.py','test_station_core.py','test_station_power.py'),
        cold_tests=('test_pwr_channel.py','test_pwr_local_cladding.py','test_coolant_channel.py','test_eos_solver_precision.py','test_component_delivery.py'),
        module='scripts.qualify_pwr_channel',imports=('coupling.pwr_channel','coupling.pwr_cladding','coupling.coolant_channel'),format='pwr-channel-delivery/v1',scientific_limit_bytes=4*1024*1024),
    'local-cladding':dict(tests=TESTS,cold_tests=COLD_TESTS,module='scripts.qualify_pwr_local_cladding',
        imports=('coupling.pwr_cladding','coupling.pwr_fuel','atomic_model_station.core'),format='pwr-local-cladding-delivery/v1'),
    'coolant-channel':dict(tests=('test_coolant_channel.py','test_conservative_exchange.py','test_thermal_hydraulics.py','test_pwr_local_cladding.py','test_component_delivery.py'),
        cold_tests=('test_coolant_channel.py','test_conservative_exchange.py','test_thermal_hydraulics.py','test_component_delivery.py'),module='scripts.qualify_coolant_channel',
        imports=('coupling.coolant_channel','coupling.conservative','physics.hydraulics.network'),format='coolant-channel-delivery/v1'),
}


def read_scientific(path,component):
    if type(component) is not str or component not in PROFILES:raise ValueError('Registered component required')
    return unseal(read_json(path,limit=PROFILES[component].get('scientific_limit_bytes',2*1024*1024)))


def run(output,timeout_s=600,*,component='local-cladding'):
    if type(component) is not str or component not in PROFILES:raise ValueError('Registered component required')
    profile=PROFILES[component];tests_selected=profile['tests'];cold_selected=profile['cold_tests']
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    if out==ROOT or any(out==ROOT/x or ROOT/x in out.parents for x in ('src','scripts','tests','data','validation','scenarios','docs','configs','examples','requirements')):
        raise ValueError('Results must be outside source and evidence trees')
    if type(timeout_s) is not int or not 1<=timeout_s<=14400:raise ValueError('Phase timeout in 1..14400 s required')
    out.mkdir(parents=True);source=source_manifest(ROOT);evidence=evidence_manifest(ROOT);phases=[]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
    for key in ('PYTHONPATH','PYTHONHOME','PYTEST_ADDOPTS','PYTEST_PLUGINS'):env.pop(key,None)
    def phase(name,args,cwd=ROOT):
        with (out/(name+'.log')).open('xb') as log:
            try:code=subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout_s).returncode
            except subprocess.TimeoutExpired:code=124
        phases.append(dict(name=name,exit_code=code))
        atomic_json(out/'progress.json',seal(dict(phases=phases,completed=False)))
        print(name,code,flush=True)
        if code:raise RuntimeError(f'{name} failed: {code}; raw output retained')
    try:
        common=['-m','pytest','-q','-p','atomic_model_release.pytest_gate','--atomic-require-executed','--atomic-no-skips']
        phase('source-tests',[sys.executable,*common,'--atomic-receipt',out/'tests.json',*[ROOT/'tests'/t for t in tests_selected]])
        phase('source-qualification',[sys.executable,'-m',profile['module'],'--output',out/'source'])
        phase('build-source',[sys.executable,'-m','build','--sdist','--no-isolation','--outdir',out/'build'])
        archives=list((out/'build').glob('*.tar.gz'))
        if len(archives)!=1:raise ValueError('Exactly one sdist required')
        with tempfile.TemporaryDirectory(prefix='atomic-component-delivery-') as temporary:
            tmp=Path(temporary);clean=tmp/'clean';clean.mkdir()
            with tarfile.open(archives[0]) as archive:archive.extractall(clean,filter='data')
            roots=list(clean.iterdir())
            if len(roots)!=1 or not (roots[0]/'pyproject.toml').is_file():raise ValueError('Unexpected sdist layout')
            phase('build-wheel',[sys.executable,'-m','build','--wheel','--no-isolation','--outdir',out/'build'],roots[0])
            wheels=list((out/'build').glob('*.whl'))
            if len(wheels)!=1:raise ValueError('Exactly one wheel required')
            wheel=wheels[0];inventory=check_wheel(wheel)
            venv.EnvBuilder(with_pip=True,system_site_packages=True).create(tmp/'venv')
            python=tmp/('venv/Scripts/python.exe' if os.name=='nt' else 'venv/bin/python')
            work=tmp/'work';work.mkdir()
            phase('install',[python,'-m','pip','install','--no-index','--no-deps',wheel],work)
            phase('installed-origin',[python,'-I','-c',
                'import importlib; modules=[importlib.import_module(n) for n in '+repr(profile['imports'])+']; '
                'assert all("site-packages" in m.__file__ for m in modules); print(modules[0].__file__)'],work)
            for name in cold_selected:shutil.copyfile(ROOT/'tests'/name,work/name)
            phase('installed-tests',[python,'-I',*common,'--import-mode=importlib','--atomic-receipt',out/'installed-tests.json',*[work/t for t in cold_selected]],work)
            phase('installed-qualification',[python,'-I','-m',profile['module'],'--output',out/'installed'],work)
            first=read_scientific(out/'source/scientific.json',component);second=read_scientific(out/'installed/scientific.json',component)
            if first!=second:raise ValueError('Whole scientific JSON differs between source and installed wheel')
        if source!=source_manifest(ROOT) or not evidence_unchanged(evidence,ROOT):raise ValueError('Source/evidence changed during delivery')
        tests=read_json(out/'tests.json');cold=read_json(out/'installed-tests.json')
        if any(t['exit_code'] or t['counts']['failed'] or t['counts']['errors'] or t['counts']['skipped'] or not t['counts']['passed'] for t in (tests,cold)):
            raise ValueError('Complete scoped tests required')
        files=[out/'tests.json',out/'installed-tests.json',out/'source/report.json',out/'source/scientific.json',
            out/'installed/report.json',out/'installed/scientific.json',archives[0],wheel]
        result=dict(format=profile['format'],component=component,passed=True,source=source,evidence=evidence,phases=phases,
            source_tests=tests,installed_tests=cold,source_test_files=list(tests_selected),installed_test_files=list(cold_selected),
            qualification=first['report'],whole_scientific_json_equal=True,wheel_inventory=inventory,
            scientific_limit_bytes=profile.get('scientific_limit_bytes',2*1024*1024),
            artifacts={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            dependencies_reused=True,network_used=False,full_library_profile_executed=False,
            long_horizon_accepted=False,experimental_validation=False,closed_tasks=[])
        atomic_json(out/'report.json',seal(result));return result
    except Exception as exc:
        atomic_json(out/'failure.json',seal(dict(passed=False,phases=phases,error=str(exc),source=source)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase-timeout-s',type=int,default=600)
    parser.add_argument('--component',choices=tuple(PROFILES),required=True)
    args=parser.parse_args();result=run(args.output,args.phase_timeout_s,component=args.component)
    print(dict(passed=result['passed'],source_tests=result['source_tests']['counts'],installed_tests=result['installed_tests']['counts']))
