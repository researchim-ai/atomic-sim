"""Qualify cylindrical nonlinear kernels, padding and energy; no time integration."""
import argparse,platform
from pathlib import Path
import numpy as np
import scipy
from atomic_model_station_regular import regular_mode
from atomic_model_station_stokes_pressure import recover_pressure
from atomic_model_station_advection import signed_spectrum,convolution_advection,fft_advection,nonlinear_power
from atomic_model_multiphysics.provenance import sha,seal
from atomic_model_training.spec import atomic_json


def identity():
    import atomic_model_station_advection as package
    from scripts.qualify_station_wall import identity as base_identity
    root=Path(package.__file__).parent
    return dict(files={p.name:sha(p) for p in sorted(root.glob('*.py'))},driver=sha(__file__),base=base_identity(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)


def modes(extent,order,quadrature):
    result=[];rng=np.random.default_rng(75)
    for m in range(extent+1):
        for k in (range(extent+1) if m==0 else range(-extent,extent+1)):
            model=regular_mode(k,m,.628319,order=order,quadrature=quadrature)
            if k==m==0:
                field=np.zeros((3,len(model.radius)),complex);field[1]=.001*model.radius*(1-model.radius**2);q=model.fit(field)
            else:q=.001*(rng.normal(size=model.dimension)+1j*rng.normal(size=model.dimension))
            result.append((model,q))
    return result


def run(output,reference=None):
    out=Path(output).absolute();source=identity()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True)
    criteria=dict(coefficient_error=1e-11,energy_power=1e-11,projected_power_error=1e-11,
        minimum_modal_transfer=1e-12,minimum_alias_error=1e-6,minimum_alias_power=1e-9,swirl_pressure_error=1e-10,swirl_momentum_error=1e-9)
    atomic_json(out/'protocol.json',seal(dict(source=source,criteria=criteria,
        model='Real Hermitian Fourier field; positive radial quadrature; analytic regular radial derivatives',
        fft_rule='Nz>2*K+Kt and Ntheta>2*M+Mt for requested targets',
        scope='Nonlinear spatial kernels and stationary centrifugal balance, not a nonlinear time integrator',
        native_backend_changed=False)))
    cases={};passed=True;aliased=None
    for extent,order,quadrature in [(1,3,32),(1,5,64),(2,3,32),(2,4,64),(2,5,96),(2,6,96)]:
        state=modes(extent,order,quadrature);model=state[0][0];r=model.radius;w=model.weights
        velocity,radial=signed_spectrum(state,r,.628319)
        direct=convolution_advection(r,.628319,velocity,radial);full,full_proof=fft_advection(r,.628319,velocity,radial)
        targets=[(k,m) for k in range(-extent,extent+1) for m in range(-extent,extent+1)]
        retained,retained_proof=fft_advection(r,.628319,velocity,radial,targets)
        error=max(float(np.max(abs(direct[key]-full[key]))) for key in direct)
        retained_error=max(float(np.max(abs(direct[key]-retained[key]))) for key in retained)
        power=nonlinear_power(velocity,direct,w);transfers={};load_power=0.
        for mode,q in state:
            key=(mode.k,mode.m);value=float(np.real(np.vdot(mode.coefficients(q),-mode.load(retained[key]))))*(1 if key==(0,0) else 2)
            transfers[str(key)]=value;load_power+=value
        projected_error=abs(load_power+power['advection_power']);maximum_transfer=max(abs(v) for v in transfers.values())
        checks=dict(full_coefficients=error<criteria['coefficient_error'],retained_coefficients=retained_error<criteria['coefficient_error'],
            energy=power['absolute_advection_power']<criteria['energy_power'],projection=projected_error<criteria['projected_power_error'],nontrivial_transfer=maximum_transfer>criteria['minimum_modal_transfer'])
        passed=bool(passed and all(checks.values()))
        cases[f'extent{extent}-order{order}-q{quadrature}']=dict(checks=checks,input_modes=len(velocity),output_modes=len(direct),
            full_fft=full_proof,retained_fft=retained_proof,maximum_coefficient_error=error,retained_coefficient_error=retained_error,power=power,
            projected_power_error=projected_error,modal_rhs_power=transfers,maximum_modal_transfer=maximum_transfer)
        if extent==2 and order==4:
            refused=False
            try:fft_advection(r,.628319,velocity,radial,targets,grid=(5,5))
            except ValueError:refused=True
            bad,proof=fft_advection(r,.628319,velocity,radial,targets,grid=(5,5),allow_aliasing=True)
            aliased=dict(refused_by_default=refused,diagnostic=proof,maximum_error=max(float(np.max(abs(direct[key]-bad[key]))) for key in retained),power=nonlinear_power(velocity,bad,w))
    atomic_json(out/'cases.json',seal(cases))
    model=regular_mode(0,0,.628319,order=6);r=model.radius;field=np.zeros((3,len(r)),complex);field[1]=r*(1-r*r)
    q=model.fit(field);velocity,radial=signed_spectrum([(model,q)],r,.628319);nonlinear=convolution_advection(r,.628319,velocity,radial)[(0,0)]
    pressure,proof=recover_pressure(model,model.load(-nonlinear));p,gradient=pressure.evaluate(r)
    expected=r*r/2-r**4/2+r**6/6-1/8
    swirl=dict(pressure_error=float(np.max(abs(p-expected))),momentum_error=float(np.max(abs(gradient+nonlinear))),
        expected_pressure='r^2/2-r^4/2+r^6/6-1/8',scope='Stationary inviscid centrifugal equilibrium only',pressure_recovery=proof)
    passed=bool(passed and aliased['refused_by_default'] and aliased['maximum_error']>criteria['minimum_alias_error'] and aliased['power']['absolute_advection_power']>criteria['minimum_alias_power']
        and swirl['pressure_error']<criteria['swirl_pressure_error'] and swirl['momentum_error']<criteria['swirl_momentum_error'])
    atomic_json(out/'physics.json',seal(dict(alias_counterexample=aliased,centrifugal_swirl=swirl)))
    if source!=identity():raise ValueError('Advection source changed')
    report=dict(passed=passed,source=source,cases=len(cases),nonlinear_spatial_kernel_qualified=passed,
        maximum_fft_coefficient_error=max(v['maximum_coefficient_error'] for v in cases.values()),
        maximum_energy_power=max(v['power']['absolute_advection_power'] for v in cases.values()),maximum_projected_power_error=max(v['projected_power_error'] for v in cases.values()),
        alias_default_refusal=aliased['refused_by_default'],alias_coefficient_error=aliased['maximum_error'],alias_energy_power=aliased['power']['absolute_advection_power'],
        stationary_swirl_pressure_error=swirl['pressure_error'],stationary_swirl_momentum_error=swirl['momentum_error'],
        nonlinear_time_integrator_qualified=False,thermal_model_supported=False,native_backend_changed=False,station_usable=False,validated=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not passed:raise ValueError('Nonlinear kernel qualification failed; evidence retained')
    if reference and {p.name:sha(p) for p in Path(reference).glob('*.json')}!={p.name:sha(p) for p in out.glob('*.json')}:raise ValueError('Installed nonlinear scientific JSON differ')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();print(run(a.output,a.reference)['passed'])

if __name__=='__main__':main()
