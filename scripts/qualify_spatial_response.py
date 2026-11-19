"""Frozen direct/adjoint slab response cases and independent dense perturbations."""
from copy import deepcopy
from pathlib import Path
import argparse
import numpy as np
from atomic_model_spatial import Model
from atomic_model_spatial.cases import slab
from atomic_model_spatial.reference import dense_k
from atomic_model_spatial_response import sensitivities,station_core_response
from atomic_model_spatial_response.model import identity
from atomic_model_station import PWRCoreSession,CoreConfig
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json


def cases():
    heterogeneous=slab(7,heterogeneous=True,boundaries=('reflecting','zero_flux'))
    heterogeneous['faces_m']=(np.linspace(0,1,8)**1.3).tolist()
    return [('homogeneous',slab(4)),('heterogeneous',heterogeneous),
        ('reflecting',slab(4,boundaries=('reflecting','reflecting'))),
        ('single-cell',slab(1,boundaries=('reflecting','reflecting')))]


def qualify_case(config):
    model=Model(config);response=sensitivities(model,tolerance=1e-12)
    rows=[];directions={};passed=True
    for field,gradient in response['gradients'].items():
        values=np.asarray(config[field]);gradient=np.asarray(gradient)
        directions[field]=values*.1*np.cos(np.arange(values.size).reshape(values.shape)+.3)
        for index in np.ndindex(values.shape):
            if field=='scatter_per_m' and index[0]==index[1]:continue
            delta=max(values[index],.001)*2e-5;plus=deepcopy(config);minus=deepcopy(config)
            a=values.copy();b=values.copy();a[index]+=delta;b[index]-=delta
            plus[field]=a.tolist();minus[field]=b.tolist()
            derivative=float((dense_k(Model(plus))-dense_k(Model(minus)))/(2*delta))
            error=abs(float(gradient[index])-derivative)
            accepted=error<=1e-7+2e-5*max(abs(float(gradient[index])),abs(derivative))
            rows.append(dict(field=field,index=list(index),analytic=float(gradient[index]),
                dense_difference=derivative,absolute_error=error,passed=accepted));passed=passed and accepted
    slope=sum(float(np.sum(np.asarray(response['gradients'][field])*direction)) for field,direction in directions.items())
    taylor=[]
    for epsilon in (.04,.02,.01):
        perturbed=deepcopy(config)
        for field,direction in directions.items():perturbed[field]=(np.asarray(config[field])+epsilon*direction).tolist()
        actual=dense_k(Model(perturbed));predicted=response['k']+epsilon*slope
        taylor.append(dict(epsilon=epsilon,k=actual,linear_prediction=predicted,absolute_error=abs(actual-predicted)))
    orders=[float(np.log(a['absolute_error']/b['absolute_error'])/np.log(2))
        for a,b in zip(taylor,taylor[1:]) if min(a['absolute_error'],b['absolute_error'])>1e-13]
    checks=dict(local_derivatives=passed,k_dense=abs(response['k']-dense_k(model))<=1e-10,
        taylor_second_order=bool(orders) and min(orders)>=1.8,
        final_taylor_error=taylor[-1]['absolute_error']/response['k']<=1e-5,
        birth_scaling=abs(np.sum(response['elasticities']['nu_fission_per_m'])-1.)<=1e-10,
        loss_scaling=abs(sum(np.sum(response['elasticities'][f]) for f in ('absorption_per_m','scatter_per_m','diffusion_m'))+1.)<=1e-10)
    checks={key:bool(value) for key,value in checks.items()}
    return dict(config=config,response=response,derivative_checks=rows,taylor=taylor,taylor_orders=orders,checks=checks,passed=all(checks.values()))


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    source=identity();driver=sha(__file__);out.mkdir(parents=True);results={};failures={}
    for name,config in cases():
        try:
            result=qualify_case(config);atomic_json(out/(name+'.json'),seal(result));results[name]=result
        except Exception as error:
            failures[name]=dict(type=type(error).__name__,message=str(error))
            atomic_json(out/(name+'.json'),seal(dict(passed=False,failure=failures[name])))
    checks=dict(all_cases=len(results)==4 and all(r['passed'] for r in results.values()))
    try:
        owner=PWRCoreSession(core=CoreConfig(kind='diffusion',cells=4),rtol=1e-10)
        before=owner.checkpoint();station=station_core_response(owner.neutron_core)
        checks['station_static_response']=abs(station['k']-1.)<=1e-10 and station['relative_power_projection_defect']<=1e-12
        checks['station_unchanged']=owner.checkpoint()==before
        atomic_json(out/'station.json',seal(station))
    except Exception as error:failures['station']=dict(type=type(error).__name__,message=str(error))
    checks['source_unchanged']=source==identity() and driver==sha(__file__)
    report=dict(source=source,driver_sha256=driver,checks=checks,failures=failures,
        passed=not failures and len(checks)==4 and all(checks.values()),attempted_cases=4,
        passed_cases=sum(r['passed'] for r in results.values()),
        local_derivatives_checked=sum(len(r['derivative_checks']) for r in results.values()),
        minimum_observed_taylor_order=min((min(r['taylor_orders']) for r in results.values() if r['taylor_orders']),default=None),
        physical_data_family='synthetic',experimental_validation=False,full_r2_accepted=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Spatial response qualification failed; all attempts retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output)['checks'])
