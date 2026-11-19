"""Physical L2 distances with absent temperature AND velocity modes retained."""
import numpy as np
from atomic_model_station_refined import compare_refined
from atomic_model_station_thermal import ThermalResult, ScalarMode


def compare_thermal(first,second,radius,weights):
    if type(first) is not ThermalResult or type(second) is not ThermalResult:
        raise ValueError('ThermalResult endpoints required')
    velocity=compare_refined(first.velocity,second.velocity,radius,weights,
        first_time=first.report['time'],second_time=second.report['time'])
    spectra=[]
    for result in (first,second):
        if type(result.temperature) is not list or not 1<=len(result.temperature)<=61:
            raise ValueError('Bounded canonical temperatures required')
        spectrum={}
        for model,q in result.temperature:
            if type(model) is not ScalarMode:
                raise ValueError('ScalarMode temperature required')
            key=(model.k,model.m)
            if key in spectrum or abs(model.k)>5 or not 0<=model.m<=5 or (model.m==0 and model.k<0):
                raise ValueError('Unique bounded canonical temperature modes required')
            if model.alpha!=result.velocity[0][0].alpha:
                raise ValueError('Temperature Fourier period differs from velocity')
            value=model.evaluate(q,radius)
            if key==(0,0) and np.max(abs(value.imag))>1e-13:
                raise ValueError('Real mean temperature required')
            spectrum[key]=value
        if set(spectrum)!={(m.k,m.m) for m,q in result.velocity}:
            raise ValueError('Matching canonical temperature and velocity spectra required')
        spectra.append(spectrum)
    a,b=spectra;w=np.asarray(weights,dtype=float)
    def squared(keys,difference):
        return sum((1 if key==(0,0) else 2)*float(np.sum(w*abs(difference(key))**2)) for key in sorted(keys))
    common=squared(set(a)&set(b),lambda key:a[key]-b[key])
    only_first=squared(set(a)-set(b),lambda key:a[key])
    only_second=squared(set(b)-set(a),lambda key:b[key])
    total=common+only_first+only_second
    if not np.isfinite(total):raise ValueError('Nonfinite thermal comparison norm')
    return dict(velocity_l2=velocity['velocity_l2'],temperature_l2=float(np.sqrt(total)),
        temperature_common_l2=float(np.sqrt(common)),temperature_only_first_l2=float(np.sqrt(only_first)),
        temperature_only_second_l2=float(np.sqrt(only_second)),absent_modes_included=True,
        norm='sqrt(sum signed modes integral |delta field|^2 r dr)',
        first_time=first.report['time'],second_time=second.report['time'])
