"""Expanded-spectrum physical L2 comparison, including all absent modes."""
import numpy as np
from atomic_model_station_regular import RegularMode
from atomic_model_station_regular.model import _finite_real
from .products import signed_modes


def compare_refined(first,second,radius,weights,*,first_time,second_time):
    for time in (first_time,second_time):_finite_real(time,'comparison epoch',0.,1e6)
    slack=max(1e-12,8*float(np.spacing(max(abs(first_time),abs(second_time)))))
    if abs(first_time-second_time)>slack:raise ValueError('Same refined comparison epoch required')
    for states in (first,second):
        if type(states) not in (list,tuple) or not 1<=len(states)<=61 or any(type(model) is not RegularMode for model,q in states):raise ValueError('Bounded refined velocity states required')
    alpha=first[0][0].alpha
    a,_=signed_modes(first,radius,alpha);b,_=signed_modes(second,radius,alpha)
    r=np.asarray(radius);w=np.asarray(weights,dtype=float)
    if w.shape!=r.shape or not np.all(np.isfinite(w)) or np.any(w<=0) or abs(w.sum()-.5)>1e-12:raise ValueError('Positive L2(r dr) comparison quadrature required')
    def square(value):return float(np.sum(abs(value)**2*w))
    common=sum(square(a[key]-b[key]) for key in sorted(set(a)&set(b)))
    only_a=sum(square(a[key]) for key in sorted(set(a)-set(b)));only_b=sum(square(b[key]) for key in sorted(set(b)-set(a)))
    total=common+only_a+only_b
    if not np.isfinite(total):raise ValueError('Nonfinite refined comparison norm')
    return dict(velocity_l2=float(np.sqrt(total)),common_l2=float(np.sqrt(common)),only_first_l2=float(np.sqrt(only_a)),only_second_l2=float(np.sqrt(only_b)),
        first_signed_modes=len(a),second_signed_modes=len(b),omitted_modes_included=True,
        norm='sqrt(sum signed modes integral|delta u|^2 r dr)',first_time=float(first_time),second_time=float(second_time))
