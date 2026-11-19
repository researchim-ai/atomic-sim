"""Correlation-aware acceptance of a time window, never an IID assumption."""
from dataclasses import dataclass,asdict
import math
import numpy as np
from scipy.stats import t as student_t


@dataclass(frozen=True)
class WindowPolicy:
    warmup:float=100.
    block_duration:float=10.
    minimum_blocks:int=8
    minimum_effective_samples:float=50.
    maximum_relative_half_width:float=.05
    maximum_relative_half_shift:float=.05
    block_to_correlation_ratio:float=5.

    def __post_init__(self):
        for name in ('warmup','block_duration','minimum_effective_samples','maximum_relative_half_width','maximum_relative_half_shift','block_to_correlation_ratio'):
            v=getattr(self,name)
            if type(v) not in (int,float) or not math.isfinite(v) or v<0 or (name!='warmup' and v==0):raise ValueError('Finite positive statistical policy: '+name)
        if type(self.minimum_blocks) is not int or not 4<=self.minimum_blocks<=1000:raise ValueError('At least four independent blocks required')


def correlation_time(values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or not 16<=len(x)<=1000000 or not np.all(np.isfinite(x)):raise ValueError('Finite correlation sample16..1000000 required')
    y=x-np.mean(x);variance=float(y@y)
    if variance<=np.finfo(float).eps**2*max(1,float(x@x)):return None
    size=1<<(2*len(y)-1).bit_length();f=np.fft.rfft(y,size)
    rho=np.fft.irfft(f*np.conj(f),size)[:len(y)]/variance
    # Geyer initial positive, monotone paired sequence, including rho0.
    pairs=[];previous=float('inf')
    for i in range(0,len(y)//2-1,2):
        pair=float(rho[i]+rho[i+1])
        if pair<=0:break
        previous=min(previous,pair);pairs.append(previous)
    return max(1.,-1+2*sum(pairs))


def analyse_window(times,observables,policy=None):
    policy=WindowPolicy() if policy is None else policy
    if type(policy) is not WindowPolicy:raise ValueError('WindowPolicy required')
    t=np.asarray(times,dtype=float)
    if t.ndim!=1 or not 2<=len(t)<=1000000 or not np.all(np.isfinite(t)) or t[0]<0 or np.any(np.diff(t)<=0):raise ValueError('Finite increasing time series required')
    dt=float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t),dt,rtol=1e-8,atol=1e-12):raise ValueError('Uniform sampling required for this correlation estimator')
    if type(observables) is not dict or not 1<=len(observables)<=16:raise ValueError('Explicit bounded observable mapping required')
    if any(type(name) is not str or not 1<=len(name)<=80 for name in observables):raise ValueError('Nonempty bounded observable names required')
    mask=t>=policy.warmup;post=t[mask];block_size=max(2,math.ceil(policy.block_duration/dt)+1);results={}
    global_reasons=[]
    if t[-1]<policy.warmup:global_reasons.append('warmup_not_completed')
    for name,values in observables.items():
        x=np.asarray(values,dtype=float)
        if x.shape!=t.shape or not np.all(np.isfinite(x)):raise ValueError('Complete finite observable: '+str(name))
        y=x[mask];count=len(y)//block_size;reasons=list(global_reasons)
        if count<policy.minimum_blocks:reasons.append('insufficient_complete_blocks')
        used=y[:count*block_size] if count else y
        mean=float(np.mean(used)) if len(used) else None
        tau=correlation_time(used) if len(used)>=16 else None
        ess=float(len(used)/tau) if tau is not None else None
        if tau is None:reasons.append('correlation_not_identifiable')
        if ess is None or ess<policy.minimum_effective_samples:reasons.append('insufficient_effective_samples')
        tau_time=tau*dt if tau is not None else None
        if tau_time is not None and (block_size-1)*dt<policy.block_to_correlation_ratio*tau_time:reasons.append('blocks_shorter_than_correlation_requirement')
        ci=None;relative_width=None;shift=None
        if count>=2:
            blocks=y[:count*block_size].reshape(count,block_size).mean(axis=1)
            width=float(student_t.ppf(.975,count-1)*np.std(blocks,ddof=1)/np.sqrt(count))
            scale=max(abs(mean),float(np.std(used)),1e-12);relative_width=width/scale;ci=[mean-width,mean+width]
            if relative_width>policy.maximum_relative_half_width:reasons.append('confidence_interval_too_wide')
            half=len(used)//2;shift=abs(float(np.mean(used[:half])-np.mean(used[half:])))/scale
            if shift>policy.maximum_relative_half_shift:reasons.append('half_window_drift')
        else:reasons.append('confidence_interval_not_identifiable')
        results[name]=dict(qualified=not reasons,reasons=reasons,post_warmup_samples=len(y),used_samples=len(used),complete_blocks=count,
            mean=mean,confidence_interval_95=ci,relative_half_width=relative_width,relative_half_shift=shift,
            correlation_samples=tau,correlation_time=tau_time,effective_samples=ess)
    return dict(qualified=all(v['qualified'] for v in results.values()),policy=asdict(policy),time_start=float(t[0]),time_end=float(t[-1]),
        sample_interval=dt,block_samples=block_size,observables=results,
        scope='Window diagnostics with estimated correlation; not proof of physical turbulence or model validity')
