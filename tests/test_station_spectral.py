from pathlib import Path
import json
import numpy as np
import pytest
from scipy.io import netcdf_file
from atomic_model_station_spectral import layout,field_energy,divergence,read_state,audit_state,compare_states,WindowPolicy,analyse_window,correlation_time
from atomic_model_station_quadrature.data import load_fixture as quadrature


def grid():
    row=quadrature()['rules']['64'];return np.array(row['radius']),np.array(row['corrected_weights'])


def checkpoint(path,r=None,time=0.,kcount=2,mcount=2):
    r=grid()[0] if r is None else np.asarray(r);k,m,f=layout(kcount,mcount);n=len(r)
    with netcdf_file(path,'w') as out:
        for key,value in dict(t=time,Re=5300.,Pr=.7,C=0.,alpha=.628319).items():setattr(out,key,np.float64(value))
        for name,size in [('ReIm',2),('H',len(k)),('r',n)]:out.createDimension(name,size)
        out.createVariable('r','d',('r',))[:]=r
        for name in ('Ur','Ut','Uz','T'):
            v=out.createVariable(name,'d',('ReIm','H','r'));v.K=np.int32(kcount);v.M=np.int32(mcount);v.Mp=np.int32(1);v[:]=0.
    return k,m


@pytest.mark.parametrize('kcount,mcount,mp',[(1,1,1),(2,2,1),(3,4,2),(80,76,1),(96,96,1)])
def test_half_plane_layout_and_conjugate_multiplicity(kcount,mcount,mp):
    k,m,w=layout(kcount,mcount,mp)
    assert len(k)==(2*kcount-1)*mcount-kcount+1
    assert len(set(zip(k,m)))==len(k) and list(zip(k[:kcount],m[:kcount]))==[(i,0) for i in range(kcount)]
    assert w[0]==1 and np.all(w[1:]==2) and np.all(m%mp==0)


@pytest.mark.parametrize('params',[(True,2,1),(0,2,1),(2,129,1),(2,2,0),(2.,2,1)])
def test_invalid_fourier_shape_is_not_interpreted_as_mean_mode(params):
    with pytest.raises(ValueError):layout(*params)


def test_parseval_for_a_real_cosine_and_a_mean():
    r,w=grid();k,m,factor=layout(2,2);a=np.zeros((len(k),len(r)),complex);a[0]=3.;a[1]=2./2
    e=field_energy(a,w,k,m,factor)
    assert e['total']==pytest.approx(3**2/2+2**2/4,abs=1e-13)
    assert e['axial_nonzero']==pytest.approx(1.) and e['azimuthal_nonzero']==0


def test_manufactured_regular_divergence_free_modes(tmp_path):
    r,w=grid();path=tmp_path/'modes.cdf';k,m=checkpoint(path)
    with netcdf_file(path,'a') as f:
        f.variables['Ur'][1,1,:]=.001*.628319*r*(1-r*r)**2
        f.variables['Uz'][0,1,:]=.001*(-2+8*r*r-6*r**4)
        index=np.flatnonzero((k==0)&(m==1))[0];f.variables['Uz'][0,index,:]=.002*r*(1-r*r)
    audit=audit_state(path,w)
    assert audit['divergence_l2']<1e-8 and audit['spatial_modes_present']
    assert audit['no_slip_mode_maximum']<1e-12
    assert audit['area_mean_velocity']==pytest.approx(.5,abs=1e-12)
    assert not audit['turbulent_statistics_qualified'] and not audit['validated']


def test_a_mean_only_checkpoint_cannot_be_called_spatially_turbulent(tmp_path):
    p=tmp_path/'mean.cdf';checkpoint(p,kcount=1,mcount=1);a=audit_state(p,grid()[1])
    assert not a['spatial_modes_present'] and a['axial_tail']['terminal_amplitude_ratio'] is None


@pytest.mark.parametrize('bad',['time_nan','pr_negative','alpha_zero','shape','mode_float','imaginary_mean','nan_field','radial_order'])
def test_invalid_checkpoint_is_rejected(tmp_path,bad):
    p=tmp_path/'bad.cdf';checkpoint(p)
    with netcdf_file(p,'a') as f:
        if bad=='time_nan':f.t=np.float64(np.nan)
        elif bad=='pr_negative':f.Pr=np.float64(-1.)
        elif bad=='alpha_zero':f.alpha=np.float64(0.)
        elif bad=='shape':
            for name in ('Ur','Ut','Uz','T'):f.variables[name].K=np.int32(3)
        elif bad=='mode_float':f.variables['T'].M=np.float64(2.)
        elif bad=='imaginary_mean':f.variables['Uz'][1,0,0]=1
        elif bad=='nan_field':f.variables['T'][0,0,0]=np.nan
        elif bad=='radial_order':f.variables['r'][:]=grid()[0][::-1]
    with pytest.raises(ValueError):read_state(p)


def test_large_checkpoint_is_refused_before_loading(tmp_path):
    p=tmp_path/'large.cdf'
    with p.open('wb') as f:f.truncate(200000001)
    with pytest.raises(ValueError,match='Bounded'):read_state(p)


def test_comparison_accepts_roundoff_epoch_but_refuses_a_different_state_time(tmp_path):
    a=tmp_path/'a.cdf';b=tmp_path/'b.cdf';checkpoint(a,time=.2);checkpoint(b,time=.2+1e-16)
    assert compare_states(a,b,grid()[1])['physical_fields_bitwise_equal']
    with netcdf_file(b,'a') as f:f.t=np.float64(.3)
    with pytest.raises(ValueError,match='epoch'):compare_states(a,b,grid()[1])


def test_radial_comparison_interpolates_without_claiming_bitwise_restart(tmp_path):
    a=tmp_path/'a.cdf';b=tmp_path/'b.cdf';r,w=grid();fine=np.linspace(.001,1,96)
    checkpoint(a,r);checkpoint(b,fine)
    for p,x in ((a,r),(b,fine)):
        with netcdf_file(p,'a') as f:f.variables['Uz'][0,0,:]=1-x*x
    report=compare_states(a,b,w)
    assert report['maximum_rms']<1e-12 and not report['physical_fields_bitwise_equal']
    assert 'cubic' in report['radial_interpolation']


def test_stationary_iid_window_can_pass_with_uncertainty():
    rng=np.random.default_rng(42);t=np.arange(8192,dtype=float);x=10+.1*rng.normal(size=len(t))
    p=analyse_window(t,dict(signal=x),WindowPolicy(warmup=100,block_duration=100))
    s=p['observables']['signal'];assert p['qualified'] and s['effective_samples']>2000
    assert s['confidence_interval_95'][0]<s['mean']<s['confidence_interval_95'][1]


def test_correlated_data_have_fewer_effective_samples_than_rows():
    rng=np.random.default_rng(7);x=np.zeros(20000)
    for i in range(1,len(x)):x[i]=.9*x[i-1]+rng.normal()
    tau=correlation_time(x[1000:]);assert 8<tau<40 and len(x)/tau<len(x)/8


def test_constant_signal_is_not_a_zero_uncertainty_certificate():
    p=analyse_window(np.arange(2048),dict(constant=np.ones(2048)),WindowPolicy(warmup=0))
    assert not p['qualified'] and 'correlation_not_identifiable' in p['observables']['constant']['reasons']


def test_drift_is_rejected_even_with_many_samples():
    t=np.arange(4096);p=analyse_window(t,dict(drift=10+4*t/t[-1]),WindowPolicy(warmup=0,block_duration=100))
    assert not p['qualified'] and 'half_window_drift' in p['observables']['drift']['reasons']


def test_short_thermal_spinup_has_no_fabricated_confidence_interval():
    p=analyse_window(np.arange(20)*.01,dict(heat=np.linspace(1,2,20)))
    s=p['observables']['heat'];assert not p['qualified'] and 'warmup_not_completed' in s['reasons']
    assert s['confidence_interval_95'] is None and s['effective_samples'] is None
    json.dumps(p,allow_nan=False)


@pytest.mark.parametrize('bad',['time_duplicate','time_gap','nonfinite','wrong_length','empty_name'])
def test_invalid_statistical_records_are_not_silently_filled(bad):
    t=np.arange(20,dtype=float);x=np.ones(20);name='signal'
    if bad=='time_duplicate':t[1]=0
    elif bad=='time_gap':t[1]=.5
    elif bad=='nonfinite':x[0]=np.nan
    elif bad=='wrong_length':x=x[:-1]
    elif bad=='empty_name':name=''
    with pytest.raises(ValueError):analyse_window(t,{name:x})


@pytest.mark.parametrize('kwargs',[dict(warmup=-1),dict(block_duration=0),dict(minimum_blocks=2),dict(minimum_blocks=True),dict(minimum_effective_samples=np.inf)])
def test_statistical_policy_requires_explicit_finite_limits(kwargs):
    with pytest.raises(ValueError):WindowPolicy(**kwargs)


def test_historical_sources_allow_only_evidenced_exact_cache_migration():
    from scripts.qualify_station_inventory import read
    from scripts.qualify_station_shutdown import identity as station
    from scripts.qualify_station_mixed_convection import identity as mixed
    from scripts.qualify_station_pipe import identity as pipe
    from scripts.qualify_station_turbulence import identity as turbulent
    from scripts.qualify_station_heated import identity as heated
    from scripts.qualify_station_quadrature import identity as quadrature_identity
    root=Path(__file__).resolve().parents[1]/'docs/benchmarks'
    for name,identity in [('station_low_flow_v63',station),('station_mixed_convection_v64',mixed),('station_pipe_v65',pipe),('station_turbulence_v66',turbulent),('station_heated_v67',heated),('station_quadrature_v68',quadrature_identity)]:
        from tests._station_history import assert_source_compatible
        assert_source_compatible(read(root/(name+'.json'))['source'],identity())
