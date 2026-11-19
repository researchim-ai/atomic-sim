import numpy as np
import pytest
from atomic_model_station_regular import regular_mode
from atomic_model_station_nonlinear import NonlinearSession
from atomic_model_station_stokes_pressure import PressureField
from atomic_model_station_wall import audit_momentum
from atomic_model_station_resolution import compare_states,audit_nonlinear_momentum


def mean():
    m=regular_mode(0,0,.628319);return m,np.zeros(m.dimension)


def axial(m,order=4):
    model=regular_mode(0,m,.628319,order=order);u=np.zeros((3,len(model.radius)),complex);u[2]=model.radius**m*(1-model.radius**2)
    return model,model.fit(u)


def compare(a,b,**kwargs):
    m=mean()[0];args=dict(first_time=.1,second_time=.1);args.update(kwargs)
    return compare_states(a,b,m.radius,m.weights,**args)


def test_missing_modes_cannot_disappear_from_comparison():
    a=[mean()];b=[mean(),axial(1)];r=compare(a,b)
    assert r['common_rms']<1e-12
    assert r['only_second_rms']==pytest.approx(np.sqrt(1/12),abs=1e-12)
    assert r['total_rms']==pytest.approx(np.sqrt(1/12),abs=1e-12)
    assert r['omitted_modes_included']


def test_both_missing_spectra_and_orthogonal_norm_decomposition():
    r=compare([mean(),axial(1)],[mean(),axial(2)])
    assert r['total_rms']**2==pytest.approx(1/12+1/30,abs=1e-12)
    assert r['total_rms']**2==pytest.approx(r['common_rms']**2+r['only_first_rms']**2+r['only_second_rms']**2)


def test_same_physical_field_in_different_bases_compares_equal():
    r=compare([mean(),axial(1,3)],[mean(),axial(1,7)])
    assert r['total_rms']<1e-12


def test_comparison_requires_matching_period_and_epoch():
    with pytest.raises(ValueError,match='epoch'):compare([mean()],[mean()],second_time=.2)
    with pytest.raises(ValueError,match='epoch'):compare([mean()],[mean()],first_time=1e12,second_time=1e12+.1)
    other=regular_mode(0,0,.7)
    with pytest.raises(ValueError,match='period'):compare([mean()],[(other,np.zeros(other.dimension))])
    assert compare([mean()],[mean()],first_time=.1,second_time=.1+1e-16)['total_rms']==0


def test_nonlinear_centrifugal_momentum_needs_advection_term():
    model=regular_mode(0,0,.628319,order=6);r=model.radius;u=np.zeros((3,len(r)),complex);u[1]=r*(1-r*r)
    q=model.fit(u);s=NonlinearSession([(model,q)],viscosity=0);before=s.modes;_,p=s.advance(.01)
    points=np.linspace(.01,1,51)
    a=audit_nonlinear_momentum(before,s.modes,p,dt=.01,viscosity=0,radius=points)
    assert a['maximum_retained_residual']<1e-11 and a['retained_wall_passed']
    old=audit_momentum(model,q,s.modes[0][1],p[(0,0)],dt=.01,viscosity=0,radius=points,force=np.zeros((3,len(points))))
    assert old['maximum_momentum_residual']>.1
    assert a['nonlinear_advection_included'] and not a['unretained_pressure_reconstructed']


def test_unretained_products_are_reported_without_claiming_pressure_balanced_error():
    model=regular_mode(1,1,.628319,order=3);q=np.arange(model.dimension)*(.001+.0003j)
    s=NonlinearSession([mean(),(model,q)])
    # Match the common64-point quadrature and preserve a genuinely interacting mode.
    before=s.modes;_,p=s.advance(.005)
    a=audit_nonlinear_momentum(before,s.modes,p,dt=.005,viscosity=.1,radius=np.linspace(.01,1,31))
    assert a['unretained_product_modes']>0 and a['unretained_nonlinear_maximum']>1e-8
    assert not a['unretained_pressure_reconstructed'] and not a['station_usable']


@pytest.mark.parametrize('kind',['axis','wall','force','force_shape','force_nan','pressure_keys','pressure_basis','velocity_basis','dt','viscosity'])
def test_invalid_nonlinear_audit_is_rejected(kind):
    m,q=mean();states=[(m,q)];p={(0,0):PressureField(m,np.zeros(m.order+1))};r=np.linspace(.01,1,11)
    args=dict(dt=.01,viscosity=.1,radius=r);after=states
    if kind=='axis':r[0]=0
    if kind=='wall':r[-1]=.99
    if kind=='force':args['force']={}
    if kind=='force_shape':args['force']={(0,0):np.zeros(3*m.order)}
    if kind=='force_nan':args['force']={(0,0):np.full((3,len(r)),np.nan)}
    if kind=='pressure_keys':p={}
    if kind=='pressure_basis':p={(0,0):PressureField(regular_mode(0,0,.7),np.zeros(m.order+1))}
    if kind=='velocity_basis':other=regular_mode(0,0,.628319,order=5);after=[(other,np.zeros(other.dimension))]
    if kind=='dt':args['dt']=0
    if kind=='viscosity':args['viscosity']=-1
    with pytest.raises(ValueError):audit_nonlinear_momentum(states,after,p,**args)


def test_bad_comparison_quadrature_and_duplicate_modes_are_refused():
    m,q=mean();w=m.weights.copy();w[0]=-w[0]
    with pytest.raises(ValueError):compare_states([(m,q)],[(m,q)],m.radius,w,first_time=0,second_time=0)
    with pytest.raises(ValueError):compare([(m,q),(m,q)],[(m,q)])
