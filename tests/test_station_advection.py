import numpy as np
import pytest
from atomic_model_station_regular import regular_mode
from atomic_model_station_stokes_pressure import recover_pressure
from atomic_model_station_advection import signed_spectrum,convolution_advection,fft_advection,nonlinear_power


def modes_for_test(extent=1,order=4,quadrature=64):
    modes=[];rng=np.random.default_rng(75)
    for m in range(extent+1):
        for k in (range(extent+1) if m==0 else range(-extent,extent+1)):
            model=regular_mode(k,m,.628319,order=order,quadrature=quadrature)
            if k==m==0:
                u=np.zeros((3,len(model.radius)),complex);u[1]=.001*model.radius*(1-model.radius**2);q=model.fit(u)
            else:q=.001*(rng.normal(size=model.dimension)+1j*rng.normal(size=model.dimension))
            modes.append((model,q))
    return modes


def test_swirl_centrifugal_term_and_independent_pressure_equilibrium():
    model=regular_mode(0,0,.628319,order=6);r=model.radius;field=np.zeros((3,len(r)),complex);field[1]=r*(1-r*r)
    q=model.fit(field);v,d=signed_spectrum([(model,q)],r,.628319)
    n=convolution_advection(r,.628319,v,d)
    np.testing.assert_allclose(n[(0,0)][0],-r*(1-r*r)**2,atol=1e-12)
    np.testing.assert_allclose(n[(0,0)][1:],0,atol=1e-12)
    pressure,proof=recover_pressure(model,model.load(-n[(0,0)]));p,g=pressure.evaluate(r)
    expected=r*r/2-r**4/2+r**6/6-1/8
    np.testing.assert_allclose(p,expected,atol=1e-12)
    np.testing.assert_allclose(g+n[(0,0)],0,atol=1e-11)


def test_vector_geometry_against_cartesian_finite_difference_advection():
    model=regular_mode(1,1,.628319,order=4);q=np.arange(model.dimension)*(.001+.0003j)
    r=np.array([.5,.6]);v,d=signed_spectrum([(model,q)],r,.628319);n=convolution_advection(r,.628319,v,d)
    theta=.43;z=.27;x=.5*np.cos(theta);y=.5*np.sin(theta)
    def cartesian(x,y,z):
        radius=np.hypot(x,y);angle=np.arctan2(y,x);u=model.evaluate(q,[radius])[:,0]
        polar=2*np.real(u*np.exp(1j*(angle+model.alpha*z)))
        return np.array([polar[0]*np.cos(angle)-polar[1]*np.sin(angle),polar[0]*np.sin(angle)+polar[1]*np.cos(angle),polar[2]])
    point=np.array([x,y,z]);h=1e-5;gradient=np.empty((3,3))
    for axis in range(3):
        delta=np.eye(3)[axis]*h;gradient[:,axis]=(cartesian(*(point+delta))-cartesian(*(point-delta)))/(2*h)
    expected=gradient@cartesian(*point)
    polar=sum(value[:,0]*np.exp(1j*(key[1]*theta+.628319*key[0]*z)) for key,value in n.items()).real
    actual=np.array([polar[0]*np.cos(theta)-polar[1]*np.sin(theta),polar[0]*np.sin(theta)+polar[1]*np.cos(theta),polar[2]])
    np.testing.assert_allclose(actual,expected,atol=1e-9)


@pytest.mark.parametrize('extent,quadrature',[(1,32),(1,64),(2,64),(2,96)])
def test_convolution_fft_energy_and_hermitian_output(extent,quadrature):
    modes=modes_for_test(extent,quadrature=quadrature);model=modes[0][0];r=model.radius
    v,d=signed_spectrum(modes,r,.628319);a=convolution_advection(r,.628319,v,d);b,proof=fft_advection(r,.628319,v,d)
    assert proof['alias_free_for_requested_targets']
    for key,value in a.items():
        np.testing.assert_allclose(value,b[key],atol=1e-12)
        np.testing.assert_allclose(value,a[(-key[0],-key[1])].conj(),atol=1e-12)
    assert nonlinear_power(v,a,model.weights)['absolute_advection_power']<1e-12


def test_retained_padding_and_explicit_alias_failure():
    modes=modes_for_test(2);model=modes[0][0];r=model.radius;v,d=signed_spectrum(modes,r,.628319)
    targets=[(k,m) for k in range(-2,3) for m in range(-2,3)]
    exact=convolution_advection(r,.628319,v,d,targets)
    with pytest.raises(ValueError,match='padding'):fft_advection(r,.628319,v,d,targets,grid=(5,5))
    bad,proof=fft_advection(r,.628319,v,d,targets,grid=(5,5),allow_aliasing=True)
    assert not proof['alias_free_for_requested_targets']
    assert max(np.max(abs(exact[key]-bad[key])) for key in exact)>1e-5
    assert nonlinear_power(v,bad,model.weights)['absolute_advection_power']>1e-9
    good,proof=fft_advection(r,.628319,v,d,targets)
    assert proof['required_grid']==[7,7]
    for key in exact:np.testing.assert_allclose(exact[key],good[key],atol=1e-12)


@pytest.mark.parametrize('kind',['axis','derivative_keys','missing_conjugate','nonhermitian','nan','shape','mode_key','duplicate_targets'])
def test_invalid_modal_inputs_are_rejected(kind):
    modes=modes_for_test();r=modes[0][0].radius.copy();v,d=signed_spectrum(modes,r,.628319);targets=None
    if kind=='axis':r[0]=0
    if kind=='derivative_keys':d.pop((1,1))
    if kind=='missing_conjugate':v.pop((-1,-1));d.pop((-1,-1))
    if kind=='nonhermitian':v[(1,1)]=v[(1,1)]+1
    if kind=='nan':v[(1,1)][0,0]=np.nan
    if kind=='shape':v[(1,1)]=v[(1,1)][:,:-1]
    if kind=='mode_key':v[(99,0)]=v.pop((0,0));d[(99,0)]=d.pop((0,0))
    if kind=='duplicate_targets':targets=[(0,0),(0,0)]
    with pytest.raises(ValueError):convolution_advection(r,.628319,v,d,targets)


@pytest.mark.parametrize('grid',[(2,3),(0,3),(True,3),(66,3)])
def test_fft_grid_must_represent_input(grid):
    modes=modes_for_test();r=modes[0][0].radius;v,d=signed_spectrum(modes,r,.628319)
    with pytest.raises(ValueError):fft_advection(r,.628319,v,d,grid=grid,allow_aliasing=True)


def test_energy_requires_complete_targets_and_positive_quadrature():
    modes=modes_for_test();model=modes[0][0];v,d=signed_spectrum(modes,model.radius,.628319);n=convolution_advection(model.radius,.628319,v,d)
    n.pop((0,0))
    with pytest.raises(ValueError):nonlinear_power(v,n,model.weights)
    n[(0,0)]=np.zeros_like(v[(0,0)]);w=model.weights.copy();w[0]=-w[0]
    with pytest.raises(ValueError):nonlinear_power(v,n,w)


def test_canonical_duplicates_wrong_alpha_and_complex_mean_are_rejected():
    modes=modes_for_test();r=modes[0][0].radius
    with pytest.raises(ValueError):signed_spectrum([modes[0],modes[0]],r,.628319)
    with pytest.raises(ValueError):signed_spectrum(modes,r,.7)
    model=modes[0][0];u=np.zeros((3,len(r)),complex);u[1]=1j*r*(1-r*r)
    with pytest.raises(ValueError):signed_spectrum([(model,model.fit(u))],r,.628319)


def test_malformed_target_cannot_bypass_bounded_mode_validation():
    modes=modes_for_test();r=modes[0][0].radius;v,d=signed_spectrum(modes,r,.628319)
    with pytest.raises(ValueError):convolution_advection(r,.628319,v,d,targets=[[0,0]])


def test_energy_diagnostic_rejects_half_spectrum():
    modes=modes_for_test();model=modes[0][0];v,d=signed_spectrum(modes,model.radius,.628319);n=convolution_advection(model.radius,.628319,v,d)
    v.pop((-1,-1))
    with pytest.raises(ValueError):nonlinear_power(v,n,model.weights)


def test_energy_diagnostic_rejects_nonhermitian_advection():
    modes=modes_for_test();model=modes[0][0];v,d=signed_spectrum(modes,model.radius,.628319);n=convolution_advection(model.radius,.628319,v,d)
    n[(1,1)]=n[(1,1)]+1j
    with pytest.raises(ValueError):nonlinear_power(v,n,model.weights)
