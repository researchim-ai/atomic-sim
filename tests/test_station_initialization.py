import numpy as np
import pytest
from scipy.linalg import lstsq, null_space
from atomic_model_station_spectral import layout,divergence
from atomic_model_station_initialization import project_velocity
from atomic_model_station_initialization.projection import constraint_matrix
from atomic_model_station_quadrature.data import load_fixture


def problem(k=2,m=2):
    rule=load_fixture()['rules']['32'];r=np.array(rule['radius']);w=np.array(rule['corrected_weights']);ks,ms,f=layout(k,m)
    rng=np.random.default_rng(849);u=[.001*(rng.normal(size=(len(ks),len(r)))+1j*rng.normal(size=(len(ks),len(r))))*(1-r*r) for _ in range(3)]
    for a in u:a[0]=a[0].real
    weight=np.sqrt(np.tile(w[:-1],3))
    for i,(axial,azimuthal) in enumerate(zip(ks,ms)):
        a,b=constraint_matrix(r,w,int(axial),int(azimuthal),.628319,.5)
        x=np.concatenate([v[i,:-1] for v in u])
        correction=lstsq(a/weight,a@x-b,lapack_driver='gelsy')[0]
        for field,values in zip(u,np.split(x-correction/weight,3)):field[i,:-1]=values
    return r,w,ks,ms,f,u


def compatible():
    r,w,k,m,f,u=problem()
    fields,_=project_velocity(*u,r,w,2,2,alpha=.628319,maximum_relative_correction=.1)
    return r,w,k,m,f,fields


def test_manufactured_state_satisfies_constraints_and_is_idempotent():
    r,w,k,m,f,u=compatible()
    v,report=project_velocity(*u,r,w,2,2,alpha=.628319,maximum_relative_correction=.01)
    assert report['divergence_l2_after']<1e-11
    assert report['relative_correction']<1e-10
    assert report['bulk_velocity_after']==pytest.approx(.5,abs=1e-13)
    assert report['wall_maximum_after']==0
    for a,b in zip(v,u):np.testing.assert_allclose(a,b,atol=1e-13)


def test_small_defect_is_removed_without_mutating_inputs():
    r,w,k,m,f,u=compatible();u=[v.copy() for v in u];u[2][1]+=1e-7*(1-r*r)
    before=[a.copy() for a in u]
    v,report=project_velocity(*u,r,w,2,2,alpha=.628319,maximum_relative_correction=.01)
    assert report['divergence_l2_before']>1e-9 and report['divergence_l2_after']<1e-11
    assert 0<report['relative_correction']<.01
    for a,b in zip(u,before):np.testing.assert_array_equal(a,b)
    assert not report['continuum_axis_regularity_qualified'] and not report['pressure_compatibility_qualified']


def test_svd_projection_agrees_with_independent_pivoted_qr_and_nullspace():
    r,w,k,m,f,u=compatible();u=[v.copy() for v in u];u[0][2]+=1e-7*(1-r*r)
    v,report=project_velocity(*u,r,w,2,2,alpha=.628319,maximum_relative_correction=.01)
    i=2;a,b=constraint_matrix(r,w,int(k[i]),int(m[i]),.628319,.5)
    weight=np.sqrt(np.tile(w[:-1],3));x=np.concatenate([q[i,:-1] for q in u]);new=np.concatenate([q[i,:-1] for q in v])
    correction=lstsq(a/weight,a@x-b,lapack_driver='gelsy')[0]
    np.testing.assert_allclose(new,x-correction/weight,atol=1e-12)
    z=null_space(a/weight);assert np.linalg.norm(z.conj().T@((x-new)*weight))<1e-11
    assert report['maximum_stationarity_residual']<1e-12


@pytest.mark.parametrize('change',[dict(alpha=0),dict(alpha=float('nan')),dict(alpha=True),dict(maximum_relative_correction=0),dict(maximum_relative_correction=.11),dict(target_bulk=-1)])
def test_bad_policy_is_rejected(change):
    r,w,k,m,f,u=compatible();args=dict(alpha=.628319,maximum_relative_correction=.01);args.update(change)
    with pytest.raises(ValueError):project_velocity(*u,r,w,2,2,**args)


@pytest.mark.parametrize('defect',['nan','real_array','shape','imaginary_mean','weights','radius'])
def test_bad_state_is_rejected(defect):
    r,w,k,m,f,u=compatible();u=[a.copy() for a in u]
    if defect=='nan':u[0][1,0]=np.nan
    if defect=='real_array':u[0]=u[0].real
    if defect=='shape':u[0]=u[0][:-1]
    if defect=='imaginary_mean':u[0][0,0]+=1j
    if defect=='weights':w[0]*=-1
    if defect=='radius':r[1]=r[0]
    with pytest.raises(ValueError):project_velocity(*u,r,w,2,2,alpha=.628319,maximum_relative_correction=.01)


def test_large_change_is_refused_instead_of_silently_projected():
    r,w,k,m,f,u=compatible();u=[a.copy() for a in u];u[2][1]+=1.*(1-r*r)
    with pytest.raises(ValueError,match='budget'):project_velocity(*u,r,w,2,2,alpha=.628319,maximum_relative_correction=1e-10)


def test_checkpoint_preparation_preserves_input_temperature_and_metadata(tmp_path):
    from scipy.io import netcdf_file
    from scripts.prepare_station_spatial_state import run
    from atomic_model_station_spectral import read_state
    from atomic_model_multiphysics.provenance import sha
    r,w,k,m,f,u=compatible();source=tmp_path/'input.cdf'
    with netcdf_file(source,'w') as out:
        for key,value in dict(t=0.,Re=5300.,Pr=.7,C=0.,alpha=.628319).items():setattr(out,key,np.float64(value))
        for name,size in [('ReIm',2),('H',len(k)),('r',len(r))]:out.createDimension(name,size)
        out.createVariable('r','d',('r',))[:]=r
        for name,values in zip(('Ur','Ut','Uz','T'),[*u,np.zeros_like(u[0])]):
            a=out.createVariable(name,'d',('ReIm','H','r'));a.K=np.int32(2);a.M=np.int32(2);a.Mp=np.int32(1)
            a[0]=values.real;a[1]=values.imag
    weights=tmp_path/'weights.dat';np.savetxt(weights,np.column_stack([r,w]),fmt='%.17e')
    initial_sha=sha(source);receipt=run(tmp_path/'candidate',source,weights,.01)
    assert sha(source)==initial_sha and receipt['thermal_field_unchanged'] and receipt['metadata_unchanged']
    assert receipt['after']['divergence_l2']<1e-11
    assert not receipt['station_usable'] and receipt['closed_tasks']==[]
    np.testing.assert_array_equal(read_state(source)['fields']['T'],read_state(tmp_path/'candidate/state.cdf.in')['fields']['T'])
    with pytest.raises(ValueError,match='Fresh'):run(tmp_path/'candidate',source,weights,.01)
    bad=w.copy();bad[0]=-bad[0];np.savetxt(weights,np.column_stack([r,bad]))
    with pytest.raises(ValueError):run(tmp_path/'rejected',source,weights,.01)
    assert not (tmp_path/'rejected').exists()


def test_zero_perturbation_uses_physical_base_for_relative_budget():
    r,w,k,m,f,u=problem();zero=[np.zeros_like(a) for a in u]
    v,report=project_velocity(*zero,r,w,2,2,alpha=.628319,maximum_relative_correction=.01)
    assert report['divergence_l2_after']<1e-12 and report['relative_correction']<1e-12
    assert report['velocity_rms_before']>0.5
