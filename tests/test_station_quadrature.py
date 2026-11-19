from pathlib import Path
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss,legvander
from scipy.linalg import null_space
from scipy.io import netcdf_file
from atomic_model_station_quadrature import correct_quadrature
from atomic_model_station_heated.native_reference import load_quadrature
from scripts import replay_station_corrected_quadrature as replay


@pytest.mark.parametrize('n,degree',[(16,0),(16,4),(16,10),(32,10),(64,16)])
def test_exact_gauss_rule_is_not_changed(n,degree):
    x,w=leggauss(n);r=np.sqrt((x+1)/2);weights=w/4
    result=correct_quadrature(r,weights,degree=degree)
    np.testing.assert_allclose(result['corrected_weights'],weights,rtol=0,atol=2e-15)
    assert result['correction_l2']<1e-14


@pytest.mark.parametrize('n',[32,64])
def test_native_correction_integrates_even_polynomials_without_moving_nodes(n):
    p=load_quadrature()['records'][str(n)];r=np.array(p['radius']);w=np.array(p['weights_rdr']);before_r=r.copy();before_w=w.copy()
    result=correct_quadrature(r,w);new=np.array(result['corrected_weights'])
    np.testing.assert_array_equal(r,before_r);np.testing.assert_array_equal(w,before_w)
    np.testing.assert_array_equal(result['radius'],r)
    assert new.min()>0 and not result['flow_data_used'] and not result['model_coefficients_fitted']
    for power in range(11):assert new@(r**(2*power))==pytest.approx(1/(2*power+2),abs=2e-14)


@pytest.mark.parametrize('n',[32,64])
def test_svd_solution_has_minimal_norm_among_other_feasible_corrections(n):
    p=load_quadrature()['records'][str(n)];r=np.array(p['radius']);w=np.array(p['weights_rdr'])
    result=correct_quadrature(r,w);delta=np.array(result['corrected_weights'])-w;a=legvander(2*r*r-1,10).T
    z=null_space(a)[:,0]*1e-5
    assert np.max(np.abs(a@z))<1e-14
    assert np.linalg.norm(delta+z)>np.linalg.norm(delta)
    assert delta@z==pytest.approx(0,abs=1e-18)


@pytest.mark.parametrize('bad',['nan','unsorted','negative_radius','beyond_wall','negative_weight','zero_weight','wrong_length','strings','complex','boolean','short','long','bad_mass'])
def test_invalid_rules_are_rejected_before_correction(bad):
    x,w=leggauss(32);r=np.sqrt((x+1)/2);w=w/4
    if bad=='nan':r[0]=np.nan
    elif bad=='unsorted':r=r[::-1]
    elif bad=='negative_radius':r[0]=-.1
    elif bad=='beyond_wall':r[-1]=1.01
    elif bad=='negative_weight':w[0]=-w[0]
    elif bad=='zero_weight':w[0]=0
    elif bad=='wrong_length':w=w[:-1]
    elif bad=='strings':r=r.astype(str)
    elif bad=='complex':r=r.astype(complex)
    elif bad=='boolean':r=r.astype(bool)
    elif bad=='short':r=r[:15];w=w[:15]
    elif bad=='long':r=np.linspace(0,1,1025);w=np.full(1025,.5/1025)
    elif bad=='bad_mass':w=w*10
    with pytest.raises(ValueError):correct_quadrature(r,w)


@pytest.mark.parametrize('degree',[-1,17,True,10.,'10'])
def test_degree_is_explicit_and_bounded(degree):
    x,w=leggauss(32)
    with pytest.raises(ValueError):correct_quadrature(np.sqrt((x+1)/2),w/4,degree=degree)


def test_nearly_collapsed_mesh_cannot_hide_a_singular_moment_system():
    r=np.linspace(1-1e-10,1,32);w=np.full(32,.5/32)
    with pytest.raises(ValueError,match='conditioning'):correct_quadrature(r,w)


def test_mesh_patch_is_narrow_and_checks_coordinates_before_using_weights():
    before=b'prefix\n      call mes_rdom_init(mes_D)\nsuffix\n';after=replay.patch_mesh(before)
    assert after==before.replace(replay.ANCHOR.encode(),(replay.ANCHOR+replay.PATCH).encode())
    assert b'mesh mismatch' in after and b'nonpositive weight' in after and b'iostat' in after
    with pytest.raises(ValueError,match='anchor'):replay.patch_mesh(b'other source')
    with pytest.raises(ValueError,match='anchor'):replay.patch_mesh(before+before)


def output_states(directory):
    for index,time in enumerate((0.,2000.,4000.,6000.,8000.)):
        p=directory/f'state{index:04}.cdf.dat';replay.original_harness.initial(p,32,0,.1)
        with netcdf_file(p,'a') as f:f.t=np.float64(time)


def test_complete_fresh_output_has_real_final_and_late_states(tmp_path):
    output_states(tmp_path);paths,final,late,error=replay.completed_states(tmp_path,32,0,.1)
    assert len(paths)==5 and final['time']==8000 and late['time']==6000 and error==0


@pytest.mark.parametrize('corruption',['early_time','nan_time','nan_prandtl','wrong_step','wrong_modes','swirl','nonfinite_field','not_settled','missing_state'])
def test_textual_success_cannot_override_invalid_external_states(tmp_path,corruption):
    output_states(tmp_path);p=tmp_path/'state0004.cdf.dat'
    if corruption=='missing_state':(tmp_path/'state0001.cdf.dat').unlink()
    else:
        with netcdf_file(p,'a') as f:
            if corruption=='early_time':f.t=np.float64(.4)
            elif corruption=='nan_time':f.t=np.float64(np.nan)
            elif corruption=='nan_prandtl':f.Pr=np.float64(np.nan)
            elif corruption=='wrong_step':f.variables['dt'][...]=.01
            elif corruption=='wrong_modes':f.variables['T'].M=np.int32(2)
            elif corruption=='swirl':f.variables['Ut'][0,0,0]=.1
            elif corruption=='nonfinite_field':f.variables['T'][0,0,0]=np.nan
            elif corruption=='not_settled':f.variables['T'][0,0,0]=.1
    with pytest.raises(ValueError):replay.completed_states(tmp_path,32,0,.1)


def test_historical_sources_allow_only_evidenced_exact_cache_migration():
    from scripts.qualify_station_inventory import read
    from scripts.qualify_station_shutdown import identity as station
    from scripts.qualify_station_mixed_convection import identity as mixed
    from scripts.qualify_station_pipe import identity as pipe
    from scripts.qualify_station_turbulence import identity as turbulent
    from scripts.qualify_station_heated import identity as heated
    folder=Path(__file__).resolve().parents[1]/'docs/benchmarks'
    for name,identity in [('station_low_flow_v63',station),('station_mixed_convection_v64',mixed),('station_pipe_v65',pipe),('station_turbulence_v66',turbulent),('station_heated_v67',heated)]:
        from tests._station_history import assert_source_compatible
        assert_source_compatible(read(folder/(name+'.json'))['source'],identity())


def test_nonpositive_minimum_norm_solution_is_rejected_without_clipping():
    r=np.linspace(0,1,32);w=np.full(32,1e-10);w[0]=.5-np.sum(w[1:])
    with pytest.raises(ValueError,match='not positive'):correct_quadrature(r,w)
