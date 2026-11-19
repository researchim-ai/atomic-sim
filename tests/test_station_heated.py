from dataclasses import replace
import io
from pathlib import Path
import tarfile
import numpy as np
import pytest
from core.status import SolverError,ModelLimitError
from atomic_model_station_heated import HeatedCase,solve_heated,continuum,flux_to_delta


@pytest.mark.parametrize('boundary',['delta_temperature','heat_flux'])
@pytest.mark.parametrize('cells',[32,128,512])
def test_forced_polynomial_and_two_different_temperature_means(boundary,cells):
    p=solve_heated(HeatedCase(boundary=boundary,cells=cells));r=np.array(p['radius'])
    a=6 if boundary=='delta_temperature' else 8
    assert np.max(np.abs(np.array(p['velocity'])-(1-r*r)))<1/cells**2
    assert np.max(np.abs(np.array(p['temperature'])-(1-a*(3-4*r*r+r**4)/16)))<2/cells**2
    assert p['nusselt_area']==pytest.approx(6,rel=2/cells**2)
    assert p['nusselt_mixing']==pytest.approx(48/11,rel=2/cells**2)
    assert p['area_mean_temperature']!=p['mixing_cup_temperature']
    assert min(p['temperature'])<0  # gauge-relative, not kelvin: never clip to[0,1]


@pytest.mark.parametrize('boundary',['delta_temperature','heat_flux'])
@pytest.mark.parametrize('c',[0,2,5])
def test_fv_converges_against_singular_axis_collocation(boundary,c):
    case=HeatedCase(boundary=boundary,buoyancy_c=c);ref=continuum(case);errors=[]
    for n in (64,128,256):
        p=solve_heated(replace(case,cells=n))
        errors.append(float(np.max(np.abs(ref['solution'].sol(p['radius'])[[0,2]]-np.array([p['velocity'],p['temperature']])))))
        assert p['checks']['heat']<1e-9 and p['checks']['kinetic_relative']<1e-9
        assert p['checks']['mass']<1e-11 and p['checks']['boundary']<1e-9
        assert not p['station_usable'] and not p['turbulent'] and not p['validated']
    assert min(np.log2(np.array(errors[:-1])/errors[1:]))>1.8


@pytest.mark.parametrize('c',[0,2,5])
def test_same_physical_case_requires_changed_c_and_pressure_reference(c):
    p=solve_heated(HeatedCase(boundary='heat_flux',buoyancy_c=c,cells=128));mapped=flux_to_delta(p)
    other=solve_heated(HeatedCase(boundary='delta_temperature',buoyancy_c=mapped['buoyancy_c'],cells=128))
    np.testing.assert_allclose(p['velocity'],other['velocity'],rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(mapped['temperature'],other['temperature'],rtol=1e-9,atol=1e-10)
    assert mapped['pressure_beta']==pytest.approx(other['pressure_beta'],abs=1e-9)
    assert p['nusselt_mixing']==pytest.approx(other['nusselt_mixing'],rel=1e-9)
    if c:assert mapped['buoyancy_c']!=c and p['pressure_beta']!=other['pressure_beta']


@pytest.mark.parametrize('boundary',['delta_temperature','heat_flux'])
def test_steady_shape_invariant_but_axial_gradient_changes_with_re_pr(boundary):
    case=HeatedCase(boundary=boundary,buoyancy_c=2)
    first=solve_heated(case);second=solve_heated(replace(case,reynolds=1000,prandtl=1.4))
    assert first['velocity']==second['velocity'] and first['temperature']==second['temperature']
    assert first['axial_gradient']*5300*.7==pytest.approx(second['axial_gradient']*1000*1.4)


@pytest.mark.parametrize('key,value',[('boundary','automatic'),('buoyancy_c',-1),('buoyancy_c',11),('buoyancy_c',True),
 ('buoyancy_c',float('nan')),('reynolds',0),('prandtl',float('inf')),('cells',True),('cells',15),('cells',1025),('cells',32.)])
def test_invalid_case_never_enters_solve(key,value):
    with pytest.raises(ValueError):HeatedCase(**{key:value})


def test_newton_failure_does_not_produce_a_result(monkeypatch):
    import atomic_model_station_heated.model as m
    monkeypatch.setattr(m,'spsolve',lambda matrix,rhs:np.full(len(rhs),np.nan))
    with pytest.raises(SolverError,match='Nonfinite'):m.solve_heated()


def test_local_reversal_is_rejected_even_if_newton_claims_success(monkeypatch):
    import atomic_model_station_heated.model as m
    original=m._newton
    def wrong(*a):
        y,it,residual=original(*a);y[0]=-1.;return y,it,residual
    monkeypatch.setattr(m,'_newton',wrong)
    with pytest.raises(ModelLimitError,match='reverses'):m.solve_heated()


def test_failed_continuum_is_not_accepted(monkeypatch):
    import atomic_model_station_heated.reference as m
    class Failed:success=False;message='independent failure'
    monkeypatch.setattr(m,'solve_bvp',lambda *a,**k:Failed())
    with pytest.raises(SolverError,match='independent failure'):m.continuum()


def test_historical_sources_allow_only_evidenced_exact_cache_migration():
    from scripts.qualify_station_inventory import read
    from scripts.qualify_station_shutdown import identity as station
    from scripts.qualify_station_mixed_convection import identity as mixed
    from scripts.qualify_station_pipe import identity as pipe
    from scripts.qualify_station_turbulence import identity as turbulence
    root=Path(__file__).resolve().parents[1]/'docs/benchmarks'
    for name,identity in [('station_low_flow_v63',station),('station_mixed_convection_v64',mixed),('station_pipe_v65',pipe),('station_turbulence_v66',turbulence)]:
        from tests._station_history import assert_source_compatible
        assert_source_compatible(read(root/(name+'.json'))['source'],identity())


def test_external_unknown_archive_is_rejected(tmp_path):
    from scripts.replay_station_heated_reference import inventory
    p=tmp_path/'wrong.tgz';p.write_bytes(b'not an archive')
    with pytest.raises(ValueError,match='checksum'):inventory(p)


@pytest.mark.parametrize('name',['../escape','/absolute','openpipeflow-hp-1.21/../../escape'])
def test_external_archive_cannot_escape_workspace(tmp_path,monkeypatch,name):
    from scripts import replay_station_heated_reference as m
    from atomic_model_multiphysics.provenance import sha
    p=tmp_path/'test.tgz'
    with tarfile.open(p,'w:gz') as t:
        item=tarfile.TarInfo(name);item.size=1;t.addfile(item,io.BytesIO(b'x'))
    monkeypatch.setattr(m,'ARCHIVE_SHA',sha(p))
    with pytest.raises(ValueError,match='Unsafe'):m.inventory(p)
    assert not (tmp_path/'escape').exists()


def test_native_series_reproduces_the_forced_polynomials():
    from atomic_model_station_heated.native_reference import coefficients
    for a in (6.,8.):
        u,t=coefficients([1.,1-3*a/16,0.,a],0.)
        np.testing.assert_array_equal(u[:3],[1.,-1.,0.])
        np.testing.assert_array_equal(t[:3],[1-3*a/16,a/4,-a/16])
        assert not np.any(u[2:]) and not np.any(t[3:])


def test_native_weights_predict_the_external_forced_temperature_bias():
    from atomic_model_station_heated.native_reference import load_quadrature,native_prediction
    from atomic_model_station_heated.external import load_external
    native=load_quadrature();fixture=load_external();errors=[]
    for n in (32,64):
        p=native['records'][str(n)];moments=p['moments'];m0=moments['0']['native'];m2=moments['2']['native'];m4=moments['4']['native']
        a=16*(m0-m2)/(3*m0-4*m2+m4)
        predicted=native_prediction('delta_temperature',0,n)
        assert predicted['axial_gradient_scaled']==pytest.approx(a,abs=1e-10)
        actual=fixture['records'][f'delta_temperature-c0-n{n}-dt0.1']['final']
        np.testing.assert_allclose(predicted['temperature'],actual['temperature'],rtol=0,atol=1e-10)
        errors.append(abs(a-6.))
    assert errors[1]<errors[0] and errors[1]>1e-5
