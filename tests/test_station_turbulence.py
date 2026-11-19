"""Published constants, independent balances and exact DNS conventions."""
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad
from atomic_model_station_turbulence import eddy_viscosity,bulk_state,at_bulk_reynolds,velocity,finite_volume,load_profile


@pytest.mark.parametrize('tau',[100.,181.0540998312275,550.,999.0327072339593,2000.])
def test_original_b2_and_wall_units_are_equivalent(tau):
    # Original Re is based on 2Ub and R; B=-dP/dz; tau=Re sqrt(B/2).
    reb=bulk_state(tau)['re_bulk'];b=2*(tau/reb)**2
    y=np.linspace(0,1,121);r=1-y;rh=reb/2;bh=2*b
    x=.42**2*rh**2*bh/9*(2*y-y*y)**2*(3-4*y+2*y*y)**2*(1-np.exp(-y*rh*np.sqrt(bh)/27))**2
    original=.5*np.sqrt(1+x)-.5
    np.testing.assert_allclose(eddy_viscosity(r,tau),original,rtol=1e-10,atol=1e-13)
    assert eddy_viscosity(1,tau)==0
    assert float(velocity(1,tau))==0
    h=1e-7
    assert float(velocity(1-h,tau))/h==pytest.approx(tau,rel=1e-6)


@pytest.mark.parametrize('tau',[100.,180.,550.,1000.,2000.])
def test_flux_inverse_and_darcy_convention(tau):
    state=bulk_state(tau);inverse=at_bulk_reynolds(state['re_bulk'])
    assert inverse['re_tau']==pytest.approx(tau,rel=1e-10)
    assert state['darcy_friction']==pytest.approx(32*tau**2/state['re_bulk']**2,rel=1e-14)
    # Independent integration of velocity, rather than the swapped bulk integral.
    ub=quad(lambda r:2*r*float(velocity(r,tau)),0,1,epsabs=1e-8)[0]
    assert ub==pytest.approx(state['bulk_velocity_plus'],rel=1e-9)


@pytest.mark.parametrize('label',[180,360,550,1000])
def test_dns_bytes_radius_and_friction_units(label):
    p=load_profile(label);r=np.array(p['radius']);u=np.array(p['velocity_plus'])
    assert r[0]==0 and r[-1]==1 and np.all(np.diff(r)>0)
    # Preserve the original DNS roundoff at the wall rather than projecting it.
    assert u[0]>u[-1] and abs(u[-1])<1e-10
    assert p['darcy_friction']==pytest.approx(32*p['re_tau']**2/p['re_bulk']**2,rel=1e-13)
    assert p['re_tau']!=float(label)  # actual header, not rounded filename
    assert not np.shares_memory(np.array(load_profile(label)['radius']),r)
    root=files('atomic_model_station_turbulence').joinpath('data');m=json.loads(root.joinpath('sources.json').read_text())
    assert m['license']=='CC-BY-4.0' and m['changes'].startswith('None:')
    entry=m['profiles'][str(label)]
    assert hashlib.md5(root.joinpath(entry['file']).read_bytes()).hexdigest()==entry['md5']


@pytest.mark.parametrize('tau',[180.,1000.,2000.])
def test_fv_pressure_work_and_wall_shear_are_conservative(tau):
    p=finite_volume(tau,512)
    assert p['work_relative_residual']<1e-9 and p['wall_stress_relative_residual']<1e-8
    assert p['pressure_work']>0 and p['dissipation']>0
    assert np.all(np.diff(p['velocity_plus'])<0)
    # No normalized post-solve flux: FV error is visible in the independent bulk.
    assert p['bulk_velocity_plus']!=bulk_state(tau)['bulk_velocity_plus']


@pytest.mark.parametrize('value',[True,'180',99.,2001.,float('nan'),float('inf'),[180]])
def test_tau_outside_contract_is_refused(value):
    with pytest.raises(ValueError):eddy_viscosity(.5,value)


@pytest.mark.parametrize('radius',[-.01,1.001,float('nan'),float('inf'),[]])
def test_invalid_coordinates_are_not_clipped(radius):
    with pytest.raises(ValueError):velocity(radius,180.)


@pytest.mark.parametrize('value',[True,'5300',-1,float('inf'),float('nan'),1e12,[5300]])
def test_invalid_bulk_is_refused(value):
    with pytest.raises(ValueError):at_bulk_reynolds(value)


@pytest.mark.parametrize('cells',[True,15,4097,32.,'32'])
def test_invalid_mesh_is_refused(cells):
    with pytest.raises(ValueError):finite_volume(180,cells)


@pytest.mark.parametrize('label',[True,180.,'180',179])
def test_unknown_dns_case_is_refused(label):
    with pytest.raises(ValueError):load_profile(label)


def test_modified_data_is_refused_before_parsing(tmp_path,monkeypatch):
    import atomic_model_station_turbulence.dns as module
    root=tmp_path/'data';root.mkdir();source=files('atomic_model_station_turbulence').joinpath('data')
    (root/'sources.json').write_bytes(source.joinpath('sources.json').read_bytes())
    (root/'180_Re_1.dat').write_bytes(source.joinpath('180_Re_1.dat').read_bytes()+b'\n')
    monkeypatch.setattr(module,'files',lambda _:tmp_path)
    with pytest.raises(ValueError,match='checksum'):module.load_profile(180)


def test_historical_sources_allow_only_evidenced_exact_cache_migration():
    from scripts.qualify_station_inventory import read
    from scripts.qualify_station_shutdown import identity as low
    from scripts.qualify_station_mixed_convection import identity as mixed
    from scripts.qualify_station_pipe import identity as pipe
    root=Path(__file__).resolve().parents[1]/'docs/benchmarks'
    for name,identity in [('station_low_flow_v63',low),('station_mixed_convection_v64',mixed),('station_pipe_v65',pipe)]:
        from tests._station_history import assert_source_compatible
        assert_source_compatible(read(root/(name+'.json'))['source'],identity())
