"""Analytic contact fluxes and real liquid graph lifecycle, no peer mocks."""
from copy import deepcopy
import numpy as np
import pytest
from atomic_model_hem_contact import ContactHEMNetwork
from atomic_model_hem_contact.flux import hllc
from atomic_model_hem_qualified import eos
from atomic_model_hem_qualified.model import QualifiedHEMNetwork
from atomic_model_multiphysics.provenance import seal
from scripts.benchmark_modelica_network import request_for


@pytest.mark.parametrize('p,tl,tr',[(1e6,430.,440.),(4e6,430.,450.),(15.5e6,550.,580.)])
@pytest.mark.parametrize('v',[-20.,-1.,0.,1.,20.])
@pytest.mark.parametrize('reference',[0.,7e5])
def test_exact_stationary_and_advected_contacts(p,tl,tr,v,reference):
    left,right=eos.pt(p,tl),eos.pt(p,tr)
    actual=hllc(left,right,v,v,wave_bound=2500.,energy_reference=reference)
    state=left if v>=0 else right;rho=state.density_kg_m3
    expected=np.array([rho*v,rho*v*v+p,v*(rho*(state.internal_energy_j_kg+.5*v*v)+p)])
    physical=actual.copy();physical[2]+=reference*actual[0]
    np.testing.assert_allclose(physical,expected,rtol=2e-14,atol=1e-4)


def test_face_orientation_reversal_and_energy_reference_invariance():
    left,right=eos.pt(4.02e6,441.),eos.pt(3.98e6,439.)
    a=hllc(left,right,.02,-.01,wave_bound=2500.,energy_reference=0.)
    b=hllc(right,left,.01,-.02,wave_bound=2500.,energy_reference=0.)
    np.testing.assert_allclose(b,a*np.array([-1,1,-1]),rtol=1e-10,atol=1e-5)
    shifted=hllc(left,right,.02,-.01,wave_bound=2500.,energy_reference=7e5)
    shifted[2]+=7e5*shifted[0]
    np.testing.assert_allclose(shifted,a,rtol=1e-9,atol=1e-4)


def test_small_pressure_difference_is_not_projected_to_stationary_contact():
    flux=hllc(eos.pt(4e6,441.),eos.pt(4e6+1.,439.),0.,0.,wave_bound=2500.,energy_reference=7e5)
    assert abs(flux[0])>1e-9


@pytest.mark.parametrize('damage',['nan','wave','phase','star'])
def test_inadmissible_flux_has_no_silent_fallback(damage):
    left,right=eos.pt(4e6,441.),eos.pt(4e6,439.)
    vl=vr=0.;wave=2500.
    if damage=='nan':vl=float('nan')
    elif damage=='wave':wave=1.
    elif damage=='phase':right=eos.pt(1e6,600.)
    else:vl=-200.;vr=200.
    with pytest.raises(ValueError):hllc(left,right,vl,vr,wave_bound=wave,energy_reference=7e5)


@pytest.mark.parametrize('scheme',['hllc','rusanov'])
def test_real_graph_checkpoint_commands_and_restart(scheme):
    request=request_for('thermal_step',2,1e-5)
    model=ContactHEMNetwork(request,scheme=scheme);model.advance(.00025)
    controls=model.snapshot()['controls']
    final=model.step(controls,.00025,expected_revision=1,controller_state={'tick':1})
    restored=ContactHEMNetwork(request,scheme=scheme);restored.restore(model.checkpoint())
    assert restored.checkpoint()==model.checkpoint()
    assert restored.snapshot()['cells']==final['cells']
    assert model.contract()['numerical_flux']==scheme
    with pytest.raises(AttributeError):model.scheme='other'


def test_hllc_stationary_temperature_contact_is_preserved():
    model=ContactHEMNetwork(request_for('thermal_step',4,1e-5))
    initial=model.snapshot();final=model.advance(.001)
    for a,b in zip(initial['cells'],final['cells'],strict=True):
        assert a['mass_kg']==b['mass_kg']
        assert abs(a['temperature_k']-b['temperature_k'])<=1e-9
    assert abs(final['global_ledger']['energy_residual_j'])<=1e-5


@pytest.mark.parametrize('damage',['forcing','phase','friction','scheme'])
def test_unqualified_scope_is_refused(damage):
    request=request_for('thermal_step');scheme='hllc'
    if damage=='forcing':request['model']['nodes'][0]['heat_power_w']=1.
    elif damage=='phase':request['model']['nodes'][0]['initial']=dict(kind='px',pressure_pa=1e6,quality=.5,velocity_m_s=0.)
    elif damage=='friction':request['model']['edges'][0]['friction']='darcy'
    else:scheme='unknown'
    with pytest.raises(ValueError):ContactHEMNetwork(request,scheme=scheme)


@pytest.mark.parametrize('damage',['source','coordinates','scheme','legacy'])
def test_refused_checkpoint_preserves_current_state(damage):
    request=request_for('coupled_gradient');model=ContactHEMNetwork(request)
    model.advance(.00025);before=model.checkpoint();cp=deepcopy(before)
    if damage=='source':cp['payload']['source']={}
    elif damage=='coordinates':cp['payload']['coordinates'][0]+=1e-6
    elif damage=='scheme':cp=ContactHEMNetwork(request,scheme='rusanov').checkpoint()
    else:cp=QualifiedHEMNetwork(request).checkpoint()
    if damage in ('source','coordinates'):cp=seal(cp['payload'])
    with pytest.raises(ValueError):model.restore(cp)
    assert model.checkpoint()==before


def test_refused_forced_stage_rolls_back_and_legacy_dynamics_are_unchanged():
    request=request_for('coupled_gradient');model=ContactHEMNetwork(request)
    before=model.checkpoint();controls=deepcopy(model.snapshot()['controls']);controls['fluid']['a']=1.
    with pytest.raises(ValueError):model.step(controls,.00025,expected_revision=0,controller_state={})
    assert model.checkpoint()==before
    legacy=QualifiedHEMNetwork(request);baseline=ContactHEMNetwork(request,scheme='rusanov')
    assert legacy.advance(.00025)==baseline.advance(.00025)
