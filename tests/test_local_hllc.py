"""Local acoustic HLLC admission, lifecycle and nonlinear physical gates."""
from copy import deepcopy
import numpy as np
import pytest
from atomic_model_hem_contact import ContactHEMNetwork
from atomic_model_hem_contact.flux import local_hllc,local_envelope,hllc
from atomic_model_hem_qualified import eos
from scripts.benchmark_modelica_network import request_for
from scripts.verify_local_hllc import face_cases


def test_nonlinear_oracle_wave_coverage_and_pressure_flux_accuracy():
    rows=face_cases()
    assert len(rows)==15 and all(r['passed'] for r in rows)
    wide=next(r for r in rows if r['case']=='wide')
    assert wide['relative_mass_flux_error']<.02


@pytest.mark.parametrize('velocity',[-20.,0.,20.])
def test_exact_contacts_and_energy_reference(velocity):
    left,right=eos.pt(4e6,441.),eos.pt(4e6,439.)
    a=local_hllc(left,right,velocity,velocity,wave_bound=2500.,energy_reference=0.)
    b=local_hllc(left,right,velocity,velocity,wave_bound=2500.,energy_reference=7e5)
    b[2]+=7e5*b[0]
    expected=hllc(left,right,velocity,velocity,wave_bound=2500.,energy_reference=0.)
    np.testing.assert_allclose(a,expected,rtol=1e-14,atol=1e-4)
    np.testing.assert_allclose(b,expected,rtol=1e-14,atol=1e-4)


def test_reflection_and_centered_energy_for_pressure_wave():
    left,right=eos.pt(14e6,441.),eos.pt(2e6,439.)
    a=local_hllc(left,right,.02,-.01,wave_bound=2500.,energy_reference=0.)
    b=local_hllc(right,left,.01,-.02,wave_bound=2500.,energy_reference=0.)
    np.testing.assert_allclose(b,a*[-1,1,-1],rtol=1e-9,atol=1e-4)
    c=local_hllc(left,right,.02,-.01,wave_bound=2500.,energy_reference=7e5);c[2]+=7e5*c[0]
    np.testing.assert_allclose(c,a,rtol=1e-9,atol=1e-4)


@pytest.mark.parametrize('damage',['cap','nan','phase','star'])
def test_local_envelope_refuses_unsupported_fans(damage):
    left=right=eos.pt(4e6,440.);vl=vr=0.;maximum=2500.
    if damage=='cap':maximum=1000.
    elif damage=='nan':vl=float('nan')
    elif damage=='phase':right=eos.pt(1e6,600.)
    else:vl=8.;vr=-8.
    with pytest.raises(ValueError):local_hllc(left,right,vl,vr,wave_bound=maximum,energy_reference=7e5)


def test_star_check_adds_a_real_restriction_without_a_fallback():
    state=eos.pt(4e6,440.)
    bound=local_envelope(state,state,8.,-8.,maximum=2500.)
    # Input envelope alone admits this fan; the admitted star does not fit it.
    assert np.isfinite(hllc(state,state,8.,-8.,wave_bound=bound,energy_reference=7e5)).all()
    with pytest.raises(ValueError,match='star exceeds'):
        local_hllc(state,state,8.,-8.,wave_bound=2500.,energy_reference=7e5)


def test_restart_scheme_binding_and_global_cfl_retained():
    request=request_for('coupled_gradient',2,1e-5)
    request['model']['max_step_s']=5e-6
    model=ContactHEMNetwork(request,scheme='hllc-local')
    original=ContactHEMNetwork(request)
    assert model.wave==original.wave and model.cfl==original.cfl
    model.advance(.00025);checkpoint=model.checkpoint()
    replay=ContactHEMNetwork(request,scheme='hllc-local');replay.restore(checkpoint)
    assert replay.advance(.00025)==model.advance(.00025)
    assert replay.checkpoint()==model.checkpoint()
    before=original.checkpoint()
    with pytest.raises(ValueError):original.restore(checkpoint)
    assert original.checkpoint()==before
    assert model.contract()['numerical_flux']=='hllc-local'


def test_refused_command_rolls_back_local_model():
    model=ContactHEMNetwork(request_for('coupled_gradient'),scheme='hllc-local')
    before=model.checkpoint();controls=deepcopy(model.snapshot()['controls'])
    controls['fluid']['a']=1.
    with pytest.raises(ValueError):model.step(controls,.00025,expected_revision=0,controller_state={})
    assert model.checkpoint()==before
