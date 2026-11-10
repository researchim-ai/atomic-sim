"""Exact per-stage sound reuse without losing star admission or failure guards."""
from copy import deepcopy
import json
import numpy as np
import pytest
from atomic_model_hem_contact import ContactHEMNetwork
from atomic_model_hem_contact import flux
from atomic_model_hem_contact.flux import _FluxEvaluation,hllc,local_hllc,local_envelope
from atomic_model_hem_qualified import eos
from atomic_model_multiphysics.provenance import seal
from scripts.benchmark_modelica_network import request_for
from scripts import benchmark_hllc_speed as bench


def inputs():return eos.pt(4.02e6,441.),eos.pt(3.98e6,439.)


def test_public_face_reuses_sounds_but_still_inverts_both_stars(monkeypatch):
    left,right=inputs();sounds=[];inversions=[]
    original_sound=eos.characteristic_speeds;original_inverse=flux._inverse
    def sound(state):sounds.append(state);return original_sound(state)
    def inverse(*args,**kwargs):inversions.append(args);return original_inverse(*args,**kwargs)
    monkeypatch.setattr(eos,'characteristic_speeds',sound);monkeypatch.setattr(flux,'_inverse',inverse)
    result=local_hllc(left,right,.02,-.01,wave_bound=2500.,energy_reference=7e5)
    assert np.isfinite(result).all() and len(inversions)==2
    assert len(sounds)==4 and len(set(sounds))==4


def test_validated_primitive_sounds_are_reused_with_original_scalar_type(monkeypatch):
    left,right=inputs();values=[eos.characteristic_speeds(s)[1] for s in (left,right)]
    expected=local_hllc(left,right,.02,-.01,wave_bound=2500.,energy_reference=7e5)
    calls=[];original=eos.characteristic_speeds
    def sound(state):calls.append(state);return original(state)
    monkeypatch.setattr(eos,'characteristic_speeds',sound)
    context=_FluxEvaluation((left,right),np.asarray(values))
    assert type(context.sound(left)) is float
    actual=context.local_hllc(left,right,.02,-.01,wave_bound=2500.,energy_reference=7e5)
    np.testing.assert_array_equal(actual,expected)
    assert len(calls)==2 and left not in calls and right not in calls


def test_full_state_keys_keep_small_pressure_differences(monkeypatch):
    left=eos.pt(4e6,440.);right=eos.pt(4e6+1e-4,440.)
    calls=[];original=eos.characteristic_speeds
    def sound(state):calls.append(state);return original(state)
    monkeypatch.setattr(eos,'characteristic_speeds',sound)
    context=_FluxEvaluation()
    for state in (left,right,left,right):context.sound(state)
    assert calls==[left,right]


def test_public_calls_do_not_share_a_persistent_cache(monkeypatch):
    left,right=inputs();calls=[];original=eos.characteristic_speeds
    def sound(state):calls.append(state);return original(state)
    monkeypatch.setattr(eos,'characteristic_speeds',sound)
    for _ in range(2):local_hllc(left,right,.02,-.01,wave_bound=2500.,energy_reference=7e5)
    assert len(calls)==8


def test_cache_hits_cannot_bypass_star_or_global_envelopes():
    state=eos.pt(4e6,440.);context=_FluxEvaluation((state,),(eos.characteristic_speeds(state)[1],))
    for _ in range(2):
        with pytest.raises(ValueError,match='star exceeds'):
            context.local_hllc(state,state,8.,-8.,wave_bound=2500.,energy_reference=7e5)
    with pytest.raises(ValueError,match='global wave cap'):
        context.local_hllc(state,state,0.,0.,wave_bound=1000.,energy_reference=7e5)
    assert np.isfinite(context.local_hllc(state,state,1.,-1.,wave_bound=2500.,energy_reference=7e5)).all()


@pytest.mark.parametrize('function',[hllc,local_hllc,local_envelope])
def test_public_api_does_not_accept_unchecked_sound_values(function):
    left,right=inputs()
    kwargs=dict(maximum=2500.) if function is local_envelope else dict(wave_bound=2500.,energy_reference=7e5)
    with pytest.raises(TypeError,match='sound'):function(left,right,0.,0.,**kwargs,sound=lambda state:1.)


def test_mismatched_checkpoint_source_is_refused_without_state_change():
    model=ContactHEMNetwork(request_for('coupled_gradient'),scheme='hllc-local')
    model.advance(.00025);before=model.checkpoint();forged=deepcopy(before['payload'])
    forged['source']['contact']['flux.py']='0'*64
    with pytest.raises(ValueError):model.restore(seal(forged))
    assert model.checkpoint()==before


def test_baseline_provenance_is_checked_before_loading_code(tmp_path):
    baseline=tmp_path/'baseline';baseline.mkdir()
    (baseline/'__init__.py').write_text('raise RuntimeError("must never be imported")')
    protocol=tmp_path/'protocol.json'
    protocol.write_text(json.dumps(seal(dict(source=dict(contact={},foundation={})))) )
    with pytest.raises(ValueError,match='Baseline'):bench.run(tmp_path/'out',baseline=baseline,baseline_protocol=protocol)
    assert not (tmp_path/'out').exists()
