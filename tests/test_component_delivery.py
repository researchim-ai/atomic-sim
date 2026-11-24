from pathlib import Path

import pytest
from scripts import verify_component_delivery as delivery
from scripts import verify_pwr_local_cladding_delivery as legacy


def test_unknown_component_cannot_create_results(tmp_path):
    out=tmp_path/'results'
    with pytest.raises(ValueError,match='Registered'):delivery.run(out,component='unknown')
    assert not out.exists()


def test_existing_results_are_never_overwritten(tmp_path):
    (tmp_path/'evidence.txt').write_text('keep')
    with pytest.raises(ValueError,match='Fresh'):delivery.run(tmp_path,component='coolant-channel')
    assert (tmp_path/'evidence.txt').read_text()=='keep'


def test_output_inside_source_tree_is_refused_before_creation(tmp_path,monkeypatch):
    monkeypatch.setattr(delivery,'ROOT',tmp_path)
    out=tmp_path/'src'/'new-results'
    with pytest.raises(ValueError,match='outside source'):delivery.run(out,component='coolant-channel')
    assert not out.exists()


def test_symlink_parent_cannot_redirect_build_results(tmp_path):
    real=tmp_path/'real';real.mkdir();link=tmp_path/'link';link.symlink_to(real,target_is_directory=True)
    with pytest.raises(ValueError,match='ordinary'):delivery.run(link/'results',component='coolant-channel')
    assert not (real/'results').exists()


def test_old_cladding_entrypoint_preserves_selection_and_timeout(monkeypatch,tmp_path):
    calls=[]
    def accept(output,timeout_s,*,component):
        calls.append((output,timeout_s,component));return 'accepted'
    monkeypatch.setattr(legacy,'_run',accept)
    assert legacy.run(tmp_path/'out',123)=='accepted'
    assert calls==[(tmp_path/'out',123,'local-cladding')]


def test_larger_coupled_report_has_explicit_bound_without_changing_other_profiles(tmp_path):
    from atomic_model_multiphysics.provenance import seal
    from atomic_model_training.spec import atomic_json
    data={'complete_physical_state':'x'*(2*1024*1024)};path=tmp_path/'scientific.json'
    atomic_json(path,seal(data))
    assert delivery.read_scientific(path,'pwr-channel')==data
    with pytest.raises(ValueError,match='bounded'):delivery.read_scientific(path,'local-cladding')


def test_even_coupled_report_limit_is_finite(tmp_path):
    path=tmp_path/'oversized.json';path.write_bytes(b' '*(4*1024*1024+1))
    with pytest.raises(ValueError,match='bounded'):delivery.read_scientific(path,'pwr-channel')
