from copy import deepcopy
import pytest
from scripts import qualify_numerical_reuse as reuse


def test_changed_dependency_cannot_reuse_prior_full_qualification(tmp_path,monkeypatch):
    source=deepcopy(reuse.source());source['dependency_qualification_digests']['pwr']='0'*64
    monkeypatch.setattr(reuse,'source',lambda:source)
    out=tmp_path/'new'
    with pytest.raises(ValueError,match='source changed'):reuse.run(out)
    assert not out.exists()


def test_prior_results_are_not_overwritten(tmp_path):
    saved=tmp_path/'evidence';saved.write_text('keep')
    with pytest.raises(ValueError,match='Fresh'):reuse.run(tmp_path)
    assert saved.read_text()=='keep'
