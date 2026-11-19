from pathlib import Path
import pytest
from atomic_model_training.spec import atomic_json
from atomic_model_multiphysics.provenance import seal
from scripts import qualify_station_mixed_convection as study
from scripts.qualify_station_inventory import read


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    root=tmp_path_factory.mktemp('mixed-study')
    assert study.run(root/'source')['passed']
    assert study.run(root/'repeat',root/'source')['passed']
    return root/'source'


def test_whole_study_repeats_all_scientific_json_and_does_not_accept_the_station(evidence):
    r=read(evidence/'report.json')
    assert r['passed'] and r['source_unchanged'] and not r['candidate_station_usable']
    assert r['closed_tasks']==[] and not r['npp_11_03_accepted'] and not r['validated']
    assert r['maximum_nusselt_relative']<2e-4 and r['minimum_observed_order']>1.8
    assert {p.name for p in evidence.glob('*.json')}=={'report.json','roots.json','laminar.json','station-screen.json','protocol.json'}


def test_boundary_sample_cannot_be_qualified_by_direction_guess_or_laminar_substitution(evidence):
    r=read(evidence/'station-screen.json')
    assert r['laminar_reference_rejected'] and not r['station_usable']
    assert r['groups']['reynolds']>2000 and r['reference_temperature_groups']['rayleigh_film']<3e7
    for case in r['opposing_candidates'].values():
        assert all(not c['eligible_for_reported_box_comparison'] for c in case.values())
    assert len(r['iaea_framework_roots'])==3


def test_tampered_reference_protocol_is_rejected_before_execution(evidence,tmp_path):
    bad=read(evidence/'protocol.json');bad['criteria']['nusselt_relative']=1.
    directory=tmp_path/'bad';directory.mkdir();atomic_json(directory/'protocol.json',seal(bad))
    with pytest.raises(ValueError,match='identity'):study.run(tmp_path/'out',directory)
    assert not (tmp_path/'out').exists()


def test_independent_error_cannot_be_accepted_as_grid_convergence(tmp_path,monkeypatch):
    original=study.collocation
    def wrong(*a,**k):
        r=original(*a,**k);r['nusselt']*=1.01;return r
    monkeypatch.setattr(study,'collocation',wrong)
    with pytest.raises(ValueError,match='qualification failed'):study.run(tmp_path/'wrong')
    assert not read(tmp_path/'wrong/report.json')['passed']


def test_study_preserves_current_source_and_acknowledges_only_exact_cache_migration(tmp_path,monkeypatch):
    from atomic_model_station_faults import FaultSession
    from scripts.qualify_station_shutdown import identity
    source=read(Path(__file__).resolve().parents[1]/'docs/benchmarks/station_low_flow_v63.json')['source']
    from tests._station_history import assert_source_compatible
    assert_source_compatible(source,identity())
    source=identity()
    def forbidden(*a,**k):raise AssertionError('Study must not advance station time')
    monkeypatch.setattr(FaultSession,'step',forbidden)
    assert study.run(tmp_path/'readonly')['passed']
    assert identity()==source
