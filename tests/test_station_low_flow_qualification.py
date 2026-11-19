import pytest
from atomic_model_station_shutdown import ShutdownCase
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from scripts import qualify_station_shutdown as study
from scripts import qualify_station_low_flow as quality
from scripts.qualify_station_inventory import read


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    root=tmp_path_factory.mktemp('low-flow-quality')
    case=ShutdownCase('coastdown',tick_s=.01,duration_s=.04,film_model='gnielinski')
    for name,(step,rtol) in zip(('coarse','fine','finer'),quality.SETTINGS):
        assert study.run(root/name,case,max_step_s=step,rtol=rtol)['passed']
    return root


def test_native_reexecutes_prefix_and_every_new_suffix_transition(runs):
    r=quality.native(runs/'native',runs/'coarse')
    assert r['passed'] and r['prefix_exact'] and r['suffix_and_final_exact']
    assert r['prefix_s']==.02 and r['suffix_s']==.02 and not r['full_horizon']
    assert len(read(runs/'native/suffix.json')['records'])==2


def test_refinement_accepts_short_agreement_without_claiming_full_horizon(runs):
    r=quality.refinement(runs/'refinement',[runs/n for n in ('coarse','fine','finer')])
    assert r['passed'] and not r['full_horizon'] and not r['npp_11_03_accepted']
    assert all(v['passed'] for v in r['refinement'].values())
    assert r['cases']['fine']['film_ranges']['reynolds'][0]>3000


def test_midpath_applicability_tampering_is_rejected_even_with_a_good_report(runs,tmp_path):
    import shutil
    shutil.copytree(runs/'coarse',tmp_path/'tampered')
    path=tmp_path/'tampered/trajectory.json';data=read(path)
    data['records'][1]['physical']['film']['richardson_d']=.1
    atomic_json(path,seal(data))
    with pytest.raises(ValueError,match='applicability'):quality.inspect(tmp_path/'tampered')


def test_wrong_numerical_setting_cannot_be_used_as_independent_refinement(runs,tmp_path):
    with pytest.raises(ValueError,match='settings differ'):
        quality.refinement(tmp_path/'wrong',[runs/'coarse']*3)
