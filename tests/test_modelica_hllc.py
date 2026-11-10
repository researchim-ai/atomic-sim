"""External HLLC export/admission; native-derived rows below are harness fixtures."""
import pytest
from scripts import benchmark_modelica_hllc as bench
from scripts import benchmark_modelica_network as network
from scripts.verify_heterogeneous_network import as_peer


@pytest.fixture(scope='module')
def native():
    return bench.contact.capture(bench.request_for('thermal_contact',2,bench.CAPS[0]),'hllc')


def test_export_uses_independent_rh_flux_and_scalar_compiler_interface(native):
    text=bench.render(native)
    assert 'function AtomicHllc' in text and 'output Real massFlux' in text
    assert '(flux[1,k],flux[2,k],flux[3,k])=AtomicHllc' in text
    assert 'EStarL := ((SL-vL)*UL[3]-pL*vL+pStarL*SM)/(SL-SM)' in text
    assert 'pofdt125(rho,T,1e-12,1)' in text
    assert 'checkL := AtomicLiquidThermo' in text and 'checkR := AtomicLiquidThermo' in text
    assert '0.5*wave*(W[j,right[k]]-W[j,left[k]])' not in text
    assert 'der(faceEnergy[k])' in text and 'der(support[i])' in text


def test_unknown_template_cannot_be_silently_exported_as_hllc():
    with pytest.raises(ValueError):bench.transform_model('model Unknown end Unknown;')


def test_changed_legacy_flux_template_refused(native):
    text=network.render(native).replace('0.5*wave*(W[j,right[k]]-W[j,left[k]])','changed_flux')
    with pytest.raises(ValueError):bench.transform_model(text)


@pytest.mark.parametrize('case',bench.CASES)
def test_declared_case_has_closed_liquid_geometry_and_explicit_cap(case):
    r=bench.request_for(case,4,bench.CAPS[1])
    assert r['model']['max_step_s']==bench.CAPS[1]
    assert r['friction']=={}
    assert len(r['model']['edges'][0]['initial'])==4
    assert all(w['conductance_w_k']==0 for w in r['model']['walls'])


def test_reverse_and_wide_pressure_profiles_are_actual_distinct_states():
    reverse=bench.request_for('reverse_pressure_step',2,bench.CAPS[0])
    assert reverse['model']['nodes'][0]['initial']['pressure_pa']<reverse['model']['nodes'][1]['initial']['pressure_pa']
    wide=bench.request_for('wide_pressure_step',2,bench.CAPS[0])
    values={n['initial']['pressure_pa'] for n in wide['model']['nodes']}
    assert values=={2e6,14e6}


def test_contact_has_separate_stricter_gate_than_general_peer_agreement(native):
    peer=as_peer(native['observations'])
    assert bench.assess(native,peer,'thermal_contact')['passed']
    peer[2]['temperature[1]']+=1e-5
    result=bench.assess(native,peer,'thermal_contact')
    assert result['checks']['peer_agreement']
    assert not result['checks']['stationary_contact_preserved']
    assert not result['passed']


def test_wave_case_does_not_claim_stationary_contact_acceptance(native):
    result=bench.assess(native,as_peer(native['observations']),'wide_pressure_step')
    assert result['passed']
    assert 'stationary_contact_preserved' not in result['checks']


def test_diagnostic_msl_copy_refused_before_compiler_or_output(tmp_path):
    compiler=tmp_path/'omc';compiler.write_text('fixture, never executed')
    library=tmp_path/'library';(library/'Modelica').mkdir(parents=True)
    (library/'Modelica/package.mo').write_text('fixture')
    (library/'counterfactual.json').write_text('{}')
    with pytest.raises(ValueError):bench.run(tmp_path/'out',omc=compiler,library=library)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('case,cells,cap',[('other',2,5e-6),('thermal_contact',3,5e-6),('thermal_contact',2,.1)])
def test_unknown_matrix_selection_refused(case,cells,cap):
    with pytest.raises(ValueError):bench.request_for(case,cells,cap)


def test_missing_runtime_cannot_create_fake_evidence(tmp_path):
    with pytest.raises(ValueError):bench.run(tmp_path/'out',omc=tmp_path/'missing',library=tmp_path)
    assert not (tmp_path/'out').exists()
