"""Native export and guard protocols; fixture peer rows are harness-only."""
import pytest
from scripts import benchmark_modelica_local_hllc as local
from scripts import benchmark_modelica_hllc as base
from scripts import verify_modelica_local_guards as guards
from scripts.verify_heterogeneous_network import as_peer


@pytest.fixture(scope='module')
def native():
    return local.contact.capture(local.request_for('thermal_contact',2,local.CAPS[0]),local.SCHEME)


def test_native_kernel_has_local_signals_both_star_guards_and_global_cap(native):
    text=local.render(native)
    assert 'bound := 1.01*max(abs(vL)+cL,abs(vR)+cR)' in text
    assert 'assert(bound<=maximum,' in text
    assert 'abs(SM)+checkL[4]<=bound and abs(SM)+checkR[4]<=bound' in text
    assert 'EStarL := ((SL-vL)*UL[3]-pL*vL+pStarL*SM)/(SL-SM)' in text
    assert 'pofdt125(rho,T,1e-12,1)' in text
    assert '0.5*wave*(W[j,right[k]]-W[j,left[k]])' not in text
    assert text.count('function AtomicHllc')==1


def test_previous_fixed_export_is_not_replaced_by_local_signals(native):
    text=base.render(native)
    assert 'bound := 1.01*max' not in text
    assert 'External local stars' not in text
    assert 'SL := -bound; SR := bound;' in text


def test_fixed_scheme_packet_cannot_be_exported_as_local():
    packet=base.contact.capture(base.request_for('thermal_contact',2,base.CAPS[0]),'hllc')
    with pytest.raises(ValueError,match='Local HLLC native'):local.render(packet)


def test_unknown_base_kernel_cannot_silently_keep_fixed_flux(native,monkeypatch):
    monkeypatch.setattr(base,'render',lambda native:'model Unknown end Unknown;')
    with pytest.raises(ValueError,match='Unknown external'):local.render(native)


def test_contact_gate_remains_stricter_than_peer_budget(native):
    peer=as_peer(native['observations'])
    assert local.assess(native,peer,'thermal_contact')['passed']
    peer[2]['temperature[1]']+=1e-5
    result=local.assess(native,peer,'thermal_contact')
    assert result['checks']['peer_agreement'] and not result['checks']['stationary_contact_preserved']
    assert not result['passed']


@pytest.mark.parametrize('runner',[local.run,guards.run])
def test_diagnostic_library_refused_without_creating_evidence(runner,tmp_path):
    compiler=tmp_path/'omc';compiler.write_text('fixture; never execute')
    library=tmp_path/'library';(library/'Modelica').mkdir(parents=True)
    (library/'Modelica/package.mo').write_text('fixture')
    (library/'counterfactual.json').write_text('{}')
    with pytest.raises(ValueError):runner(tmp_path/'out',omc=compiler,library=library)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('runner',[local.run,guards.run])
def test_missing_runtime_cannot_become_external_evidence(runner,tmp_path):
    with pytest.raises(ValueError):runner(tmp_path/'out',omc=tmp_path/'missing',library=tmp_path)
    assert not (tmp_path/'out').exists()


def test_negative_probes_are_dynamic_and_guards_are_distinct():
    text=guards.render(8.,2500.)
    assert 'velocity=8+1e-6*time' in text
    assert '(massFlux,momentumFlux,energyFlux)=AtomicHllc(' in text
    assert [p[3] for p in guards.PROBES]==[None,'External local stars exceed signal envelope','External local envelope exceeds global CFL cap']
    assert 'Python' not in text
