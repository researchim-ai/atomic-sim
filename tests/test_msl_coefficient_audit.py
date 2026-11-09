"""Frozen direct external measurements and explicit counterfactual provenance."""
import json
from pathlib import Path
import pytest
from atomic_model_multiphysics.provenance import unseal,sha
from physics.materials import iapws
from scripts import audit_msl_if97 as audit

DATA=unseal(json.loads(Path(__file__).with_name('fixtures').joinpath('msl_gibbs_delta_v1.json').read_text()))


@pytest.mark.parametrize('row',DATA['records'])
def test_analytic_coefficient_delta_explains_actual_external_measurement(row):
    expected=audit.delta(row['t'])
    for key,value in expected.items():
        assert abs(value-row['measured_difference'][key])<=audit.BUDGETS[key]


def test_kernel_uses_normative_stable_table_not_observed_peer_coefficients():
    assert tuple(iapws._R2_N0)==audit.N
    assert tuple(iapws._R2_J0)==audit.J
    assert all(a!=b for a,b in zip(audit.MSL,audit.N,strict=True))


@pytest.mark.parametrize('temperature',[float('nan'),float('inf'),299.,901.])
def test_coefficient_audit_outside_declared_domain_refused(temperature):
    with pytest.raises(ValueError):audit.delta(temperature)


def test_counterfactual_copy_preserves_original_and_changes_only_gibbs_file(tmp_path):
    original=tmp_path/'original';target=tmp_path/'corrected'
    p=original/audit.FILE;p.parent.mkdir(parents=True)
    text='      function g2 "fixture"\n g.g := '+' + '.join(audit.MSL_LITERALS)+';\n g.gpi := 0;\n      end g2;\n'
    p.write_text(text)
    other=original/'Modelica/unrelated.mo';other.write_text('unchanged')
    before=sha(p)
    audit.corrected_copy(original,target)
    assert sha(p)==before and other.read_text()==(target/'Modelica/unrelated.mo').read_text()
    receipt=unseal(json.loads((target/'counterfactual.json').read_text()))
    assert receipt['counterfactual'] and not receipt['baseline_modified']
    assert receipt['changed_files']==[audit.FILE]
    assert 'EXPLICIT COUNTERFACTUAL' in (target/audit.FILE).read_text()
    with pytest.raises(ValueError):audit.corrected_copy(original,target)


def test_unrecognized_coefficients_cannot_be_patched(tmp_path):
    p=tmp_path/'source'/audit.FILE;p.parent.mkdir(parents=True)
    p.write_text('      function g2 "fixture"\n g.g := 0; g.gpi := 0;\n      end g2;')
    with pytest.raises(ValueError):audit.corrected_copy(tmp_path/'source',tmp_path/'new')
    assert not (tmp_path/'new').exists()


def test_enthalpy_delta_and_heat_capacity_delta_obey_dh_dT():
    t=650.;dt=.01
    derivative=(audit.delta(t+dt)['h']-audit.delta(t-dt)['h'])/(2*dt)
    assert derivative==pytest.approx(audit.delta(t)['cp'],abs=1e-11)
