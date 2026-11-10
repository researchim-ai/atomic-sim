"""Direct HLLC transport retains every RHS bit and nontransport guard."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from atomic_model_hem_contact import ContactHEMNetwork
from atomic_model_hem_qualified.model import QualifiedHEMNetwork
from atomic_model_multiphysics.provenance import unseal,sha
from scripts.benchmark_modelica_network import request_for
from scripts.benchmark_hllc_speed import equivalent


@pytest.fixture(scope='module')
def baseline():
    root=Path(__file__).resolve().parents[1]/'docs/benchmarks/baselines'
    package=root/'hem_contact_v15'
    expected=unseal(json.loads((root/'hem_contact_v15_source.json').read_text()))['source']['contact']
    assert {p.name:sha(p) for p in package.glob('*.py')}==expected
    # Both implementations run on the current foundation. No environment-pinned
    # baseline certificate is claimed by these portable regression tests.
    name='_atomic_rhs_baseline_test'
    spec=importlib.util.spec_from_file_location(name,package/'__init__.py',submodule_search_locations=[str(package)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module.ContactHEMNetwork


@pytest.mark.parametrize('scheme',['hllc','hllc-local','rusanov'])
@pytest.mark.parametrize('case',['thermal_step','pressure_thermal_step'])
def test_rhs_and_aux_are_bitwise_equal_to_previous_wrapper(baseline,scheme,case):
    request=request_for(case,4,1e-5);old=baseline(request,scheme=scheme);new=ContactHEMNetwork(request,scheme=scheme)
    for _ in range(2):
        a,aa=old.evaluate(old._y,old._controls);b,bb=new.evaluate(new._y,new._controls)
        np.testing.assert_array_equal(a.view(np.uint64),b.view(np.uint64))
        assert aa.keys()==bb.keys()
        for key in aa:np.testing.assert_array_equal(aa[key].view(np.uint64),bb[key].view(np.uint64))
        assert old.advance(.00025)==new.advance(.00025)


@pytest.mark.parametrize('scheme',['hllc','hllc-local'])
def test_steady_hllc_evaluation_no_longer_calls_parent_transport(monkeypatch,scheme):
    model=ContactHEMNetwork(request_for('pressure_thermal_step'),scheme=scheme)
    def forbidden(*args,**kwargs):raise AssertionError('Discarded transport evaluated')
    monkeypatch.setattr(QualifiedHEMNetwork,'_evaluate_point',forbidden)
    rhs,_=model.evaluate(model._y,model._controls)
    assert np.isfinite(rhs).all()


def test_rusanov_still_uses_existing_parent_branch(monkeypatch):
    model=ContactHEMNetwork(request_for('thermal_step'),scheme='rusanov')
    def sentinel(*args,**kwargs):raise RuntimeError('Original parent branch')
    monkeypatch.setattr(QualifiedHEMNetwork,'_evaluate_point',sentinel)
    with pytest.raises(RuntimeError,match='Original parent'):model.evaluate(model._y,model._controls)


@pytest.mark.parametrize('scheme',['hllc','hllc-local'])
def test_stationary_node_momentum_guard_remains_active(scheme):
    model=ContactHEMNetwork(request_for('thermal_step'),scheme=scheme)
    before=model.checkpoint();trial=model._y.copy();trial[model.n]=1e-6
    with pytest.raises(ValueError,match='Stationary node acquired momentum'):model.evaluate(trial,model._controls)
    assert model.checkpoint()==before


def test_exact_result_check_does_not_hide_signed_zero():
    def packet(value):return dict(observations=[dict(x=value)],checkpoint=dict(payload=dict(source={},x=value)))
    assert equivalent(packet(0.),packet(-0.))==dict(observations_exact=False,numerical_checkpoint_exact=False)
