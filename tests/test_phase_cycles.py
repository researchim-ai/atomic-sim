"""Exact-stock phase cycles and independent forward-chart reference contracts."""
from copy import deepcopy
import numpy as np
import pytest
from atomic_model_hem_qualified import eos
from atomic_model_hem_qualified.model import QualifiedHEMNetwork
from scripts.verify_phase_cycles import (
    BUDGETS, CASES, compare_snapshot, execute, forward_reference, offset_at, request_for,
)


@pytest.mark.parametrize('endpoint',[0.,1.])
@pytest.mark.parametrize('offset',[-50.,0.,50.])
def test_forward_reference_satisfies_stocks_without_inverse(monkeypatch,endpoint,offset):
    saturated=eos.px(1e6,endpoint)
    def forbidden(*args,**kwargs):
        raise AssertionError('Reference cannot call production inverse EOS')
    monkeypatch.setattr(eos,'rho_u',forbidden)
    state=forward_reference(endpoint,offset)
    assert state.density_kg_m3==pytest.approx(saturated.density_kg_m3,rel=2e-12)
    assert state.internal_energy_j_kg==pytest.approx(saturated.internal_energy_j_kg+offset,abs=2e-6,rel=0)
    assert state.phase == ('two_phase' if offset<0 else ('liquid' if endpoint==0 else 'vapor') if offset>0 else saturated.phase)


@pytest.mark.parametrize('start',[-50.,50.])
def test_cycle_energy_path_hits_endpoint_twice_and_returns(start):
    assert [offset_at(start,t) for t in (0.,.0005,.001,.0015,.002)]==[start,0.,-start,0.,start]
    with pytest.raises(ValueError): offset_at(start,.003)


@pytest.mark.parametrize('case',list(CASES))
def test_actual_cycle_crosses_and_returns_with_exact_stocks(case):
    result=execute(case,2,2e-5)
    assert result['record']['passed'],result['record']
    assert len(result['observations'])==5
    assert all(e<=BUDGETS[k] for k,e in result['record']['maximum_errors'].items())


def test_cycle_comparison_detects_mass_energy_and_velocity_error():
    initial=QualifiedHEMNetwork(request_for('vapor_exit')).snapshot()
    reference=forward_reference(1.,-50.)
    for field,budget in [('mass_kg','mass_kg'),('total_energy_j','total_energy_j'),('velocity_m_s','velocity_m_s')]:
        changed=deepcopy(initial)
        changed['cells'][0][field]+=10*BUDGETS[budget]
        assert compare_snapshot(changed,initial,reference,0.)[budget]>BUDGETS[budget]
    changed=deepcopy(initial)
    changed['cells'][0]['pressure_pa']=float('nan')
    with pytest.raises(ValueError): compare_snapshot(changed,initial,reference,0.)


@pytest.mark.parametrize('case,cells,cap',[('unknown',2,2e-5),('vapor_exit',8,2e-5),('vapor_exit',2,0.)])
def test_undeclared_program_refused(case,cells,cap):
    with pytest.raises(ValueError): request_for(case,cells,cap)


def test_restart_exactly_after_crossing_and_before_return():
    result=execute('vapor_exit',4,1e-5)
    assert result['record']['checks']['restart_exact']
    assert result['record']['checks']['exact_external_heat']


def test_quality_comparison_only_allows_one_sided_exact_endpoints():
    from scripts.verify_phase_cycles import quality_error
    endpoint=eos.px(1e6,0.)
    assert quality_error(dict(phase='liquid',quality=None),endpoint)==0
    with pytest.raises(ValueError): quality_error(dict(phase='vapor',quality=None),endpoint)
    mixture=eos.px(1e6,.1)
    with pytest.raises(ValueError): quality_error(dict(phase='liquid',quality=None),mixture)
    with pytest.raises(ValueError): quality_error(dict(phase='two_phase',quality=float('nan')),mixture)
    liquid=forward_reference(0.,50.)
    with pytest.raises(ValueError): quality_error(dict(phase='two_phase',quality=0.),liquid)


@pytest.mark.parametrize('pressure',[.5e6,8e6,15.5e6])
@pytest.mark.parametrize('endpoint',[0.,1.])
@pytest.mark.parametrize('offset',[-50.,50.])
def test_forward_reference_pressure_matrix_without_inverse(monkeypatch,pressure,endpoint,offset):
    saturated=eos.px(pressure,endpoint)
    def forbidden(*args,**kwargs): raise AssertionError('Production inverse used by reference')
    monkeypatch.setattr(eos,'rho_u',forbidden)
    state=forward_reference(endpoint,offset,pressure_pa=pressure)
    assert state.density_kg_m3==pytest.approx(saturated.density_kg_m3,rel=2e-12)
    assert state.internal_energy_j_kg==pytest.approx(saturated.internal_energy_j_kg+offset,abs=2e-6,rel=0)


@pytest.mark.parametrize('case',list(CASES))
def test_actual_pwr_pressure_phase_cycle(case):
    result=execute(case,2,2e-5,pressure_pa=15.5e6)
    assert result['record']['passed'],result['record']


@pytest.mark.parametrize('endpoint',[0.,1.])
def test_full_cycle_preflight_refuses_upper_domain_without_creating_output(endpoint,tmp_path):
    from scripts.verify_phase_cycles import verify
    with pytest.raises(ValueError): forward_reference(endpoint,50.,pressure_pa=16e6)
    case='liquid_exit' if endpoint==0 else 'vapor_exit'
    # Initially admissible mixture cannot conceal an inadmissible return path.
    with pytest.raises(ValueError): request_for(case,pressure_pa=16e6)
    output=tmp_path/'refused'
    with pytest.raises(ValueError): verify(output,pressures=(16e6,))
    assert not output.exists()


def test_domain_exit_is_transactional_without_stock_projection():
    from atomic_model_hem_qualified.cases import baseline
    request=baseline(0.,velocity=0.,pressure=16e6-1000.)
    request['model']['walls'][0]['conductance_w_k']=0.
    model=QualifiedHEMNetwork(request)
    before=model.checkpoint()
    observation=model.snapshot()
    controls=model.control()
    controls['fluid']={c['id']:c['mass_kg']*5e4 for c in observation['cells']}
    with pytest.raises(ValueError):
        model.step(controls,.001,expected_revision=0,controller_state={'rejected':True})
    assert model.checkpoint()==before
    assert model.snapshot()==observation
    controls['fluid']={name:-power for name,power in controls['fluid'].items()}
    clean=QualifiedHEMNetwork(request)
    resumed=model.step(controls,.0001,expected_revision=0,controller_state={'recovered':True})
    expected=clean.step(controls,.0001,expected_revision=0,controller_state={'recovered':True})
    assert resumed==expected and model.checkpoint()==clean.checkpoint()


@pytest.mark.parametrize('pressures',[(),(1e6,1e6),(float('nan'),),(.1e6,), (True,)])
def test_undeclared_pressure_campaign_refused_before_output(tmp_path,pressures):
    from scripts.verify_phase_cycles import verify
    output=tmp_path/'refused'
    with pytest.raises(ValueError): verify(output,pressures=pressures)
    assert not output.exists()


@pytest.mark.parametrize('pressure',[eos.P_MIN,eos.P_MIN+50.,eos.P_MAX-50.,eos.P_MAX])
def test_dynamic_characteristics_refuse_static_domain_margin(pressure):
    from atomic_model_hem_qualified.cases import baseline
    with pytest.raises(ValueError,match='Pressure derivative margin'):
        QualifiedHEMNetwork(baseline(0.,velocity=0.,pressure=pressure))
