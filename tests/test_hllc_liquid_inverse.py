"""Exact first-chart EOS optimization; boundary decisions stay in old adapter."""
import numpy as np
import pytest
from atomic_model_hem_contact.inverse import liquid_first_rho_u
from atomic_model_hem_qualified import eos


@pytest.mark.parametrize('pressure',[1.1e5,2e5,1e6,4e6,10e6,15.5e6])
@pytest.mark.parametrize('fraction',[.05,.5,.95])
@pytest.mark.parametrize('offset',[(1.,0.),(1.,2.),(1.005,-1.)])
def test_liquid_newton_matches_complete_state_exactly(pressure,fraction,offset):
    temperature=300.+(eos.saturation(pressure)[0].temperature_k-301.)*fraction
    state=eos.pt(pressure,temperature)
    args=(state.density_kg_m3,state.internal_energy_j_kg,pressure*offset[0],temperature+offset[1])
    expected=eos.rho_u(*args);actual=liquid_first_rho_u(*args)
    assert actual==expected and eos._matches(actual,args[0],args[1])


@pytest.mark.parametrize('pressure',[1e6,4e6,15.5e6])
@pytest.mark.parametrize('quality',[0.,1e-6,.5,1.])
def test_phase_and_endpoint_decisions_delegate_unchanged(pressure,quality,monkeypatch):
    state=eos.px(pressure,quality)
    args=(state.density_kg_m3,state.internal_energy_j_kg,pressure,state.temperature_k)
    expected=eos.rho_u(*args);original=eos.rho_u;calls=[]
    def full(*args,**kwargs):calls.append(args);return original(*args,**kwargs)
    monkeypatch.setattr(eos,'rho_u',full)
    assert liquid_first_rho_u(*args)==expected
    assert len(calls)==1


@pytest.mark.parametrize('epsilon',[2e-8,1e-7,1e-5])
def test_near_saturation_liquid_uses_same_forward_residual(epsilon):
    p=4e6;state=eos.pt(p,eos.saturation(p)[0].temperature_k-epsilon)
    args=(state.density_kg_m3,state.internal_energy_j_kg,p,state.temperature_k)
    assert liquid_first_rho_u(*args)==eos.rho_u(*args)


def test_failed_fast_attempt_retries_unchanged_full_adapter(monkeypatch):
    state=eos.pt(4e6,440.);args=(state.density_kg_m3,state.internal_energy_j_kg,4.01e6,442.)
    expected=eos.rho_u(*args);raw=eos._raw;full=eos.rho_u;calls=[];attempts=[]
    def probe(*args):
        attempts.append(args)
        if len(attempts)==1:raise ValueError('Injected failed fast attempt')
        return raw(*args)
    def qualified(*args):calls.append(args);return full(*args)
    monkeypatch.setattr(eos,'_raw',probe);monkeypatch.setattr(eos,'rho_u',qualified)
    assert liquid_first_rho_u(*args)==expected and calls==[args]


def test_reused_gibbs_values_reduce_work_without_changing_candidate(monkeypatch):
    state=eos.pt(4e6,440.);args=(state.density_kg_m3,state.internal_energy_j_kg,4.01e6,442.)
    eos.rho_u(*args) # identical saturation cache warmup for both paths
    raw=eos._raw;calls=[]
    def probe(*args):calls.append(args);return raw(*args)
    monkeypatch.setattr(eos,'_raw',probe)
    before=eos.rho_u(*args);old_count=len(calls);calls.clear()
    after=liquid_first_rho_u(*args);new_count=len(calls)
    assert before==after and 0<new_count<old_count


@pytest.mark.parametrize('args',[(0.,7e5,4e6,440.),(-1.,7e5,4e6,440.),(float('nan'),7e5,4e6,440.),(900.,float('inf'),4e6,440.),(900.,7e5,1.,440.)])
def test_invalid_conserved_inputs_are_not_projected(args):
    with pytest.raises((ValueError,TypeError)):liquid_first_rho_u(*args)
