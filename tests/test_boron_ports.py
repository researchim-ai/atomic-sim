from copy import deepcopy
from dataclasses import replace
import pytest

from atomic_model_multiphysics.provenance import seal,unseal
from physics.materials.iapws import properties_pt,saturation_properties
from coupling.boron_ports import assess_liquid_ports
from coupling.pwr_boron import BoronExchangeConfig,BoronExchangePWRSession
from coupling.refinement import coupled_pwr_mesh


def inputs(outlet_k=420.):
    supply=properties_pt(1e6,350.);donor=properties_pt(15.5e6,560.)
    desired=properties_pt(15.5e6,outlet_k)
    return dict(flow_kg_s=50.,primary_pressure_pa=15.5e6,supply_pressure_pa=1e6,receiver_pressure_pa=1e6,
        supply_enthalpy_j_kg=float(supply.h_j_kg),pump_work_j_kg=float(14.5e6/supply.rho_kg_m3/.85),
        donor_enthalpy_j_kg=float(donor.h_j_kg),recovered_heat_w=float(50.*(donor.h_j_kg-desired.h_j_kg)))


def test_if97_single_phase_ports_reconstruct_energy_and_produce_entropy():
    r=assess_liquid_ports(**inputs())
    assert r['passed'] and r['checked'] and all(r['gates'].values())
    assert r['states']['letdown_inlet']['temperature_k']==pytest.approx(420.,abs=1e-7)
    assert r['states']['letdown_inlet']['requested_enthalpy_j_kg']==r['states']['letdown_outlet']['requested_enthalpy_j_kg']
    assert all(x['region']==1 and x['enthalpy_residual_j_kg']<=.01 for x in r['states'].values())
    assert min(r['pump_entropy_w_k'],r['exchanger_entropy_w_k'],r['valve_entropy_w_k'])>0


def test_uncooled_letdown_flashes_without_projection():
    x=inputs();x['recovered_heat_w']=0.;r=assess_liquid_ports(**x)
    assert not r['passed'] and not r['gates']['liquid']
    outlet=r['states']['letdown_outlet']
    assert outlet['region']==4 and 0<outlet['quality']<1
    assert outlet['requested_enthalpy_j_kg']==x['donor_enthalpy_j_kg']


def test_exact_saturated_liquid_is_outside_positive_subcooling_envelope():
    x=inputs();liquid,_=saturation_properties(1e6)
    x['recovered_heat_w']=50.*(x['donor_enthalpy_j_kg']-float(liquid.h_j_kg))
    r=assess_liquid_ports(**x)
    assert not r['gates']['liquid']
    assert abs(r['states']['letdown_outlet']['subcooling_k'])<1e-7


@pytest.mark.parametrize('margin,accepted',[(.5,False),(2.,True)])
def test_declared_one_kelvin_margin_is_enforced(margin,accepted):
    x=inputs();liquid,_=saturation_properties(1e6);target=properties_pt(1e6,float(liquid.T_k)-margin)
    x['recovered_heat_w']=50.*(x['donor_enthalpy_j_kg']-float(target.h_j_kg))
    r=assess_liquid_ports(**x)
    assert r['gates']['liquid'] is accepted


def test_endpoint_temperature_crossing_is_rejected_without_clipping_heat():
    x=inputs(330.);r=assess_liquid_ports(**x)
    assert not r['passed'] and not r['gates']['endpoint_pinches']
    assert r['states']['letdown_inlet']['temperature_k']==pytest.approx(330.,abs=1e-7)


def test_zero_flow_has_no_fictitious_port_qualification():
    x=inputs();x.update(flow_kg_s=0.,recovered_heat_w=0.)
    r=assess_liquid_ports(**x)
    assert r['passed'] and not r['checked'] and not r['active'] and r['states']=={}
    x['recovered_heat_w']=1.
    with pytest.raises(ValueError,match='Zero flow'):assess_liquid_ports(**x)


def test_mutating_report_cannot_poison_exact_state_cache():
    x=inputs();before=assess_liquid_ports(**x);changed=assess_liquid_ports(**x)
    changed['states']['letdown_outlet']['temperature_k']=-1.
    assert assess_liquid_ports(**x)==before
    x['donor_enthalpy_j_kg']+=1.
    assert assess_liquid_ports(**x)['states']['primary_donor']['enthalpy_j_kg']!=before['states']['primary_donor']['enthalpy_j_kg']


@pytest.mark.parametrize('change',[dict(flow_kg_s=True),dict(primary_pressure_pa=17e6),dict(receiver_pressure_pa=16e6),
    dict(supply_enthalpy_j_kg=float('nan')),dict(pump_work_j_kg=-1.),dict(margin_k=0.)])
def test_invalid_inputs_rejected(change):
    x=inputs();x.update(change)
    with pytest.raises(ValueError):assess_liquid_ports(**x)


@pytest.mark.parametrize('value',[0,1,None,'true'])
def test_port_policy_requires_a_boolean(value):
    with pytest.raises(ValueError,match='boolean'):replace(BoronExchangeConfig(),require_liquid_ports=value)


@pytest.fixture(scope='module')
def strict_seed():
    return BoronExchangePWRSession(**coupled_pwr_mesh(2),boron=BoronExchangeConfig(
        pressure_pa=1e6,regenerator_ua_w_k=500000.,require_liquid_ports=True),max_step_s=.05,rtol=2e-9)


def test_strict_policy_checks_live_ports_and_preserves_registry_schema(strict_seed):
    s=deepcopy(strict_seed);r=s.advance(.05,{'boron_flow_target_kg_s':50.})
    assert r['finite_boron_exchange']['liquid_ports']['checked']
    assert r['finite_boron_exchange']['liquid_ports']['passed']
    assert r['energy_relative']<=1e-9 and r['mass_relative']<=1e-10 and r['boron_relative']<=1e-10
    registry=s.parameter_registry()
    from atomic_model_station.foundation import validate_registry
    assert validate_registry(registry)
    assert s.exchange_policy()['require_liquid_ports'] is True
    assert r['boron_exchange_policy']==s.exchange_policy()
    assert 'boron_exchange.require_liquid_ports' not in registry['parameters']


def test_phase_failure_rolls_back_full_step_and_forbids_policy_control_patch():
    s=BoronExchangePWRSession(**coupled_pwr_mesh(2),boron=BoronExchangeConfig(pressure_pa=1e6,require_liquid_ports=True))
    before=s.checkpoint()
    with pytest.raises(ValueError,match='Unsupported liquid exchange ports'):s.advance(.1,{'boron_flow_target_kg_s':50.})
    assert s.checkpoint()==before
    with pytest.raises(ValueError):s.advance(.1,{'require_liquid_ports':False})
    assert s.checkpoint()==before


def test_checkpoint_policy_tampering_is_not_silently_adopted(strict_seed):
    s=deepcopy(strict_seed);s.advance(.05,{'boron_flow_target_kg_s':50.});p=unseal(s.checkpoint())
    p['config']['require_liquid_ports']=False
    with pytest.raises(ValueError,match='replay differs'):BoronExchangePWRSession.restore(seal(p))


@pytest.mark.parametrize('cases,legacy',[({},{}),({'dilution':{},'boration':{}},{})])
def test_incomplete_liquid_port_matrix_cannot_be_accepted(cases,legacy):
    from scripts.qualify_boron_ports import assess
    with pytest.raises(ValueError,match='Complete'):assess(cases,legacy,{})
