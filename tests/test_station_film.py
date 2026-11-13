import json
from types import SimpleNamespace

import numpy as np
import pytest

from atomic_model_station import FilmConfig, PWRFilmSession, PWRPrimarySession, CoreConfig
from atomic_model_station.film import film_transfer, phase_inventory
from atomic_model_station.foundation import validate_registry
from atomic_model_plant_extension.volume import StrictVolume
from atomic_model_plant_extension import water
from atomic_model_multiphysics.provenance import seal, unseal
from core.status import DomainError


def algebraic_case():
    state = SimpleNamespace(region=1, T_k=500., viscosity_pa_s=.001,
                            conductivity_w_m_k=.6, cp_j_kg_k=600.)
    geometry = dict(diameter_m=.01, length_m=1., flow_area_m2=.01, heated_area_m2=4.)
    return state, geometry


def test_dittus_arithmetic_independent_reference_at_unit_prandtl():
    state, geometry = algebraic_case()
    r = film_transfer(state, 100., geometry, 510.)
    assert r['reynolds'] == pytest.approx(100000.) and r['prandtl'] == pytest.approx(1.)
    assert r['nusselt'] == pytest.approx(230.)
    assert r['coefficient_w_m2_k'] == pytest.approx(13800.)
    assert r['heat_w'] == pytest.approx(552000.)
    assert r['chf_w_m2'] is None and r['dnbr'] is None


def test_film_changes_with_flow_instead_of_fixed_ua():
    state, geometry = algebraic_case()
    a = film_transfer(state, 100., geometry, 510.)
    b = film_transfer(state, 200., geometry, 510.)
    assert b['conductance_w_k']/a['conductance_w_k'] == pytest.approx(2**.8)


@pytest.mark.parametrize('flow', [0., -10., 1., float('nan')])
def test_unsupported_flow_rejected_without_transition_extrapolation(flow):
    state, geometry = algebraic_case()
    with pytest.raises((ValueError, DomainError)):
        film_transfer(state, flow, geometry, 510.)


def test_vapor_and_reverse_heat_transfer_are_not_admitted_as_heating():
    state, geometry = algebraic_case()
    with pytest.raises(ValueError, match='Heating-only'):
        film_transfer(state, 100., geometry, 490.)
    state.region = 2
    with pytest.raises(ValueError, match='liquid'):
        film_transfer(state, 100., geometry, 510.)


def test_geometry_sizing_preserves_volume_nominal_ua_and_initial_stocks():
    s = PWRFilmSession(); old = PWRPrimarySession(); g = s.film_geometry
    assert np.array_equal(s.state, old.state)
    assert s._stored() == old._stored()
    assert g['flow_area_m2']*g['length_m'] == pytest.approx(g['core_volume_m3'])
    assert g['heated_area_m2']*g['diameter_m']/4 == pytest.approx(g['core_volume_m3'])
    assert s.snapshot()['film']['conductance_w_k'] == pytest.approx(s.plant.clad_coolant_ua, rel=1e-12)
    assert validate_registry(s.parameter_registry())


def test_live_coastdown_reduces_ua_and_closes_first_law():
    s=PWRFilmSession(max_step_s=.2); initial=s.snapshot()
    r=s.advance(1.,dict(scram=True,turbine_trip=True,primary_pump_on=False))
    assert r['film']['conductance_w_k'] < initial['film']['conductance_w_k']
    assert r['energy_relative'] < 1e-8 and abs(r['instantaneous_energy_defect_w']) < .01
    assert r['film']['heat_w']==r['clad_to_coolant_w']
    assert min(r['phase_inventory'][n]['subcooling_k'] for n in ['core','hot_leg','sg_primary','cold_leg'])>.1


def test_equilibrium_phase_mass_and_energy_belong_to_existing_inventory():
    node=StrictVolume('test',1.,0.)
    node.set_quality(6e6,.2)
    r=phase_inventory(node)
    assert r['quality']==pytest.approx(.2,abs=1e-9)
    assert r['liquid_mass_kg']+r['vapor_mass_kg']==pytest.approx(node.mass_kg)
    assert r['mixture_energy_residual_j_kg']<.01
    assert r['mixture_density_relative_residual']<1e-8


def substitute_phase(s, name, quality):
    y=s.state.copy();node=s.plant.network.volumes[name]
    point=water.px(node.pressure_pa,quality)
    i=13+3*s.plant.network.volume_names().index(name)
    y[i]=node.volume_m3*point.density_kg_m3
    y[i+1]=y[i]*point.internal_energy_j_kg
    return y


@pytest.mark.parametrize('quality', [0.,1-1e-9])
def test_sg_dry_or_no_vapor_reserve_is_refused(quality):
    s=PWRFilmSession()
    with pytest.raises(ValueError, match='reserve'):
        s._admit(substitute_phase(s,'sg_secondary',quality))


def test_tiny_primary_quality_is_not_promoted_to_a_boiling_model():
    s=PWRFilmSession()
    with pytest.raises(ValueError,match='liquid'):
        s._admit(substitute_phase(s,'core',1e-6))


def test_json_restart_preserves_live_film_and_foreign_backend_is_refused():
    s=PWRFilmSession(max_step_s=.2);s.advance(.2,dict(scram=True,turbine_trip=True))
    saved=json.loads(json.dumps(s.checkpoint())); t=PWRFilmSession.restore(saved)
    assert t.advance(.2)==s.advance(.2)
    with pytest.raises((ValueError,KeyError)):
        PWRPrimarySession.restore(saved)
    payload=unseal(saved);payload['film_source']['correlation_sha256']='0'*64
    with pytest.raises(ValueError,match='source'):
        PWRFilmSession.restore(seal(payload))


def test_invalid_command_preserves_full_film_checkpoint():
    s=PWRFilmSession();saved=s.checkpoint()
    with pytest.raises(ValueError):s.advance(.1,dict(scram=True,load_fraction=2.))
    assert s.checkpoint()==saved


@pytest.mark.parametrize('values',[dict(heated_length_m=True),dict(minimum_subcooling_k=float('nan')),
                                    dict(minimum_phase_fraction=0.)])
def test_invalid_film_configuration(values):
    with pytest.raises(ValueError):FilmConfig(**values)


def test_spatial_owner_remains_compatible_with_flow_dependent_heat_transfer():
    s=PWRFilmSession(core=CoreConfig(kind='diffusion',cells=2),max_step_s=.2)
    r=s.advance(.2)
    assert len(r['core']['owned_state'])==14
    assert r['energy_relative']<1e-8
