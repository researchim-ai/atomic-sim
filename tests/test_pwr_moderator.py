from copy import deepcopy
import json

import numpy as np
import pytest
from core.status import DomainError
from physics.materials.iapws import properties_pt
from atomic_model_multiphysics.provenance import seal,unseal
from coupling.conservative import ConservativeMap
from coupling.pwr_moderator import moderator_fields,LocalModeratorPWRSession
from coupling.pwr_stationary import StationaryChannelPWRSession


def test_mass_weighted_boron_is_not_a_plain_average_of_ppm():
    op=ConservativeMap([0.,2.],[0.,1.,2.])
    r=moderator_fields(op,[1000.,500.],[0.,.001],[1.,1.])
    np.testing.assert_allclose(r['density_kg_m3'],[750.])
    np.testing.assert_allclose(r['boron_ppm'],[2/3])
    assert r['mass_by_neutron_cell_kg'].sum()==1500
    assert r['boron_by_neutron_cell_kg'].sum()==pytest.approx(.001)


def test_nonmatching_fields_preserve_water_and_boron_inventories():
    op=ConservativeMap([0.,.5,1.5,4.],[0.,1.,2.,4.])
    r=moderator_fields(op,[700.,750.,1600.],[.7,.6,1.2],[1.,1.,2.])
    assert r['mass_relative_defect']<1e-14 and r['boron_relative_defect']<1e-14
    assert min(r['density_kg_m3'])>0 and min(r['boron_ppm'])>=0


def test_zero_boron_is_supported_without_a_positive_floor():
    r=moderator_fields(ConservativeMap([0.,1.,2.],[0.,2.]),[1400.],[0.],[2.])
    np.testing.assert_array_equal(r['boron_ppm'],[0.,0.])
    assert r['boron_relative_defect']==0


@pytest.mark.parametrize('m,b,v',[([0.,1.],[0.,0.],[1.,1.]),([1.,1.],[-1.,0.],[1.,1.]),
    ([1.,1.],[.02,0.],[1.,1.]),([1.,1.],[0.,0.],[1.,2.]),([1.,np.nan],[0.,0.],[1.,1.]),
    ([True,1.],[0.,0.],[1.,1.]),([1.,1.],[0.+0j,0.],[1.,1.]),([1.,1.],[0.],[1.,1.])])
def test_invalid_moderator_fields_refused(m,b,v):
    with pytest.raises((ValueError,DomainError)):moderator_fields(ConservativeMap([0.,2.],[0.,1.,2.]),m,b,v)


@pytest.fixture(scope='module')
def reference():return LocalModeratorPWRSession()


def test_reference_preserves_baseline_and_new_fixed_reference_jacobian_is_full_rank(reference):
    s=deepcopy(reference);report=s.local_stationary_report;r=s.snapshot()
    assert report['baseline_rhs_max_difference']==0.
    assert report['constrained_rank']==report['constrained_unknowns']==46
    assert report['max_selected_tolerance_ratio']<=1 and report['reference_frozen_during_differentiation']
    assert max(abs(np.r_[r['local_moderator']['density_reactivity_correction'],r['local_moderator']['boron_reactivity_correction']]))==0
    assert r['local_density_boron_feedback'] and r['core']['local_moderator_feedback_connected']
    assert not r['local_moderator']['validated']


def test_registry_exposes_increment_coefficients_and_constructed_references(reference):
    from atomic_model_station.foundation import validate_registry
    registry=reference.parameter_registry();assert validate_registry(registry)
    rows={k:v for k,v in registry['parameters'].items() if k.startswith('local_moderator.')}
    assert len(rows)==4+2*reference.neutron_core.cells
    assert all(v['family']=='illustrative' and v['uncertainty'] is None for v in rows.values())
    assert rows['local_moderator.density_coefficient']['value']==reference.plant.density_coef
    assert rows['local_moderator.boron_coefficient']['value']==reference.plant.boron_coef


def test_local_boron_redistribution_changes_neutrons_with_unchanged_bulk_concentration(reference):
    s=deepcopy(reference);y=s.state.copy();initial=s._rhs(0.,y);rho=np.array(s.core_terms['local_reactivity'])
    fluid=s._fluid(y);delta=fluid[0,0]*1e-6;fluid[0,2]+=delta;fluid[-1,2]-=delta
    y=s._reconstruct_state(y);before_bulk=s.state[s.core_alias_index+2]/s.state[s.core_alias_index]
    assert y[s.core_alias_index+2]/y[s.core_alias_index]==pytest.approx(before_bulk,abs=1e-15)
    actual=s._rhs(0.,y);fields=s._moderator_fields(y)
    expected=s.plant.boron_coef*(fields['boron_ppm']-s.moderator_reference.boron_ppm)
    np.testing.assert_allclose(np.array(s.core_terms['local_reactivity'])-rho,expected,rtol=1e-10,atol=1e-14)
    n=s.neutron_core.cells;owned=y[s.core_index:s.core_index+n]
    np.testing.assert_allclose(actual[s.core_index:s.core_index+n]-initial[s.core_index:s.core_index+n],expected*owned/s.plant.kinetics.generation_time_s,rtol=1e-9,atol=1e-9)
    assert min(expected)<0<max(expected)


def test_warming_one_cell_produces_local_negative_density_feedback(reference):
    s=deepcopy(reference);y=s.state.copy();s._rhs(0.,y);rho=np.array(s.core_terms['local_reactivity'])
    v=s.channel.volumes['cell0'];c=v.boron_fraction;state=properties_pt(v.pressure_pa,v.temperature_k+.1)
    y[s.fluid_index]=state.rho_kg_m3*v.volume_m3;y[s.fluid_index+1]=y[s.fluid_index]*state.u_j_kg
    y[s.fluid_index+2]=c*y[s.fluid_index];y=s._reconstruct_state(y);s._rhs(0.,y)
    expected=s.plant.density_coef*(s._moderator_fields(y)['density_kg_m3']-s.moderator_reference.density_kg_m3)
    np.testing.assert_allclose(np.array(s.core_terms['local_reactivity'])-rho,expected,rtol=1e-9,atol=1e-13)
    assert min(expected)<0 and max(expected)<=0


def test_cold_leg_density_is_removed_from_the_neutron_feedback_change(reference):
    s=deepcopy(reference);y=s.state.copy();s._rhs(0.,y);rho=np.array(s.core_terms['local_reactivity'])
    cold=s.plant.network.volumes['cold_leg'];state=properties_pt(cold.pressure_pa,cold.temperature_k+.1)
    i=13+3*s.plant.network.volume_names().index('cold_leg');c=cold.boron_fraction
    y[i]=state.rho_kg_m3*cold.volume_m3;y[i+1]=y[i]*state.u_j_kg;y[i+2]=c*y[i]
    s._rhs(0.,y)
    np.testing.assert_allclose(s.core_terms['local_reactivity'],rho,rtol=0,atol=1e-13)
    assert max(abs(np.array(s.moderator_terms['density_reactivity_correction'])))>1e-6


def test_reference_does_not_follow_the_transient_or_change_after_rejected_commands(reference):
    s=deepcopy(reference);ref=s.moderator_reference;r=s.advance(.02,{'scram':True})
    assert s.moderator_reference==ref
    assert r['energy_relative']<1e-9 and r['mass_relative']<1e-10 and r['boron_relative']<1e-10
    assert r['core']['neutron_precursor_inventory']['relative_defect']<1e-10
    assert r['local_moderator']['mass_relative_defect']<1e-12 and r['local_moderator']['boron_relative_defect']<1e-12
    before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(.01,{'load_fraction':99})
    assert s.checkpoint()==before and s.moderator_reference==ref


def test_json_restart_reconstructs_reference_and_exact_scram_suffix(reference):
    s=deepcopy(reference);s.advance(.01);saved=json.loads(json.dumps(s.checkpoint()))
    t=LocalModeratorPWRSession.restore(saved)
    assert s.advance(.02,{'scram':True})==t.advance(.02,{'scram':True})
    assert s.checkpoint()==t.checkpoint()
    with pytest.raises(ValueError):StationaryChannelPWRSession.restore(saved)


@pytest.mark.parametrize('kind',['source','format','extra','history','initial_source','initial_history','stationary_config','snapshot_reference'])
def test_resealed_metadata_corruption_cannot_redefine_reference(reference,kind):
    p=unseal(reference.checkpoint())
    if kind=='source':p['source']['moderator_sha256']='0'*64
    elif kind=='format':p['format']='other'
    elif kind=='extra':p['extra']=True
    elif kind=='history':p['history']='bad'
    elif kind=='snapshot_reference':p['snapshot']['local_moderator']['reference']['density_kg_m3'][0]+=1.
    else:
        initial=unseal(p['initial_stationary'])
        if kind=='initial_source':initial['source']['initializer_sha256']='0'*64
        elif kind=='initial_history':initial['history']=[dict(duration_s=.01,patch=None)]
        else:initial['stationary']['extra']=1
        p['initial_stationary']=seal(initial)
    with pytest.raises(ValueError):LocalModeratorPWRSession.restore(seal(p))
