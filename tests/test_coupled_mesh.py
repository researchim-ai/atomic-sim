from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
from coupling.refinement import coupled_pwr_mesh
from scripts.qualify_coupled_mesh import FIELDS,assess,compare_snapshots,face_coefficient_defect,source
from scripts.qualify_fluid_geometry import coefficients
from coupling.pwr_channel import ChannelPWRSession


def row(n):
    recipe=coupled_pwr_mesh(n);fwidth=np.diff(recipe['fluid_grid'].faces_m);twidth=np.diff(recipe['fuel_grid'].faces_m)
    return dict(core_config=asdict(recipe['core']),fluid_grid=asdict(recipe['fluid_grid']),fuel_grid=asdict(recipe['fuel_grid']),
        momentum_geometry=dict(dual=dict(area_m2=4.5),channel_reference_density_kg_m3=700.,boundaries={}),
        distributed_fluid=dict(inventory_by_cell=np.outer(fwidth,[140.,140e6,.14]).tolist(),temperature_k=[550.]*n,pressure_pa=[15.5e6]*n,density_kg_m3=[700.]*n),
        distributed_fuel=dict(temperature_k=[900.]*(2*n),decay_stored_j=np.outer([10.,20.,30.],twidth).tolist(),
            fission_by_fuel_cell_w=(1e5*twidth).tolist(),thermal_source_by_cell_w=(.9e5*twidth).tolist(),fuel_to_clad_by_cell_w=(.9e5*twidth).tolist()),
        distributed_cladding=dict(temperature_k=[600.]*(2*n),clad_to_coolant_by_cell_w=(.9e5*twidth).tolist()),
        core=dict(fission_by_cell_w=(1e5*fwidth).tolist(),local_precursors=[[float(i+1)]*n for i in range(6)]))


@pytest.mark.parametrize('cells',[2,4,8])
def test_recipe_refines_all_three_meshes_and_keeps_physical_interfaces(cells):
    recipe=coupled_pwr_mesh(cells)
    assert recipe['core'].cells==cells and len(recipe['fluid_grid'].faces_m)==cells+1
    assert len(recipe['fuel_grid'].faces_m)==2*cells+1
    assert set(recipe['fluid_grid'].faces_m)<=set(recipe['fuel_grid'].faces_m)
    assert recipe['fluid_grid'].momentum_geometry=='complete_dual'
    assert recipe['fluid_grid'].internal_loss_coefficient==1.


@pytest.mark.parametrize('cells',[True,1,3,16,2.])
def test_invalid_nested_resolution_rejected(cells):
    with pytest.raises(ValueError):coupled_pwr_mesh(cells)


def test_declared_length_and_loss_are_preserved_in_configurations():
    recipe=coupled_pwr_mesh(4,length_m=5.,loss_coefficient=2.)
    assert recipe['core'].length_m==recipe['fluid_grid'].faces_m[-1]==recipe['fuel_grid'].faces_m[-1]==5.
    assert recipe['fluid_grid'].internal_loss_coefficient==2.


@pytest.mark.parametrize('options',[dict(length_m=True),dict(length_m=float('inf')),dict(loss_coefficient=-1.),dict(loss_coefficient=True)])
def test_invalid_physical_parameters_are_not_silently_coerced(options):
    with pytest.raises(ValueError):coupled_pwr_mesh(4,**options)


def test_all22_constant_profiles_are_mesh_independent_with_integral_semantics():
    result=compare_snapshots(row(2),row(8),{k:1. for k in FIELDS})
    assert len(result)==22 and all(r['normalized_l2']==0. for r in result.values())


@pytest.mark.parametrize('changed',['diffusion','loss','area','reference_density'])
def test_physical_changes_cannot_be_presented_as_joint_refinement(changed):
    a,b=row(2),row(8)
    if changed=='diffusion':b['core_config']['diffusion_m']=.02
    elif changed=='loss':b['fluid_grid']['internal_loss_coefficient']=2.
    elif changed=='area':b['momentum_geometry']['dual']['area_m2']=5.
    else:b['momentum_geometry']['channel_reference_density_kg_m3']=800.
    with pytest.raises(ValueError,match='resolution'):compare_snapshots(a,b,{k:1. for k in FIELDS})


def test_individual_precursor_or_decay_family_cannot_hide_behind_total_power():
    a,b=row(2),row(8)
    b['core']['local_precursors'][5]=[12.]*8
    b['distributed_fuel']['decay_stored_j'][2]=[v*2 for v in b['distributed_fuel']['decay_stored_j'][2]]
    result=compare_snapshots(a,b,{k:1. for k in FIELDS})
    assert result['neutron.precursor5']['normalized_l2']>0 and result['thermal.decay2']['normalized_l2']>0
    assert result['neutron.fission']['normalized_l2']==0.


def test_incomplete_matrix_and_missing_field_scales_are_rejected():
    with pytest.raises(ValueError,match='two temporal'):assess({'2_fine':{}})
    with pytest.raises(ValueError,match='All22'):compare_snapshots(row(2),row(8),{})


def test_local_hydraulic_partition_check_detects_equal_total_redistribution():
    s=ChannelPWRSession(**coupled_pwr_mesh(4));snapshot=s.snapshot();actual=coefficients(s)
    assert face_coefficient_defect(snapshot,actual)<1e-12
    changed=deepcopy(actual);changed['loss_per_m4'][1]+=.001;changed['loss_per_m4'][2]-=.001
    assert sum(changed['loss_per_m4'])==pytest.approx(sum(actual['loss_per_m4']))
    assert face_coefficient_defect(snapshot,changed)>.01


def test_scientific_identity_includes_recipe_and_both_physics_audit_helpers():
    keys=set(source()['helpers_sha256'])
    assert {'coupling.refinement','scripts.qualify_fluid_geometry','scripts.qualify_thermal_mesh'}<=keys
