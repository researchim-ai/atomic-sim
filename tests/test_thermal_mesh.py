from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest

from atomic_model_station import CoreConfig
from core.status import DomainError
from coupling.conservative import ConservativeMap
from coupling.thermal_grid import aligned_fuel_grid
from scripts.qualify_thermal_mesh import (FIELDS, assess, coefficient_defect,
    compare_snapshots, grid, source, thermal_audit)


def fixture(cells):
    faces=grid(cells).faces_m;width=np.diff(faces)
    incoming=150*width;outgoing=40*width
    pairs=ConservativeMap(faces,(0.,1.,2.,4.)).power_matrix*outgoing[None,:]
    coefficients=dict(fuel_capacity_j_k=(2*width).tolist(),cladding_capacity_j_k=(3*width).tolist(),
        fuel_cladding_ua_w_k=(.5*width).tolist(),cladding_fluid_ua_w_k=(2*width).tolist())
    row=dict(core_config=asdict(CoreConfig(kind='diffusion',cells=4)),
        fuel_grid=dict(faces_m=list(faces)),fluid_grid=dict(faces_m=[0.,1.,2.,4.],internal_loss_coefficient=0.),
        distributed_fuel=dict(temperature_k=[900.]*cells,decay_stored_j=[(j*width).tolist() for j in (10,20,30)],
            fission_by_fuel_cell_w=(160*width).tolist(),thermal_source_by_cell_w=(160*width).tolist(),
            fuel_to_clad_by_cell_w=incoming.tolist(),fuel_temperature_rate_k_s=[5.]*cells),
        distributed_cladding=dict(temperature_k=[600.]*cells,temperature_rate_k_s=[110./3]*cells,
            fuel_to_clad_by_cell_w=incoming.tolist(),clad_to_coolant_by_cell_w=outgoing.tolist(),
            stored_energy_j=7200.,conductive_entropy_generation_w_k=(incoming*(1/600-1/900)+outgoing*(1/580-1/600)).tolist()),
        distributed_fluid=dict(volume_m3=[4.5,4.5,9.],pair_heat_w=pairs.tolist(),coolant_heat_by_cell_w=pairs.sum(axis=1).tolist(),
            contact_entropy_w_k=float(sum(outgoing)*(1/580-1/600))))
    return row,coefficients


def scales():
    return {k:1. for k in FIELDS}


def test_alignment_preserves_every_physical_interface_without_snapping():
    result=aligned_fuel_grid((0.,.7,2.,4.),(0.,1.,2.2,4.))
    assert result.faces_m==(0.,.7,1.,2.,2.2,4.)
    refined=aligned_fuel_grid((0.,.7,2.,4.),(0.,1.,2.2,4.),subdivisions=2)
    assert set(result.faces_m)<=set(refined.faces_m) and len(refined.faces_m)==11


@pytest.mark.parametrize('cells',[4,8,16])
def test_qualified_meshes_are_nested_and_preserve_source_and_fluid_faces(cells):
    result=grid(cells)
    assert len(result.faces_m)==cells+1
    assert set(grid(4).faces_m)<=set(result.faces_m)


@pytest.mark.parametrize('factor',[True,0,-1,1.5,17])
def test_bad_subdivision_is_refused(factor):
    with pytest.raises(ValueError):aligned_fuel_grid((0.,4.),(0.,4.),subdivisions=factor)


def test_state_budget_is_not_silently_expanded():
    with pytest.raises(ValueError,match='state budget'):
        aligned_fuel_grid((0.,1.,2.,3.,4.),(0.,4.),subdivisions=5)


def test_nonmatching_endpoints_and_near_duplicate_faces_are_not_merged():
    with pytest.raises(DomainError):aligned_fuel_grid((0.,4.),(0.,5.))
    with pytest.raises(DomainError):aligned_fuel_grid((0.,1.,4.),(0.,np.nextafter(1.,2.),4.))


def test_uniform_child_profiles_preserve_temperatures_heat_and_each_decay_inventory():
    left,_=fixture(4);right,_=fixture(16)
    result=compare_snapshots(left,right,scales())
    assert len(result)==9 and all(v['normalized_l2']==0. for v in result.values())


def test_third_decay_reservoir_cannot_be_hidden_by_other_profiles():
    left,_=fixture(4);right,_=fixture(8)
    right['distributed_fuel']['decay_stored_j'][2]=[v+1. for v in right['distributed_fuel']['decay_stored_j'][2]]
    result=compare_snapshots(left,right,scales())
    assert result['decay2']['normalized_l2']>0
    assert all(result[k]['normalized_l2']==0. for k in FIELDS if k!='decay2')


def test_per_cell_energy_contact_and_entropy_balances_match_hand_solution():
    row,coefficients=fixture(8)
    assert all(thermal_audit(row,coefficients).values())
    json.dumps(thermal_audit(row,coefficients),allow_nan=False)


def test_nonpositive_temperature_is_rejected_before_entropy_division():
    row,coefficients=fixture(4);row['distributed_cladding']['temperature_k'][0]=0.
    with pytest.raises(DomainError):thermal_audit(row,coefficients)


@pytest.mark.parametrize('defect',['fuel_rate','clad_rate','pair_heat','stored','entropy'])
def test_global_or_cached_checks_cannot_mask_a_local_heat_defect(defect):
    row,coefficients=fixture(8)
    if defect=='fuel_rate':row['distributed_fuel']['fuel_temperature_rate_k_s'][0]+=1.
    elif defect=='clad_rate':row['distributed_cladding']['temperature_rate_k_s'][0]+=1.
    elif defect=='pair_heat':row['distributed_fluid']['pair_heat_w'][0][0]+=1.
    elif defect=='stored':row['distributed_cladding']['stored_energy_j']+=1.
    else:row['distributed_cladding']['conductive_entropy_generation_w_k'][0]=-1.
    assert not all(thermal_audit(row,coefficients).values())


def test_equal_totals_cannot_mask_changed_local_constitutive_density():
    trajectories={}
    for n in (4,8,16):
        row,coefficients=fixture(n)
        for preset in ('coarse','fine'):
            trajectories[f'{n}_{preset}']=dict(initial=row,measures=deepcopy(coefficients))
    assert coefficient_defect(trajectories)==0.
    values=trajectories['8_fine']['measures']['fuel_capacity_j_k'];values[0]+=.1;values[1]-=.1
    assert coefficient_defect(trajectories)>.01


def test_fixed_spatial_model_and_complete_fields_are_required():
    left,_=fixture(4);right,_=fixture(8)
    right['core_config']['cells']=8
    with pytest.raises(ValueError,match='remain fixed'):compare_snapshots(left,right,scales())
    right,_=fixture(8);right['distributed_fuel']['decay_stored_j'].pop()
    with pytest.raises(ValueError,match='Three complete'):compare_snapshots(left,right,scales())


def test_incomplete_matrix_cannot_produce_an_acceptance():
    with pytest.raises(ValueError,match='two temporal'):assess({'4_fine':{}})


def test_identity_binds_grid_builder_and_numeric_helpers():
    assert set(source()['helpers_sha256'])=={'coupling.mesh_comparison','coupling.thermal_grid',
        'scripts.qualify_moderator_transients','scripts.qualify_pwr_distributed_fuel'}
