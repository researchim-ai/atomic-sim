from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
from atomic_model_station import CoreConfig
from coupling.pwr_channel import ChannelPWRSession,FluidGrid
from scripts.qualify_fluid_geometry import FIELDS,assess,audit_geometry,coefficients,compare_snapshots


def snapshot(n):
    w=4./n
    return dict(core_config=asdict(CoreConfig(kind='diffusion',cells=4)),fuel_grid=dict(faces_m=[0.,2.,4.]),
        fluid_grid=dict(faces_m=np.linspace(0.,4.,n+1).tolist(),internal_loss_coefficient=1.,momentum_geometry='complete_dual'),
        distributed_fluid=dict(inventory_by_cell=[[2*w,30*w,.001*w] for _ in range(n)],
            temperature_k=[550.]*n,pressure_pa=[15.5e6]*n,density_kg_m3=[700.]*n))


def test_uniform_physical_fields_are_compared_independently_of_inventory_cell_sizes():
    result=compare_snapshots(snapshot(2),snapshot(8),{k:1. for k in FIELDS})
    assert len(result)==6 and all(v['normalized_l2']==0. for v in result.values())


def test_equal_total_boron_does_not_hide_a_spatial_transport_error():
    a,b=snapshot(2),snapshot(8)
    b['distributed_fluid']['inventory_by_cell'][0][2]+=.0001
    b['distributed_fluid']['inventory_by_cell'][1][2]-=.0001
    result=compare_snapshots(a,b,{k:1. for k in FIELDS})
    assert result['boron']['normalized_l2']>0
    assert abs(result['boron']['integral_difference'])<1e-18


@pytest.mark.parametrize('change',['core','loss','mode'])
def test_model_changes_cannot_be_presented_as_fluid_refinement(change):
    a,b=snapshot(2),snapshot(8)
    if change=='core':b['core_config']['cells']=8
    elif change=='loss':b['fluid_grid']['internal_loss_coefficient']=2.
    else:b['fluid_grid']['momentum_geometry']='legacy'
    with pytest.raises(ValueError):compare_snapshots(a,b,{k:1. for k in FIELDS})


def test_incomplete_temporal_spatial_matrix_is_rejected():
    with pytest.raises(ValueError,match='two temporal'):assess({'2_fine':{}})


@pytest.fixture(scope='module')
def native():
    s=ChannelPWRSession(fluid_grid=FluidGrid((0.,1.,4.),1.,'complete_dual'))
    return s.snapshot(),coefficients(s)


def test_actual_native_coefficients_satisfy_geometry_momentum_and_kinetic_audits(native):
    row,c=native
    assert audit_geometry(row,c)['passed']


@pytest.mark.parametrize('field',['inertance_m_inv','kinetic_coefficients_m2_kg','loss_per_m4'])
def test_copied_diagnostics_cannot_hide_changed_actual_hydraulic_coefficients(native,field):
    row,original=native;c=deepcopy(original);c[field][0]*=1.1
    assert not audit_geometry(row,c)['passed']
