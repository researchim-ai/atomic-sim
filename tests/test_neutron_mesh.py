from dataclasses import asdict

import pytest

from atomic_model_station import CoreConfig
from scripts.qualify_neutron_mesh import compare_snapshots, assess, source


def test_scientific_identity_binds_protocol_and_budget_helpers():
    assert set(source()['helpers_sha256'])=={'scripts.qualify_moderator_transients','scripts.qualify_pwr_distributed_fuel'}


def snapshot(cells):
    return dict(core_config=asdict(CoreConfig(kind='diffusion',cells=cells)),
                core=dict(fission_by_cell_w=[4./cells]*cells,
                          local_precursors=[[i+1.]*cells for i in range(6)]))


def scales():
    return dict(fission=1.,**{'precursor'+str(i):i+1. for i in range(6)})


def test_uniform_power_and_all_six_precursors_have_exact_mesh_independent_norms():
    result=compare_snapshots(snapshot(2),snapshot(8),scales())
    assert len(result)==7
    assert all(r['normalized_l2']==0. for r in result.values())


def test_equal_total_power_does_not_hide_a_wrong_spatial_shape():
    left=snapshot(2);right=snapshot(4)
    right['core']['fission_by_cell_w']=[0.,2.,0.,2.]
    result=compare_snapshots(left,right,scales())['fission']
    assert result['integral_difference']==0.
    assert result['normalized_l2']==1.


def test_each_precursor_family_is_checked_separately():
    left=snapshot(2);right=snapshot(4)
    right['core']['local_precursors'][5]=[12.]*4
    result=compare_snapshots(left,right,scales())
    assert result['precursor5']['normalized_l2']==1.
    assert all(result[k]['normalized_l2']==0. for k in result if k!='precursor5')


@pytest.mark.parametrize('field,value', [('length_m',5.),('diffusion_m',.02),('axial_bias_reactivity',1e-5),('kind','point')])
def test_comparison_cannot_disguise_a_physical_model_change_as_refinement(field,value):
    left=snapshot(2);right=snapshot(4);right['core_config'][field]=value
    with pytest.raises(ValueError):compare_snapshots(left,right,scales())


def test_missing_precursor_family_is_rejected():
    right=snapshot(4);right['core']['local_precursors'].pop()
    with pytest.raises(ValueError,match='six-family'):compare_snapshots(snapshot(2),right,scales())


def test_missing_normalization_cannot_skip_a_family():
    values=scales();values.pop('precursor2')
    with pytest.raises(ValueError,match='six precursor'):compare_snapshots(snapshot(2),snapshot(4),values)


@pytest.mark.parametrize('trajectories',[{}, {'2_fine':{}}, {f'{n}_fine':{} for n in (2,4,8)}])
def test_incomplete_spatial_temporal_matrix_cannot_be_accepted(trajectories):
    with pytest.raises(ValueError,match='two temporal'):assess(trajectories)
