import numpy as np
import pytest

from core.status import DomainError
from scripts.qualify_boron_mesh import compare_fronts,assess_case,matched_cases,FIELDS,LIMITS


def row(n,ppm):
    values=[float(ppm)]*n if type(ppm) in (int,float) else list(ppm)
    return dict(core_config=dict(cells=n,length_m=4.),fluid_grid=dict(faces_m=np.linspace(0.,4.,n+1).tolist()),
        distributed_fluid=dict(boron_fraction=[x*1e-6 for x in values]),local_moderator=dict(boron_ppm=values))


def test_own_initial_references_remove_trim_offset_but_preserve_same_increment():
    result=compare_fronts(row(2,991.),row(8,994.),row(2,990.),row(8,993.))
    assert result['neutron.boron_increment_ppm']['l2_rms']==0.
    assert result['fluid.boron_increment_ppm']['l2_rms']<1e-12


def test_zero_mean_front_error_cannot_cancel_in_spatial_norm():
    result=compare_fronts(row(2,990.),row(4,[991.,989.,991.,989.]),row(2,990.),row(4,990.))
    assert result['neutron.boron_increment_ppm']['l2_rms']==1.
    assert result['neutron.boron_increment_ppm']['integral_difference']==0.
    assert result['fluid.boron_increment_ppm']['l2_rms']==pytest.approx(1.)


def test_initial_reference_must_belong_to_the_same_mesh():
    with pytest.raises(ValueError,match='own initial'):compare_fronts(row(2,991.),row(4,991.),row(4,990.),row(4,990.))


@pytest.mark.parametrize('bad',[True,float('nan'),float('inf')])
def test_invalid_reference_values_are_rejected(bad):
    initial=row(2,990.);initial['local_moderator']['boron_ppm'][0]=bad
    with pytest.raises((ValueError,DomainError)):compare_fronts(row(2,991.),row(4,991.),initial,row(4,990.))


def test_broadcasting_cannot_hide_a_missing_initial_concentration_cell():
    initial=row(2,990.);initial['local_moderator']['boron_ppm']=[990.]
    with pytest.raises((ValueError,DomainError)):compare_fronts(row(2,991.),row(4,991.),initial,row(4,990.))


def test_both_increment_profiles_have_explicit_absolute_ppm_limits():
    assert len(FIELDS)==24 and len(set(FIELDS))==24
    assert LIMITS['fluid.boron_increment_ppm']==LIMITS['neutron.boron_increment_ppm']==.25


def test_incomplete_or_unknown_chemical_matrix_cannot_pass():
    with pytest.raises(ValueError):assess_case({},'dilution')
    with pytest.raises(ValueError):assess_case({},'unknown')
    with pytest.raises(ValueError):matched_cases({'dilution':{}})
