import math

import numpy as np
import pytest

from core.status import DomainError
from coupling.mesh_comparison import profile_error


def compare(a, b, x=(0., 1., 2.), y=(0., 2.), **kwargs):
    return profile_error(x, a, y, b, kind=kwargs.pop('kind', 'cell_average'),
                         scale=kwargs.pop('scale', 1.), **kwargs)


def test_opposing_subcell_errors_cannot_cancel_under_coarse_averaging():
    result = compare([1., -1.], [0.])
    assert result['integral_difference'] == 0.
    assert result['l1_mean'] == result['l2_rms'] == result['linf'] == 1.


def test_exact_nonmatching_overlap_norms_against_hand_integrals():
    # Differences are 1 over length .5 and 2 over length 1.5.
    result = compare([3., 0.], [2., 2.], x=(0., .5, 2.), y=(0., 1.5, 2.), scale=2.)
    assert result['l1_mean'] == pytest.approx(1.75)
    assert result['l2_rms'] == pytest.approx(math.sqrt(3.25))
    assert result['linf'] == pytest.approx(2.)
    assert result['integral_difference'] == pytest.approx(-2.5)
    assert result['normalized_l1'] == pytest.approx(.875)


def test_integrated_powers_are_compared_as_densities_not_cell_totals():
    result = compare([10., 30.], [40.], x=(0., 1., 4.), y=(0., 4.), kind='cell_integral', scale=10.)
    assert result['l1_mean'] == result['linf'] == 0.


def test_identical_step_field_on_refined_mesh_has_zero_error():
    result = compare([2., 5.], [2., 2., 5.], y=(0., .4, 1., 2.))
    assert result['normalized_l2'] == result['normalized_linf'] == 0.


def test_reversing_comparison_preserves_norms_and_reverses_integral_sign():
    a=profile_error((0., .4, 2.), [2., -3.], (0., 1., 2.), [4., 1.], kind='cell_average', scale=3.)
    b=profile_error((0., 1., 2.), [4., 1.], (0., .4, 2.), [2., -3.], kind='cell_average', scale=3.)
    for key in ('l1_mean','l2_rms','linf','normalized_l1','normalized_l2','normalized_linf'):
        assert a[key] == pytest.approx(b[key])
    assert a['integral_difference'] == pytest.approx(-b['integral_difference'])


def test_very_large_representable_error_does_not_overflow_during_squaring():
    result=compare([1e200, -1e200], [0.], scale=1e200)
    assert result['normalized_l2'] == 1.


def test_normalization_does_not_erase_representable_absolute_errors():
    result=compare([1e-100,-1e-100],[0.],scale=1e300)
    assert result['l1_mean']==result['l2_rms']==result['linf']==1e-100


def test_equal_large_background_cannot_erase_a_local_small_error():
    result=compare([1e300,1e-100],[1e300,0.],y=(0.,1.,2.))
    assert result['linf']==1e-100
    assert result['l1_mean']==5e-101
    assert result['l2_rms']==pytest.approx(1e-100/math.sqrt(2),rel=1e-14,abs=0.)


def test_overflow_between_nonoverlapping_cells_is_irrelevant():
    result=compare([1e308,-1e308],[1e308,-1e308],y=(0.,1.,2.))
    assert result['l1_mean']==result['l2_rms']==result['linf']==0.


@pytest.mark.parametrize('change',[
    dict(scale=0.),dict(scale=True),dict(scale=float('inf')),dict(kind='auto'),
    dict(values=[True, 0.]),dict(values=[1.+0j,0.]),dict(values=[np.nan,0.]),
    dict(values=[1.]),dict(reference_faces=(0.,3.)),dict(faces=(0.,1.,1.)),
    dict(values=[1e308,-1e308],reference_values=[-1e308],scale=1.),
])
def test_invalid_or_unrepresentable_comparison_is_rejected(change):
    options=dict(faces=(0.,1.,2.),values=[1.,2.],reference_faces=(0.,2.),reference_values=[1.],kind='cell_average',scale=1.)
    options.update(change)
    with pytest.raises((DomainError,ValueError)):profile_error(**options)
