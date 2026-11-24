import numpy as np
import pytest
from core.status import DomainError
from scripts.qualify_moderator_transients import protocol,separation_metrics,assess


def test_physical_law_effect_is_separated_from_both_numerical_errors():
    r=separation_metrics([100.,95.001],[100.,95.],[100.,90.001],[100.,90.],100.)
    assert r['resolved']
    assert r['power_effect_relative']==pytest.approx(.05)
    assert r['numerical_preset_change_relative']==pytest.approx(1e-5)
    assert r['separation_ratio']==pytest.approx(5000.)


def test_large_apparent_effect_is_rejected_when_numerically_unresolved():
    r=separation_metrics([100.,110.],[100.,95.],[100.,90.],[100.,90.],100.)
    assert not r['resolved'] and r['separation_ratio']<1


def test_absent_effect_cannot_pass_with_zero_numerical_difference():
    r=separation_metrics([100,90],[100,90],[100,90],[100,90],100.)
    assert not r['resolved'] and r['separation_ratio'] is None


def test_zero_preset_difference_has_no_nonfinite_ratio_in_json():
    import json
    r=separation_metrics([100,95],[100,95],[100,90],[100,90],100.)
    assert r['resolved'] and r['separation_ratio'] is None
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('traces,reference',[
    (([100],[100],[100],[100]),0.),
    (([100],[100],[100],[100]),True),
    (([100],[100],[100],[100]),float('inf')),
    (([100,90],[100],[100],[100]),100.),
    (([100],[-1],[100],[100]),100.),
    (([np.nan],[100],[100],[100]),100.),
    (([100+0j],[100],[100],[100]),100.),
    (([True],[100],[100],[100]),100.),
    (([],[],[],[]),100.),
])
def test_invalid_comparison_data_cannot_produce_an_acceptance(traces,reference):
    with pytest.raises((ValueError,DomainError)):separation_metrics(*traces,reference)


@pytest.mark.parametrize('case,total,target',[('insertion',2.,.27),('return',3.,.26)])
def test_public_rod_protocol_spans_motion_and_return_boundaries(case,total,target):
    steps=protocol(case,.26)
    assert sum(s['duration_s'] for s in steps)==total
    assert sum(s['duration_s'] for s in steps[:2])==.5
    assert steps[0]['patch']['rod_target']==pytest.approx(.27)
    assert next(s['patch']['rod_target'] for s in reversed(steps) if s['patch'])==pytest.approx(target)


@pytest.mark.parametrize('case,position',[('unknown',.2),('return',True),('return',1.),('return',float('nan'))])
def test_invalid_rod_protocol_refused(case,position):
    with pytest.raises(ValueError):protocol(case,position)


def test_incomplete_law_matrix_is_not_assessed_as_a_comparison():
    with pytest.raises(ValueError,match='four'):assess({'local_fine':{}},[])
