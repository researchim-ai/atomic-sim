"""Phase root admission uses event sides, not interpolated phase labels."""
import csv
import pytest
from scripts import verify_phase_event as bench


def write(path,rows):
    with path.open('w') as stream:
        out=csv.DictWriter(stream,fieldnames=['time','phase']);out.writeheader();out.writerows(rows)


def test_external_exact_left_right_event_keeps_time_and_both_phases(tmp_path):
    path=tmp_path/'trace.csv'
    write(path,[dict(time=0.,phase=2.),dict(time=.005,phase=2.),dict(time=.005,phase=1.),dict(time=.01,phase=1.)])
    assert bench.external_events(path)==dict(time_s=.005,before_phase=2.,after_phase=1.)


@pytest.mark.parametrize('damage',['prefix','interpolated','nan','ordering','phase','no_event','multiple'])
def test_missing_or_corrupt_event_evidence_cannot_pass(tmp_path,damage):
    rows=[dict(time=0.,phase=2.),dict(time=.005,phase=2.),dict(time=.005,phase=1.),dict(time=.01,phase=1.)]
    if damage=='prefix':rows.pop()
    elif damage=='interpolated':rows[2]['time']=.0051
    elif damage=='nan':rows[1]['time']=float('nan')
    elif damage=='ordering':rows.reverse()
    elif damage=='phase':rows[1]['phase']=3.
    elif damage=='no_event':
        for row in rows:row['phase']=2.
    else:rows.insert(3,dict(time=.005,phase=2.))
    path=tmp_path/'trace.csv';write(path,rows)
    with pytest.raises(ValueError):bench.external_events(path)


@pytest.mark.parametrize('case',bench.CASES)
def test_independent_energy_quadrature_agrees_with_both_event_integrators(case):
    result=bench.reference_event(case)
    assert result['passed'],result
    assert {r['method'] for r in result['ode']}=={'DOP853','Radau'}
