from copy import deepcopy
import math

import pytest

from atomic_model_station_faults import FaultSession
from atomic_model_station_inventory import capture, summarize, archive_summary


def trace():
    result = []
    for t in (0., 1., 2.):
        stocks = dict(feed=dict(mass_kg=100.-t, rate_kg_s=-1., minimum_mass_kg=20.,
                               capacity_mass_kg=200., usable_mass_kg=80.-t, free_capacity_kg=100.+t,
                               expected_rate_kg_s=-1., continuity_residual_kg_s=0.),
                      drain=dict(mass_kg=40.+t, rate_kg_s=1., expected_rate_kg_s=1., continuity_residual_kg_s=0.))
        result.append(dict(format='station-inventory/v1', time_s=t, stocks=stocks,
                           total_mass_kg=140., total_rate_kg_s=0., maximum_local_continuity_residual_kg_s=0.,
                           energy_relative=0., mass_relative=0., boron_relative=0., trip=False))
    return result


def test_conservative_transfer_and_declining_reserve_are_distinct_from_leakage():
    report = summarize(trace(), requested_horizon_s=100.)
    assert report['total_mass_change_kg'] == 0.
    feed = next(r for r in report['stocks'] if r['name']=='feed')
    assert feed['change_kg']==-2. and feed['quadrature_residual_kg']==0.
    forecast = feed['conditional_linear_forecasts']['observed_mean']
    assert forecast['seconds_to_model_boundary']==78.
    assert forecast['reaches_boundary_within_requested_horizon']
    assert not report['stationary_accepted'] and not report['r1_accepted']


def test_zero_final_rate_does_not_erase_mean_reserve_consumption():
    rows = trace(); rows[-1]['stocks']['feed'].update(rate_kg_s=0., expected_rate_kg_s=0.)
    rows[-1]['stocks']['drain'].update(rate_kg_s=0., expected_rate_kg_s=0.)
    report = summarize(rows)
    f = next(r for r in report['stocks'] if r['name']=='feed')['conditional_linear_forecasts']
    assert f['observed_mean']['seconds_to_model_boundary']==78.
    assert f['endpoint']['seconds_to_model_boundary'] is None


def test_sparse_quadrature_is_marked_and_not_reported_as_solver_conservation():
    rows = trace(); rows[1]['stocks']['feed'].update(rate_kg_s=-3., expected_rate_kg_s=-3.)
    rows[1]['stocks']['drain'].update(rate_kg_s=3., expected_rate_kg_s=3.)
    report = summarize(rows, maximum_gap_s=.5)
    assert not report['sampling_adequate']
    assert report['maximum_total_rate_kg_s']==0.
    assert next(r for r in report['stocks'] if r['name']=='feed')['quadrature_residual_kg']==2.
    missing=trace()
    for row in missing:
        del row['stocks']['feed']['expected_rate_kg_s']
        del row['stocks']['feed']['continuity_residual_kg_s']
    with pytest.raises(ValueError): summarize(missing)
    shifted=trace(); shifted[1]['stocks']['feed'].update(capacity_mass_kg=201.,free_capacity_kg=102.)
    with pytest.raises(ValueError): summarize(shifted)


@pytest.mark.parametrize('change', [dict(time_s=0.), dict(time_s=float('nan')), dict(total_mass_kg=141.),
    dict(total_rate_kg_s=1.), dict(maximum_local_continuity_residual_kg_s=1.), dict(energy_relative=-1.),
    dict(trip=1), dict(stocks={})])
def test_corrupt_inventory_evidence_is_rejected(change):
    rows = trace(); rows[1].update(change)
    with pytest.raises(ValueError): summarize(rows)


@pytest.mark.parametrize('change', [dict(mass_kg=True), dict(rate_kg_s=float('inf')), dict(usable_mass_kg=10.),
    dict(capacity_mass_kg=50.), dict(continuity_residual_kg_s=1.)])
def test_inconsistent_local_mass_or_reserve_evidence_is_rejected(change):
    rows = trace(); rows[1]['stocks']['feed'].update(change)
    with pytest.raises(ValueError): summarize(rows)


def test_live_capture_is_read_only_and_audits_all_local_mass_equations():
    owner = FaultSession(tick_s=.01)
    owner.step({'feed_inventory_control':True, 'feed_inventory_fraction':.25})
    before = owner.checkpoint(); result = capture(owner)
    assert owner.checkpoint()==before
    assert len(result['stocks'])==11
    assert all('continuity_residual_kg_s' in s for s in result['stocks'].values())
    assert result['maximum_local_continuity_residual_kg_s']<1e-10
    assert abs(result['total_rate_kg_s'])<1e-10
    assert result['stocks']['drain']['rate_kg_s']>0
    assert result['stocks']['makeup']['rate_kg_s']==0
    assert result['stocks']['drain']['capacity_mass_kg'] < owner.control.plant.feed_tanks['drain']['total_volume_m3']*owner.control.plant.feed_rho
    owner.step({'steam_bypass':.1})
    assert owner.control.plant.snapshot()['header_bypass']['flow_kg_s']>0
    before=owner.checkpoint(); opened=capture(owner)
    assert opened['maximum_local_continuity_residual_kg_s']<1e-10
    assert abs(opened['total_rate_kg_s'])<1e-10 and owner.checkpoint()==before


def test_actual_makeup_and_drain_pulses_have_opposite_reserve_directions(tmp_path, monkeypatch):
    owner = FaultSession(tick_s=.05)
    rows = [capture(owner)]
    for fraction in (.25, .25, .75, .75):
        owner.step({'feed_inventory_control':True, 'feed_inventory_fraction':fraction})
        rows.append(capture(owner))
    report = summarize(rows, maximum_gap_s=.05)
    by_name = {r['name']:r for r in report['stocks']}
    assert by_name['drain']['change_kg']>0
    assert by_name['makeup']['change_kg']<0
    assert abs(report['total_mass_change_kg'])<1e-8
    assert report['maximum_local_continuity_residual_kg_s']<1e-10
    from atomic_model_station_inventory.__main__ import main
    from atomic_model_multiphysics.provenance import unseal
    import json
    output=tmp_path/'cli'
    monkeypatch.setattr('sys.argv', ['inventory','--output',str(output),'--duration-s','.1','--tick-s','.05'])
    assert main()==0
    saved=unseal(json.loads((output/'report.json').read_text()))
    assert saved['completed_time_s']==pytest.approx(.1) and saved['source_unchanged']


def test_archive_trends_never_invent_missing_instantaneous_rates():
    owner = FaultSession(tick_s=.01); p=owner.control.plant
    rows=[]
    for t in (0., 1., 2.):
        y=p.state.tolist(); y[p.feed_index]-=t; y[p.feed_index+4]+=t
        rows.append(dict(time_s=t, physical_state=y))
    before=deepcopy(rows)
    report=archive_summary(rows, owner, window_s=2.)
    assert rows==before
    assert abs(report['total_mass_change_kg'])<1e-8
    assert not report['instantaneous_rates_available']
    assert next(r for r in report['stocks'] if r['name']=='drain')['mean_observed_rate_kg_s']==1.
    assert not report['reserve_exchange']['opposing_reserve_transfers_observed']
    assert report['reserve_margins']['drain']['conditional_seconds_to_mass_or_pressure_bound']>0
    opposed=deepcopy(rows)
    for row in opposed:
        row['physical_state'][p.feed_index+2]-=row['time_s']*.5
        row['physical_state'][p.feed_index]+=row['time_s']*.5
    both=archive_summary(opposed, owner, window_s=2.)
    assert both['reserve_exchange']['opposing_reserve_transfers_observed']
    assert both['reserve_exchange']['minimum_opposing_transfer_kg']==1.
    assert both['reserve_exchange']['net_feed_contribution_kg']==-1.
    with pytest.raises(ValueError): archive_summary(rows, owner, window_s=3.)
    rows[1]['physical_state'][p.feed_index]=math.nan
    with pytest.raises(ValueError): archive_summary(rows, owner, window_s=2.)


@pytest.mark.parametrize('kw', [dict(requested_horizon_s=True), dict(requested_horizon_s=0.), dict(maximum_gap_s=0.)])
def test_invalid_diagnostic_horizon_or_gap(kw):
    with pytest.raises(ValueError): summarize(trace(), **kw)
