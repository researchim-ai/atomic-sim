"""Local water continuity and finite-reserve diagnostics for the hot station.

All rates come from a private copy's current RHS. Forecasts are conditional
linear extrapolations, not operating limits or acceptance of a stationary hold.
"""
from copy import deepcopy
import math

from atomic_model_station_faults import FaultSession
from atomic_model_station_readiness import schema


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _mass_indices(owner):
    p = owner.control.plant
    indices = {name: 13 + 3*i for i, name in enumerate(p.plant.network.volume_names())}
    indices.update(hotwell=p.base_size-3, feed=p.feed_index,
                   makeup=p.feed_index+2, drain=p.feed_index+4, header=p.steam_index)
    if len(set(indices.values())) != len(indices):
        raise ValueError('Inventory coordinates overlap')
    return indices


def _bounds(p, name):
    tank = p.feed_tanks[name]
    gas_low = tank['initial_gas_m3']*(tank['initial_pressure_pa']/1e4)**(1/1.4)
    gas_high = tank['initial_gas_m3']*(tank['initial_pressure_pa']/2e6)**(1/1.4)
    return (float(max(p.feed_config.minimum_mass_kg, (tank['total_volume_m3']-gas_low)*p.feed_rho)),
            float((tank['total_volume_m3']-gas_high)*p.feed_rho))


def capture(owner):
    """Read actual finite stocks and their local continuity equations, without mutation."""
    if type(owner) is not FaultSession:
        raise ValueError('FaultSession inventory owner required')
    trial = deepcopy(owner)
    p = trial.control.plant
    rate = p._rhs(p.time, p.state)
    terms = p.feed_terms
    q = p.plant.network.junctions
    feed = float(terms['feed_flow_kg_s'])
    condensate = float(terms['condensate_flow_kg_s'])
    makeup = float(terms['makeup_flow_kg_s'])
    drain = float(terms['drain_flow_kg_s'])
    steam = float(q['steam'].mass_flow_kg_s)
    bypass = float(q['dump'].mass_flow_kg_s)
    header_bypass = float(p.bypass_terms['flow_kg_s'])
    turbine = float(p.state[p.steam_index+2])
    expected = {name: 0. for name in p.plant.network.volume_names()}
    for edge in q.values():
        if edge.upstream in expected:
            expected[edge.upstream] -= float(edge.mass_flow_kg_s)
        if edge.downstream in expected:
            expected[edge.downstream] += float(edge.mass_flow_kg_s)
    expected.update(header=steam-turbine-header_bypass, hotwell=turbine+bypass+header_bypass-condensate,
                    feed=condensate-feed+makeup-drain, makeup=-makeup, drain=drain)
    stocks = {}
    for name, index in _mass_indices(trial).items():
        entry = dict(mass_kg=float(p.state[index]), rate_kg_s=float(rate[index]))
        if name in expected:
            entry['expected_rate_kg_s'] = expected[name]
            entry['continuity_residual_kg_s'] = entry['rate_kg_s']-expected[name]
        if name in p.feed_tanks:
            # Adiabatic cushions hit the declared pressure domain before gas
            # volume reaches zero. Temperature/suction constraints may be earlier.
            minimum, capacity = _bounds(p, name)
            entry.update(minimum_mass_kg=minimum, capacity_mass_kg=capacity,
                         usable_mass_kg=max(0., entry['mass_kg']-minimum),
                         free_capacity_kg=max(0., capacity-entry['mass_kg']))
        stocks[name] = entry
    snap = p.snapshot()
    residual = max(abs(s.get('continuity_residual_kg_s', 0.)) for s in stocks.values())
    return dict(format='station-inventory/v1', time_s=float(p.time), stocks=stocks,
                total_mass_kg=math.fsum(s['mass_kg'] for s in stocks.values()),
                total_rate_kg_s=math.fsum(s['rate_kg_s'] for s in stocks.values()),
                maximum_local_continuity_residual_kg_s=residual,
                energy_relative=float(snap['energy_relative']),
                mass_relative=float(snap['mass_relative']), boron_relative=float(snap['boron_relative']),
                trip=bool(trial.control.trip_latched),
                scope='Privileged model diagnostics; not agent sensors or experimental validation')


def _validate(records):
    if type(records) not in (list, tuple) or not 2 <= len(records) <= 8193:
        raise ValueError('Inventory trace requires 2..8193 records')
    previous = None
    names = None
    domains = None
    for row in records:
        if type(row) is not dict or row.get('format') != 'station-inventory/v1':
            raise ValueError('Inventory record format')
        time = row.get('time_s')
        if not _finite(time) or time < 0 or (previous is not None and time <= previous):
            raise ValueError('Inventory monotone finite clock')
        previous = time
        stocks = row.get('stocks')
        if type(stocks) is not dict or not stocks or len(stocks) > 64 or any(type(k) is not str for k in stocks):
            raise ValueError('Inventory stocks')
        if names is None:
            names = set(stocks)
            domains = {name: {key:entry.get(key) for key in ('minimum_mass_kg', 'capacity_mass_kg')}
                       for name,entry in stocks.items() if type(entry) is dict}
        if set(stocks) != names:
            raise ValueError('Inventory stock schema changed')
        for name,entry in stocks.items():
            required = {'mass_kg', 'rate_kg_s', 'expected_rate_kg_s', 'continuity_residual_kg_s'}
            if type(entry) is not dict or not required <= set(entry):
                raise ValueError('Inventory mass/rate and local continuity evidence required')
            if any(not _finite(v) for v in entry.values()) or entry['mass_kg'] <= 0:
                raise ValueError('Inventory finite positive mass')
            bounded = {'minimum_mass_kg', 'capacity_mass_kg', 'usable_mass_kg', 'free_capacity_kg'}
            if bounded & set(entry):
                if not bounded <= set(entry) or not 0 < entry['minimum_mass_kg'] < entry['capacity_mass_kg']:
                    raise ValueError('Inventory reserve domain')
                if not entry['minimum_mass_kg'] <= entry['mass_kg'] <= entry['capacity_mass_kg']:
                    raise ValueError('Inventory reserve outside admissibility')
                if abs(entry['usable_mass_kg']-(entry['mass_kg']-entry['minimum_mass_kg'])) > 1e-8 or abs(entry['free_capacity_kg']-(entry['capacity_mass_kg']-entry['mass_kg'])) > 1e-8:
                    raise ValueError('Inventory reserve margins inconsistent')
            if {key:entry.get(key) for key in ('minimum_mass_kg', 'capacity_mass_kg')} != domains.get(name):
                raise ValueError('Inventory mass/pressure domain changed within trace')
            if ('expected_rate_kg_s' in entry) != ('continuity_residual_kg_s' in entry):
                raise ValueError('Inventory local continuity evidence incomplete')
            if 'expected_rate_kg_s' in entry and abs(entry['continuity_residual_kg_s']-(entry['rate_kg_s']-entry['expected_rate_kg_s'])) > 1e-10:
                raise ValueError('Inventory local continuity residual inconsistent')
        for key in ('total_mass_kg', 'total_rate_kg_s', 'maximum_local_continuity_residual_kg_s', 'energy_relative', 'mass_relative', 'boron_relative'):
            if not _finite(row.get(key)):
                raise ValueError('Inventory finite audit fields')
        if type(row.get('trip')) is not bool or any(row[k] < 0 for k in ('energy_relative', 'mass_relative', 'boron_relative')):
            raise ValueError('Inventory physical audit fields')
        if abs(row['total_mass_kg']-math.fsum(s['mass_kg'] for s in stocks.values())) > 1e-8 or abs(row['total_rate_kg_s']-math.fsum(s['rate_kg_s'] for s in stocks.values())) > 1e-10:
            raise ValueError('Inventory global totals inconsistent')
        maximum = max(abs(s.get('continuity_residual_kg_s', 0.)) for s in stocks.values())
        if abs(row['maximum_local_continuity_residual_kg_s']-maximum) > 1e-10:
            raise ValueError('Inventory maximum local residual inconsistent')
    return sorted(names)


def summarize(records, *, requested_horizon_s=3600., maximum_gap_s=5.):
    """Explain local redistribution, observed trends, and conditional reserve margins.

    Trapezoidal integration is explicitly approximate. Its residual cannot be
    interpreted as the solver's conservation residual or physical leakage.
    """
    if not _finite(requested_horizon_s) or not 0 < requested_horizon_s <= 7200 or not _finite(maximum_gap_s) or maximum_gap_s <= 0:
        raise ValueError('Inventory diagnostic horizon/gap')
    names = _validate(records)
    duration = records[-1]['time_s']-records[0]['time_s']
    gap = max(b['time_s']-a['time_s'] for a, b in zip(records, records[1:]))
    rows = []
    for name in names:
        samples = [r['stocks'][name] for r in records]
        first, last = samples[0], samples[-1]
        change = last['mass_kg']-first['mass_kg']
        integral = math.fsum(.5*(a['stocks'][name]['rate_kg_s']+b['stocks'][name]['rate_kg_s'])*(b['time_s']-a['time_s']) for a, b in zip(records, records[1:]))
        average = change/duration
        entry = dict(name=name, initial_mass_kg=first['mass_kg'], final_mass_kg=last['mass_kg'],
                     change_kg=change, mean_observed_rate_kg_s=average, final_rate_kg_s=last['rate_kg_s'],
                     span_kg=max(s['mass_kg'] for s in samples)-min(s['mass_kg'] for s in samples),
                     sampled_rate_integral_kg=integral, quadrature_residual_kg=change-integral)
        if 'minimum_mass_kg' in last:
            entry.update(usable_mass_kg=last['usable_mass_kg'], free_capacity_kg=last['free_capacity_kg'])
            # Separate observed mean from endpoint rate; do not imply a changing
            # rate will remain constant or that an endpoint zero proves safety.
            forecasts = {}
            for key, speed in [('observed_mean', average), ('endpoint', last['rate_kg_s'])]:
                reserve = last['usable_mass_kg'] if speed < 0 else last['free_capacity_kg']
                limit = None if speed == 0 else reserve/abs(speed)
                forecasts[key] = dict(rate_kg_s=speed, seconds_to_model_boundary=limit,
                                      boundary='minimum_mass_or_pressure' if speed < 0 else 'maximum_cushion_pressure' if speed > 0 else None,
                                      reaches_boundary_within_requested_horizon=limit is not None and limit <= requested_horizon_s)
            entry['conditional_linear_forecasts'] = forecasts
        rows.append(entry)
    rows.sort(key=lambda r: (-abs(r['mean_observed_rate_kg_s']), r['name']))
    return dict(format='station-inventory-summary/v1', start_s=records[0]['time_s'], end_s=records[-1]['time_s'],
                duration_s=duration, sample_count=len(records), maximum_gap_s=gap, sampling_adequate=gap <= maximum_gap_s+1e-9,
                requested_horizon_s=requested_horizon_s, stocks=rows,
                total_mass_change_kg=records[-1]['total_mass_kg']-records[0]['total_mass_kg'],
                maximum_total_rate_kg_s=max(abs(r['total_rate_kg_s']) for r in records),
                maximum_local_continuity_residual_kg_s=max(r['maximum_local_continuity_residual_kg_s'] for r in records),
                maximum_audits={k:max(r[k] for r in records) for k in ('energy_relative', 'mass_relative', 'boron_relative')},
                trip_observed=any(r['trip'] for r in records), stationary_accepted=False, r1_accepted=False,
                forecast_scope='Conditional constant-rate extrapolation to mass/pressure bounds only; temperature, phase or suction limits may occur earlier; not a guaranteed operating horizon',
                integration_scope='Sampled trapezoidal quadrature; residual includes sampling error and is not a solver balance error')


def archive_summary(rows, owner, *, window_s=600.):
    """Mass trends from existing raw-state rows, without inventing absent RHS rates."""
    if type(owner) is not FaultSession or type(rows) not in (list, tuple) or not 2 <= len(rows) <= 8193 or not _finite(window_s) or window_s <= 0:
        raise ValueError('Archived inventory owner/window/rows')
    layout = schema(owner)
    indices = _mass_indices(owner)
    previous = None
    for row in rows:
        if type(row) is not dict or not _finite(row.get('time_s')) or row['time_s'] < 0 or (previous is not None and row['time_s'] <= previous):
            raise ValueError('Archive monotone finite clock')
        values = row.get('physical_state')
        if type(values) is not list or len(values) != layout['size'] or any(not _finite(v) for v in values):
            raise ValueError('Archive full finite physical state required')
        if any(values[i] <= 0 for i in indices.values()):
            raise ValueError('Archive positive inventories required')
        previous = row['time_s']
    end = rows[-1]['time_s']
    window = [r for r in rows if r['time_s'] >= end-window_s-1e-9]
    if len(window) < 2 or window[0]['time_s'] > end-window_s+1e-8:
        raise ValueError('Archive incomplete inventory window')
    duration = window[-1]['time_s']-window[0]['time_s']
    stocks = []
    for name, index in indices.items():
        values = [r['physical_state'][index] for r in window]
        stocks.append(dict(name=name, initial_mass_kg=values[0], final_mass_kg=values[-1],
                           change_kg=values[-1]-values[0], mean_observed_rate_kg_s=(values[-1]-values[0])/duration,
                           span_kg=max(values)-min(values)))
    stocks.sort(key=lambda r: (-abs(r['mean_observed_rate_kg_s']), r['name']))
    totals = [math.fsum(r['physical_state'][i] for i in indices.values()) for r in window]
    by_name = {s['name']:s for s in stocks}
    reserves = {}
    for name in owner.control.plant.feed_tanks:
        stock = by_name[name]
        minimum, capacity = _bounds(owner.control.plant, name)
        final = stock['final_mass_kg']
        if not minimum <= final <= capacity:
            raise ValueError('Archived tank outside declared mass/pressure bounds')
        average = stock['mean_observed_rate_kg_s']
        margin = final-minimum if average < 0 else capacity-final
        reserves[name] = dict(minimum_mass_kg=minimum, capacity_mass_kg=capacity,
                              usable_mass_kg=final-minimum, free_capacity_kg=capacity-final,
                              mean_observed_rate_kg_s=average,
                              conditional_seconds_to_mass_or_pressure_bound=None if average==0 else margin/abs(average))
    makeup_used = max(0., -by_name['makeup']['change_kg'])
    drain_received = max(0., by_name['drain']['change_kg'])
    return dict(format='station-inventory-archive/v1', start_s=window[0]['time_s'], end_s=end, sample_count=len(window),
                maximum_gap_s=max(b['time_s']-a['time_s'] for a, b in zip(window, window[1:])),
                stocks=stocks, total_mass_change_kg=totals[-1]-totals[0], total_mass_span_kg=max(totals)-min(totals),
                reserve_margins=reserves,
                reserve_exchange=dict(makeup_mass_used_kg=makeup_used, drain_mass_received_kg=drain_received,
                    opposing_reserve_transfers_observed=makeup_used>0 and drain_received>0,
                    minimum_opposing_transfer_kg=min(makeup_used, drain_received),
                    net_feed_contribution_kg=makeup_used-drain_received,
                    scope='Net stock changes over the same window; does not prove simultaneous valve opening or identify individual water parcels'),
                forecast_scope='Constant observed mean rate to mass/pressure bounds only; phase, suction or temperature limits may be earlier',
                instantaneous_rates_available=False, stationary_accepted=False, r1_accepted=False,
                scope='Raw-coordinate trends only; no reconstructed instantaneous RHS or invented local residuals')
