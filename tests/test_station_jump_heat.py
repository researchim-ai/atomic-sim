from copy import deepcopy
import pytest
from atomic_model_station import PWRPowerSession,BackupConfig
from atomic_model_station_heat_accounting import assess_heat
from atomic_model_station_jump_heat import assess_segments,record_power_window
from atomic_model_station_jump_heat import model


def sample(t,after=False):
    boundary=10*t if not after else 10+110*(t-1)
    return dict(energy=dict(time_s=t,counters=dict(gross_generated_j=40*t,own_needs_j=10*t,
        net_generated_j=30*t,net_exported_j=30*t,backup_used_j=0.),
        powers=dict(gross_electrical_w=40.,own_needs_w=10.,net_generation_w=30.,net_export_w=30.),
        grid_available=True,trip=False,energy_relative=0.),stored_j=1000+boundary,boundary_j=boundary,
        fluxes=dict(nuclear_heat_w=200. if after else 100.,primary_pump_loss_out_w=0.,
            external_cooling_heat_out_w=60.,external_net_w=110. if after else 10.))


def segments():return [[sample(0.),sample(1.)],[sample(1.,True),sample(2.,True)]]


def test_exact_piecewise_integral_does_not_smear_jump():
    rows=segments();saved=deepcopy(rows);result=assess_segments(rows)
    assert result['passed'] and rows==saved
    assert result['sampled_integrals_j']['nuclear_heat_w']==300.
    assert result['boundaries'][0]['flux_jumps_w']['external_net_w']==100.
    assert result['absolute_defect_sums_j']==dict(heat_closure_j=0.,window_storage_defect_j=0.,boundary_quadrature_defect_j=0.)
    assert not assess_heat([rows[0][0],rows[0][1],rows[1][1]])['passed']


@pytest.mark.parametrize('field',['stored_j','boundary_j'])
def test_constant_inventory_offset_cannot_hide_event_impulse(field):
    rows=segments()
    for s in rows[1]:s[field]+=1.
    assert 'boundary_energy_impulse' in assess_segments(rows)['reasons']


def test_counter_offsets_across_event_are_refused():
    rows=segments()
    for s in rows[1]:
        for k in ('gross_generated_j','net_generated_j','net_exported_j'):s['energy']['counters'][k]+=5.
    assert 'boundary_energy_impulse' in assess_segments(rows)['reasons']


def test_boundary_gap_is_not_time_integrated():
    rows=segments();rows[1][0]['energy']['time_s']+=.01
    assert 'boundary_clock' in assess_segments(rows)['reasons']


@pytest.mark.parametrize('rows',[[],[[]],[[sample(0.)]]])
def test_empty_segments_are_not_qualified(rows):
    with pytest.raises(ValueError):assess_segments(rows)


def test_nonfinite_sample_is_rejected():
    rows=segments();rows[1][0]['fluxes']['nuclear_heat_w']=float('nan')
    with pytest.raises(ValueError):assess_segments(rows)


def prepared():
    session=PWRPowerSession(backup=BackupConfig(initial_energy_j=1e4),rtol=1e-10)
    session.advance(.08,{'grid_available':False});return session


def test_backup_root_records_two_sides_without_mutating_original():
    session=prepared();before=session.checkpoint()
    result=record_power_window(session,.04,subdivisions=16)
    assert session.checkpoint()==before and result['native_root_match'] and result['passed']
    assert len(result['events'])==1
    event=result['events'][0]
    assert event['time_s']==pytest.approx(.1,abs=1e-9)
    assert event['before']['energy']['time_s']==event['after']['energy']['time_s']
    assert event['before']['energy']['powers']['own_needs_w']==1e5
    assert event['after']['energy']['powers']['own_needs_w']==0.
    assert abs(event['before']['stored_j']-event['after']['stored_j'])<=1e-3


def test_unsupported_full_grid_loss_window_fails_without_mutation():
    session=PWRPowerSession(backup=BackupConfig(initial_energy_j=1e4),rtol=1e-10);before=session.checkpoint()
    with pytest.raises(ValueError,match='Unsupported root'):
        record_power_window(session,.12,{'grid_available':False},subdivisions=8)
    assert session.checkpoint()==before


def test_source_change_does_not_mutate_original(monkeypatch):
    session=prepared();before=session.checkpoint();versions=iter([1,2])
    monkeypatch.setattr(model,'identity',lambda:next(versions))
    with pytest.raises(ValueError,match='source changed'):record_power_window(session,.04,subdivisions=8)
    assert session.checkpoint()==before


def test_small_impulses_cannot_accumulate_unchecked():
    a=[sample(0.),sample(1.)]
    b=[sample(1.),sample(2.)]
    c=[sample(2.),sample(3.)]
    for row in b:row['stored_j']+=.0006
    for row in c:row['stored_j']+=.0012
    report=assess_segments([a,b,c])
    assert 'boundary_energy_impulse' not in report['reasons']
    assert 'accumulated_boundary_impulse' in report['reasons']


def test_terminal_exhaustion_requires_a_right_interval():
    session=prepared();before=session.checkpoint()
    with pytest.raises(ValueError,match='[Tt]erminal'):
        record_power_window(session,.02,subdivisions=8)
    assert session.checkpoint()==before
