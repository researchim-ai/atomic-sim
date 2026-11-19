from copy import deepcopy
import pytest
from atomic_model_station import PWRPowerSession,FeedConfig
from atomic_model_station_hybrid_heat.events import assemble_segments
from atomic_model_station_hybrid_heat import record_hybrid_window
from atomic_model_station_jump_heat import assess_segments


def sample(t,heat=100.,stored=None):
    boundary=10*t if stored is None else stored-1000.
    return dict(energy=dict(time_s=t,counters=dict(gross_generated_j=40*t,own_needs_j=10*t,
        net_generated_j=30*t,net_exported_j=30*t,backup_used_j=0.),
        powers=dict(gross_electrical_w=40.,own_needs_w=10.,net_generation_w=30.,net_export_w=30.),
        grid_available=True,trip=False,energy_relative=0.),stored_j=1000.+boundary,boundary_j=boundary,
        fluxes=dict(nuclear_heat_w=heat,primary_pump_loss_out_w=0.,external_cooling_heat_out_w=60.,external_net_w=heat-90.))


def fixture():
    a=sample(1.);b=sample(1.,150.);c=sample(1.,200.)
    events=[dict(time_s=1.,kind='feed_check_closed',before=a,after=b),
            dict(time_s=1.,kind='condensate_check_closed',before=deepcopy(b),after=c)]
    return [sample(0.),sample(2.,200.,1120.)],events


def test_simultaneous_maps_have_one_boundary_and_exact_integral():
    samples,events=fixture();original=deepcopy((samples,events))
    result=assemble_segments(samples,events)
    assert (samples,events)==original
    assert result['simultaneous_group_count']==1 and len(result['segments'])==2
    assert [e['kind'] for e in result['groups'][0]['events']]==['feed_check_closed','condensate_check_closed']
    audit=assess_segments(result['segments'])
    assert audit['passed'] and audit['sampled_integrals_j']['nuclear_heat_w']==300.


def test_cancelling_intermediate_impulses_are_not_erased():
    samples,events=fixture()
    events[0]['after']['stored_j']+=.0006;events[1]['before']['stored_j']+=.0006
    with pytest.raises(ValueError,match='Accumulated'):assemble_segments(samples,events)


def test_broken_intermediate_chain_is_rejected():
    samples,events=fixture();events[1]['before']['energy']['trip']=True
    with pytest.raises(ValueError,match='intermediate event chain'):assemble_segments(samples,events)


@pytest.mark.parametrize('time',[0.,2.,float('nan'),float('inf')])
def test_invalid_or_boundary_event_times_are_refused(time):
    samples,events=fixture();events[0]['time_s']=time
    with pytest.raises(ValueError):assemble_segments(samples,events)


def test_close_but_distinct_events_are_not_merged():
    samples,events=fixture();t=1.+1e-10;events[1]['time_s']=t
    for side in ('before','after'):events[1][side]['energy']['time_s']=t
    result=assemble_segments(samples,events)
    assert result['simultaneous_group_count']==0 and len(result['segments'])==3
    assert result['segments'][1][-1]['energy']['time_s']>result['segments'][1][0]['energy']['time_s']


def test_existing_grid_sample_is_replaced_by_event_sides():
    samples,events=fixture();samples.insert(1,sample(1.))
    result=assemble_segments(samples,events)
    assert [len(s) for s in result['segments']]==[2,2]
    assert assess_segments(result['segments'])['passed']


def test_event_side_clock_and_sequence_are_checked():
    samples,events=fixture();events[1]['after']['energy']['time_s']=1.1
    with pytest.raises(ValueError,match='clock mismatch'):assemble_segments(samples,events)
    samples,events=fixture();samples.reverse()
    with pytest.raises(ValueError,match='sample clock'):assemble_segments(samples,events)


def test_condensate_close_window_passes_and_original_is_unchanged():
    session=PWRPowerSession(feed=FeedConfig(condensate_inertance_m_inv=1.),rtol=1e-10,max_step_s=.01)
    session.advance(.005,{'condensate_pump_on':False});saved=session.checkpoint()
    result=record_hybrid_window(session,.01,subdivisions=16)
    assert result['passed'] and session.checkpoint()==saved
    assert [e['kind'] for e in result['events']]==['condensate_check_closed']
    assert len(result['event_groups'])==1
