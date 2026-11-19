"""Explicit reproducible research cases; not plant-calibrated agent policies."""
from dataclasses import asdict
import math
from atomic_model_station import PWRPowerSession,FeedConfig


def prepare_condensate_reopening():
    """Return a naturally closed session, a sub-threshold pump command, and recipe.

    All state is obtained by native integration. This fixed case qualifies an
    internal opening; it does not claim an operating range for real equipment.
    """
    from .model import identity
    feed=FeedConfig(condensate_inertance_m_inv=1.)
    session=PWRPowerSession(feed=feed,rtol=1e-10,max_step_s=.01)
    session.advance(.006,{'condensate_pump_on':False})
    session.snapshot()
    if session.condensate_open or float(session.state[session.feed_index+6])!=0.:
        raise ValueError('Preparation did not reach a closed condensate valve')
    back=session.feed_terms['tanks']['feed']['pressure_pa']-session.plant.true_hotwell['p']
    margin=100.;head=back-margin
    if not math.isfinite(head) or not 0<head<2e6:raise ValueError('Prepared pump head outside command domain')
    patch=dict(condensate_pump_on=True,condensate_pump_fraction=math.sqrt(head/2e6))
    recipe=dict(format='condensate-internal-reopening/v1',source=identity(),feed=asdict(feed),
        preparation_s=.006,duration_s=.02,rtol=1e-10,max_step_s=.01,
        back_pressure_pa=float(back),head_margin_pa=margin,command=dict(patch),
        expected_event='condensate_check_opened',manual_state_projection=False,
        plant_calibrated=False,agent_policy=False)
    return session,patch,recipe


def prepare_feed_reopening():
    """Fixed small-gas-cushion research case with pressure-driven feed reopening."""
    from .model import identity
    feed=FeedConfig(feed_gas_volume_m3=.1)
    session=PWRPowerSession(feed=feed,rtol=1e-10,max_step_s=.01)
    session.advance(.008,{'feed_pump_on':False})
    session.snapshot()
    if session.plant._feed_open or float(session.state[session.plant.feed_index])!=0.:
        raise ValueError('Preparation did not reach a closed feed valve')
    back=session.plant.feed_terms()[2]
    margin=10000.;head=back-margin;maximum=session.plant.maximum_feed_head_pa
    if not math.isfinite(head) or not 0<head<maximum:raise ValueError('Prepared feed head outside command domain')
    patch=dict(feed_pump_on=True,feed_pump_fraction=math.sqrt(head/maximum))
    recipe=dict(format='feed-internal-reopening/v1',source=identity(),feed=asdict(feed),
        preparation_s=.008,duration_s=.02,rtol=1e-10,max_step_s=.01,
        back_pressure_pa=float(back),head_margin_pa=margin,command=dict(patch),
        expected_event='feed_check_opened',manual_state_projection=False,
        plant_calibrated=False,agent_policy=False)
    return session,patch,recipe


def prepare_relief_opening():
    """Fixed narrow-hysteresis relay case; not a licensed relief-valve setting."""
    from atomic_model_station import PWRBypassSession,BypassConfig
    from .model import identity
    bypass=BypassConfig(relief_open_pa=6368300.,relief_close_pa=6368290.)
    session=PWRBypassSession(bypass=bypass,rtol=1e-10,max_step_s=.01)
    session.advance(.016,{'turbine_trip':True})
    session.snapshot()
    if session.relief_latched or session.header.thermo.p_pa>=bypass.relief_open_pa:
        raise ValueError('Preparation is not before pressure-relief opening')
    recipe=dict(format='pressure-relief-opening/v1',source=identity(),bypass=asdict(bypass),
        preparation_s=.016,duration_s=.004,rtol=1e-10,max_step_s=.01,
        preparation_command=dict(turbine_trip=True),command=None,
        expected_event='bypass_relief_open',manual_state_projection=False,
        plant_calibrated=False,full_pressure_cycle_qualified=False)
    return session,None,recipe


def prepare_relief_cycle():
    """Declared short pressure cycle; numerical windows do not add commands."""
    from atomic_model_station import PWRBypassSession,BypassConfig
    from .model import identity
    bypass=BypassConfig(area_m2=.03,stroke_time_s=.1,relief_open_pa=6368300.,relief_close_pa=6368290.)
    session=PWRBypassSession(bypass=bypass,rtol=1e-10,max_step_s=.01)
    schedule=[dict(duration_s=dt,patch=patch) for dt,patch in [
        (.002,{'steam_admission':.99}),(.001,None),(.007,None),
        (.001,{'steam_admission':1.}),(.001,None),(.010,None)]]
    recipe=dict(format='short-pressure-relief-cycle/v1',source=identity(),bypass=asdict(bypass),
        rtol=1e-10,max_step_s=.01,schedule=schedule,duration_s=.022,
        expected_pressure_events=['bypass_relief_open','bypass_relief_close'],
        manual_state_projection=False,plant_calibrated=False,full_actuator_stroke=False)
    return session,schedule,recipe
