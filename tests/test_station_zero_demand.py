"""Exact zero-flow demand preserves stocks and refuses branch changes."""
from copy import deepcopy
import math
import pytest
from core.status import DomainError
from atomic_model_station import PWRFeedSession


def declared_small_flow(cls=PWRFeedSession):
    s=cls(max_step_s=.2)
    s.scram_latched=True;s.turbine_tripped=True;s.state[12]=1.
    # A declared fresh initial momentum, not a modification of operating history.
    s.state[s.plant.feed_index]=1e-10;s.state[s.feed_index+6]=1e-10
    s.controls.update(feed_flow_control=True,feed_flow_fraction=0.,condensate_flow_control=True,condensate_flow_fraction=0.)
    s.origin_energy=s._stored();s._rhs(0.,s.state);s._admit(s.state)
    return s


def test_positive_exponential_survives_many_e_folds_with_stock_balances():
    s=declared_small_flow();before=s.state.copy();r=s.advance(2.)
    assert s.state[s.plant.feed_index]==pytest.approx(1e-10*math.exp(-10),rel=1e-14,abs=0.)
    assert s.state[s.feed_index+6]==pytest.approx(1e-10*math.exp(-4),rel=1e-14,abs=0.)
    assert s.state[s.plant.feed_index]>0 and s.state[s.feed_index+6]>0
    assert r['energy_relative']<1e-8 and r['mass_relative']<1e-10 and r['boron_relative']<1e-10
    assert r['feed_train']['makeup_flow_kg_s']==r['feed_train']['drain_flow_kg_s']==0
    assert any(m['kind']=='exact_closed_loop_exponential' for m in r['exact_mode_reduction'])
    assert not any('check_closed' in e['kind'] for e in r['events'])


def test_new_command_releases_exact_mode_without_flow_assignment():
    s=declared_small_flow();s.advance(.1);q=float(s.state[s.plant.feed_index])
    r=s.advance(.1,dict(feed_flow_fraction=.01,condensate_flow_fraction=.01))
    assert s.state[s.plant.feed_index]>q and s.state[s.feed_index+6]>0
    assert not any(m['kind']=='exact_closed_loop_exponential' for m in r['exact_mode_reduction'])
    assert r['energy_relative']<1e-8


def test_changed_zero_demand_branch_is_refused_and_rolled_back():
    class ChangingDemand(PWRFeedSession):
        def _rhs(self,t,y):
            rate=super()._rhs(t,y)
            if self.feed_ready and t>.01:self.plant.feed_controller['requested_flow_kg_s']=1.
            return rate
    s=declared_small_flow(ChangingDemand);before=s.checkpoint()
    with pytest.raises(ValueError,match='mode changed inside interval'):s.advance(.1)
    assert s.checkpoint()==before


def test_reverse_flow_is_still_strictly_rejected():
    s=declared_small_flow();state=s.state.copy();state[s.plant.feed_index]=-1e-14
    with pytest.raises(DomainError,match='nonnegative flow'):s._admit(state)


def test_saturated_controller_does_not_get_an_exact_decay_shortcut():
    s=declared_small_flow();s.controls['feed_pump_on']=False;s._rhs(0.,s.state)
    known={};modes=[];s._extend_known(known,modes)
    assert s.plant.feed_controller['saturated']
    assert s.plant.feed_index not in known
    assert 'feed' not in s._zero_demand_modes
