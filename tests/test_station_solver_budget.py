from copy import deepcopy
import io,json
import pytest
from core.status import DomainError,SolverError
from atomic_model_station import PWRFeedSession
from atomic_model_station import pwr_train
from atomic_model_station_faults import FaultSession
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_env.model import ActionError
from atomic_model_station_env.__main__ import serve


@pytest.mark.parametrize('limit',[1,5,50])
def test_rhs_work_limit_survives_jacobian_fallback_and_rolls_back_all_stocks(monkeypatch,limit):
    s=PWRFeedSession(max_step_s=.1);before=s.checkpoint()
    monkeypatch.setattr(pwr_train,'MAX_RHS_EVALUATIONS',limit)
    with pytest.raises(pwr_train.RHSWorkLimit) as info:s.advance(.1)
    assert info.value.requested==limit+1 and info.value.details()['limit']==limit
    assert s.checkpoint()==before


def test_solver_counters_are_deterministic_replay_data_not_wall_time():
    s=PWRFeedSession(max_step_s=.1);p=s.advance(.05);stats=p['solver_statistics']
    assert stats['rhs_evaluations']>=stats['nfev']>0 and stats['rhs_evaluations']<=stats['rhs_limit']
    assert stats['scope']=='last_integrated_segment' and 'wall_s' not in stats
    restored=PWRFeedSession.restore(s.checkpoint());assert restored.checkpoint()==s.checkpoint()


@pytest.mark.parametrize('error,code',[(DomainError,'model_domain_rejected'),(SolverError,'model_solver_rejected')])
def test_agent_errors_are_structured_and_atomic(monkeypatch,error,code):
    env=StationEnv(EnvConfig(max_steps=3,tick_s=.01));env.reset();before=env.checkpoint();observation=env.observation()
    def reject(self,*args,**kwargs):raise error('deliberate model refusal')
    monkeypatch.setattr(FaultSession,'step',reject)
    with pytest.raises(ActionError) as info:env.step()
    assert info.value.code==code and env.checkpoint()==before and env.observation()==observation
    stream=io.StringIO();serve(io.StringIO('{"method":"step"}\n{"method":"observe"}\n'),stream,env)
    values=[json.loads(line) for line in stream.getvalue().splitlines()]
    assert values[0]['error']['code']==code and values[1]['ok']
