from copy import deepcopy
import numpy as np
import pytest
from scripts import station_benchmark as b
from atomic_model_station import StationConfig, StationResidual, build_r0


def samples(name='source-step'):
    spec=b.contract();case=next(c for c in spec['cases'] if c['name']==name)
    t,y=b._reference(StationConfig(**spec['config']),case,spec['sample_spacing_s'])
    return spec,case,t.tolist(),y.tolist()


def test_contract_pins_scope_units_budgets_and_all_cases():
    spec=b.contract()
    assert len(spec['cases'])==3 and len(spec['methods'])==4
    assert spec['units']==['K']*4+['J']*4
    assert not spec['plant_validation'] and not spec['full_station_comparison']
    assert b.contract()==spec


@pytest.mark.parametrize('power,connected',[(0.,True),(8e5,True),(1e6,False),(1.2e6,True)])
def test_independent_component_balances_equal_port_assembly(power,connected):
    c=StationConfig();rng=np.random.default_rng(918)
    matrix,constant=build_r0(c,power,connected).operators()
    for _ in range(20):
        y=StationResidual(c).steady()[0]*(1+rng.uniform(-.01,.01,5))
        rates=b.component_rates(c,y,power,connected)
        assert np.allclose(rates[:5],matrix@y+constant,rtol=1e-10,atol=1e-7)
        weights=np.array([c.primary_capacity_j_k,c.wall_capacity_j_k,c.secondary_capacity_j_k,c.cold_capacity_j_k,1.])
        assert weights@rates[:5]==pytest.approx(rates[5]-rates[6]-rates[7],abs=1e-8)


@pytest.mark.parametrize('name',['nominal','source-step','load-disconnect'])
def test_exact_reference_passes_all_samples_and_ledgers(name):
    spec,case,t,y=samples(name)
    r=b.assess_submission(spec['sha256'],name,t,y)
    assert r['full_grid'] and r['state_and_integrals'] and r['energy_budget']
    assert not r['external_execution_cost_verified']


@pytest.mark.parametrize('method',b.METHODS)
@pytest.mark.parametrize('case_index',range(3))
def test_actual_native_and_independent_solvers(method,case_index):
    spec=b.contract();r=b.run_attempt(spec,spec['cases'][case_index],method)
    assert r['passed'] and r['failure'] is None and len(r['values'])==41
    assert r['maximum_normalized_error']<spec['maximum_normalized_error']
    if method!='assembled-exact':assert r['rhs_calls']>=r['nfev']>0


def test_truncated_record_cannot_pass_horizon():
    spec,case,t,y=samples()
    r=b.assess_submission(spec['sha256'],case['name'],t[:3],y[:3])
    assert not r['full_grid'] and r['state_and_integrals']


@pytest.mark.parametrize('mutation',['missing','shift','reverse','nan','bool','width','counter-origin'])
def test_malformed_or_misaligned_submission_refused(mutation):
    spec,case,t,y=samples()
    if mutation=='missing':t.pop(3);y.pop(3)
    elif mutation=='shift':t[1]+=.001
    elif mutation=='reverse':t.reverse();y.reverse()
    elif mutation=='nan':y[2][0]=float('nan')
    elif mutation=='bool':y[2][0]=True
    elif mutation=='width':y[2].pop()
    else:y[0][5]=1.
    with pytest.raises(ValueError):b.assess_submission(spec['sha256'],case['name'],t,y)


def test_wrong_contract_or_case_cannot_be_compared():
    spec,case,t,y=samples()
    with pytest.raises(ValueError,match='contract'):b.assess_submission('0'*64,case['name'],t,y)
    with pytest.raises(ValueError,match='case'):b.assess_submission(spec['sha256'],'invented',t,y)


def test_errors_in_ledgers_are_not_removed_to_make_comparison_pass():
    spec,case,t,y=samples()
    y[-1][6]+=1e5
    r=b.assess_submission(spec['sha256'],case['name'],t,y)
    assert not r['state_and_integrals'] and not r['energy_budget']


def test_solver_failure_is_retained_with_initial_sample(monkeypatch):
    def fail(*args,**kwargs):raise RuntimeError('deliberate solver refusal')
    monkeypatch.setattr(b,'solve_ivp',fail)
    spec=b.contract();r=b.run_attempt(spec,spec['cases'][0],'Radau')
    assert not r['passed'] and r['failure']['message']=='deliberate solver refusal'
    assert r['times_s']==[0.] and len(r['values'])==1


def test_rhs_budget_is_enforced_inside_independent_solver(monkeypatch):
    initial=np.r_[StationResidual(StationConfig()).steady()[0],0.,0.,0.]/b.SCALES
    def exhaust(fun,*args,**kwargs):
        for _ in range(20001):fun(0.,initial)
    monkeypatch.setattr(b,'solve_ivp',exhaust)
    spec=b.contract();r=b.run_attempt(spec,spec['cases'][0],'BDF')
    assert not r['passed'] and not r['checks']['work_budget']
    assert r['failure']['message']=='rhs_budget'


def test_output_requires_fresh_ordinary_directory(tmp_path):
    with pytest.raises(ValueError,match='Fresh'):b.run(tmp_path)
    link=tmp_path/'link';link.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match='Fresh'):b.run(link/'nested')


def test_matrix_retains_failed_attempts_and_does_not_claim_global_superiority():
    spec=b.contract()
    attempts=[dict(case=c['name'],method=m,passed=False,maximum_normalized_error=1.,
                   wall_s=.1,rhs_calls=10) for c in spec['cases'] for m in spec['methods']]
    matrix=b.comparison_matrix(spec,attempts)
    assert not matrix['whole_simulator_superiority_established']
    assert matrix['methods']['BDF']['attempted']==3 and matrix['methods']['BDF']['passed']==0
    with pytest.raises(ValueError,match='exactly once'):b.comparison_matrix(spec,attempts[:-1])
    with pytest.raises(ValueError,match='exactly once'):b.comparison_matrix(spec,attempts+[attempts[0]])


def test_external_submission_uses_our_checks_and_keeps_unverified_identity(tmp_path):
    from atomic_model_training.spec import atomic_json
    spec,case,t,y=samples()
    data=dict(format='station-r0-submission/v1',contract_sha256=spec['sha256'],case=case['name'],
              simulator='external declared name',times_s=t,values=y)
    path=tmp_path/'input.json';atomic_json(path,data)
    result=b.assess_file(path,tmp_path/'accepted')
    assert result['passed'] and not result['simulator_identity_verified']
    assert not result['assessment']['external_execution_cost_verified']
    data['values'][-1][6]+=1e5;atomic_json(path,data)
    result=b.assess_file(path,tmp_path/'failed')
    assert not result['passed'] and (tmp_path/'failed/assessment.json').is_file()
    data['passed']=True;atomic_json(path,data)
    with pytest.raises(ValueError,match='schema'):b.assess_file(path,tmp_path/'forged')
