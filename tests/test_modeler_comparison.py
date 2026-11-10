from copy import deepcopy
import math
import pytest
from scripts import compare_modelers as comparison


def timings():
    return dict(passed=True,status='completed',timings={
        'atomic':dict(samples_ns_per_state=[10.,11.,12.,13.,14.],median_ns_per_state=12.,scope='test'),
        'coolprop_if97':dict(samples_ns_per_state=[1.,2.,3.,4.,5.],median_ns_per_state=3.)})


def test_speed_is_component_only_with_no_invented_confidence():
    result=comparison.speed_profile(timings())
    assert result['coolprop_speedup_over_atomic']==4.
    assert result['confidence_interval'] is None

@pytest.mark.parametrize('damage',['negative','nan','boolean','median','accuracy','short'])
def test_speed_cannot_accept_bad_or_ineligible_measurements(damage):
    profile=timings();samples=profile['timings']['atomic']['samples_ns_per_state']
    if damage=='negative':samples[0]=-1.
    elif damage=='nan':samples[0]=math.nan
    elif damage=='boolean':samples[0]=True
    elif damage=='median':profile['timings']['atomic']['median_ns_per_state']=1.
    elif damage=='accuracy':profile['passed']=False
    else:samples.pop()
    with pytest.raises(ValueError):comparison.speed_profile(profile)


def modelica():
    return dict(passed=True,source_unchanged=True,records=[dict(case=c,method=m,passed=True)
        for c in comparison.modelica.heated.CASES for m in comparison.modelica.METHODS])


def test_external_coverage_is_exact_cartesian_product():
    assert comparison.check_modelica(modelica())['passed']==6

@pytest.mark.parametrize('damage',['missing','duplicate','failed','source'])
def test_external_qualification_rejects_incomplete_or_failed_matrix(damage):
    report=modelica()
    if damage=='missing':report['records'].pop()
    elif damage=='duplicate':report['records'][-1]=deepcopy(report['records'][0])
    elif damage=='failed':report['records'][0]['passed']=False
    else:report['source_unchanged']=False
    with pytest.raises(ValueError):comparison.check_modelica(report)


def agents():
    return dict(policies={k:dict(episodes=80,completed=80,mean_iae_completed_s=.01) for k in ['hold','pid','rule']},
        split='test',public_corpus=True,blind_test=False,recorded_episodes=240,paired=[],bootstrap={})


def test_public_agent_results_never_become_external_environment_comparison():
    result=comparison.check_agents(agents())
    assert not result['blind_test'] and not result['external_environment_compared']

@pytest.mark.parametrize('damage',['missing','count','metric','blind'])
def test_agent_summary_refuses_unsupported_claims_or_invalid_metrics(damage):
    report=agents()
    if damage=='missing':del report['policies']['hold']
    elif damage=='count':report['policies']['hold']['completed']=81
    elif damage=='metric':report['policies']['pid']['mean_iae_completed_s']=math.nan
    else:report['blind_test']=True
    with pytest.raises(ValueError):comparison.check_agents(report)


def test_cross_environment_source_check_keeps_code_hashes():
    a=dict(python='3.12',source=dict(numpy='2.2',model='a'))
    b=dict(python='3.13',source=dict(numpy='2.3',model='a'))
    assert comparison.code_only(a)==comparison.code_only(b)
    b['source']['model']='changed'
    assert comparison.code_only(a)!=comparison.code_only(b)
