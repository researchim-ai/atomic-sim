"""Rebuild evidence by facet; completed comparisons never imply a global winner."""
import argparse
import json
import math
from pathlib import Path
import statistics
from atomic_model_hem_heated.model import signature
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import atomic_json
from scripts import benchmark_external as external
from scripts import benchmark_modelica_heated as modelica
from agent_envs.benchmark import rebuild
from sdk.experiment import _source_fingerprint


def read(path):return unseal(json.loads(Path(path).read_text()))


def code_only(value):
    if isinstance(value,dict):return {k:code_only(v) for k,v in value.items() if k not in ('python','numpy','scipy')}
    if isinstance(value,list):return [code_only(v) for v in value]
    return value


def speed_profile(profile):
    if profile.get('passed') is not True or profile.get('status')!='completed':raise ValueError('EOS accuracy must pass before speed reporting')
    medians={}
    for key in ('atomic','coolprop_if97'):
        result=profile['timings'][key];samples=result['samples_ns_per_state']
        if len(samples)<5 or any(type(v) not in (float,int) or not math.isfinite(v) or v<=0 for v in samples):raise ValueError('At least five finite positive timing trials required')
        median=statistics.median(samples)
        if result['median_ns_per_state']!=median:raise ValueError('Timing median mismatch')
        medians[key]=median
    return dict(median_ns_per_state=medians,coolprop_speedup_over_atomic=medians['atomic']/medians['coolprop_if97'],
        scope=profile['timings']['atomic']['scope'],confidence_interval=None,
        limitation='Seven alternating-order warm-cache trials; no randomized order, confidence interval or whole-environment speed comparison')


def check_modelica(report):
    expected={(case,method) for case in modelica.heated.CASES for method in modelica.METHODS}
    rows=report['records'];keys=[(r.get('case'),r.get('method')) for r in rows]
    if len(keys)!=len(expected) or set(keys)!=expected or len(set(keys))!=len(keys):raise ValueError('Missing/duplicate external heated comparisons')
    if report.get('passed') is not True or report.get('source_unchanged') is not True or any(r.get('passed') is not True for r in rows):raise ValueError('External heated comparison failed')
    return dict(comparisons=len(rows),passed=len(rows),scope='Same spatial equations, independent Modelica/MSL implementation; not off-the-shelf Modelica.Fluid')


def check_agents(report):
    policies=report['policies']
    if set(policies)!={'hold','pid','rule'} or report.get('split')!='test' or report.get('public_corpus') is not True or report.get('blind_test') is not False:raise ValueError('Public agent test corpus required')
    if report['recorded_episodes']!=240 or any(p['episodes']!=80 for p in policies.values()):raise ValueError('Incomplete agent coverage')
    for p in policies.values():
        if type(p['completed']) is not int or not 0<=p['completed']<=p['episodes']:raise ValueError('Invalid completion count')
        value=p['mean_iae_completed_s']
        if p['completed'] and (type(value) not in (float,int) or not math.isfinite(value) or value<0):raise ValueError('Invalid agent metric')
    return dict(policies=policies,paired=report['paired'],bootstrap=report['bootstrap'],
                public_corpus=True,blind_test=False,external_environment_compared=False,
                limitation='Near-nominal two-second scenarios; synthetic sensor fault labels; baseline controllers, not trained RL/LLM agents')


def run(output,*,eos,heated,agents):
    output,eos,heated,agents=map(lambda p:Path(p).absolute(),(output,eos,heated,agents))
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    output.mkdir(parents=True,exist_ok=False)
    er=read(eos/'report.json');ep=read(eos/'protocol.json');hr=read(heated/'report.json');hp=read(heated/'protocol.json')
    if not er['source_unchanged'] or code_only(ep['source'])!=code_only(external.source_manifest()):raise ValueError('EOS code source changed')
    if hp['source']!=signature() or any(sha(p)!=v for p,v in hp['drivers'].items()):raise ValueError('Heated benchmark source changed')
    ar=rebuild(agents)
    if ar['source_sha256']!=_source_fingerprint():raise ValueError('Agent runtime source changed')
    result=dict(completed=True,eos=speed_profile(er['profiles']['coolprop']),heated=check_modelica(hr),agents=check_agents(ar),
                superiority_established=False,whole_plant_compared=False,validated=False,plant_referenced=False,
                conclusion='Component agreement established; CoolProp speed and controller performance measured separately; no global modeler ranking',
                remaining=['Randomized paired end-to-end environment throughput at matched accuracy','Equivalent external closed-loop PWR model','Long-horizon and out-of-distribution control tasks','Trained-agent learning curves at equal compute budget','Experimental validation'],
                evidence={str(p):sha(p) for p in [eos/'report.json',eos/'protocol.json',heated/'report.json',heated/'protocol.json',agents/'manifest.json',agents/'report.json']})
    atomic_json(output/'report.json',seal(result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('output','eos','heated','agents'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();run(args.output,eos=args.eos,heated=args.heated,agents=args.agents)
    print('Comparative evidence rebuilt; global superiority is not established')

if __name__=='__main__':main()
