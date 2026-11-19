"""Short actual lifecycle/restart fixtures; deliberately not a stationarity proof."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_campaign.engine import signature
from atomic_model_station_hot import HotConfig, HotSession, signature as hot_signature
from atomic_model_station_readiness import ReadinessConfig
from scripts.qualify_station_inventory import read


def successful_fixture():
    return HotConfig(maximum_warmup_s=.04,hold_s=.03,tick_s=.01,max_step_s=.1,power_target_w=3e9,
        readiness=ReadinessConfig(window_s=.02,maximum_gap_s=.01,relative_span=1.,stored_power_relative=1.))


def run(output,reference=None):
    out=Path(output).absolute()
    if any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Symlink output refused')
    out.mkdir(parents=True,exist_ok=False)
    coarse_cfg=successful_fixture();fine_cfg=replace(coarse_cfg,max_step_s=.01,rtol=2e-8)
    strict_cfg=replace(coarse_cfg,readiness=ReadinessConfig(window_s=.02,maximum_gap_s=.01,relative_span=1e-12))
    protocol=dict(source=signature(),hot_source=hot_signature(),driver_sha256=sha(__file__),
        coarse=asdict(coarse_cfg),fine=asdict(fine_cfg),strict=asdict(strict_cfg),
        maximum_relative_refinement=1e-3,stationary_accepted=False,closed_tasks=[],
        scope='Short permissive lifecycle fixture plus strict refusal; not industrial stationary criteria')
    if reference is not None and read(Path(reference)/'protocol.json')!=protocol:
        raise ValueError('Installed hot lifecycle identity differs')
    atomic_json(out/'protocol.json',seal(protocol))
    cases={};saved={}
    for name,config in [('coarse',coarse_cfg),('fine',fine_cfg),('strict',strict_cfg)]:
        case=HotSession(config)
        while not case.finished:
            case.step()
            if name=='coarse' and case.attempts in (2,4):
                stage='warm' if case.attempts==2 else 'hold'
                saved[stage]=case.checkpoint();atomic_json(out/(stage+'-checkpoint.json'),saved[stage])
        cases[name]=case
        atomic_json(out/(name+'-trace.json'),seal(dict(records=case.records,layout=case.layout)))
        atomic_json(out/(name+'-final.json'),case.checkpoint())
        atomic_json(out/(name+'-summary.json'),seal(dict(**case.report(),inventory=case.inventory_summary())))
    exact=True
    for stage,checkpoint in saved.items():
        restored=HotSession.restore(checkpoint)
        while not restored.finished:restored.step()
        exact=exact and restored.checkpoint()==cases['coarse'].checkpoint()
        atomic_json(out/('restart-'+stage+'-final.json'),restored.checkpoint())
    excluded=set(cases['coarse'].layout['accounting_indices'])
    aligned=len(cases['coarse'].records)==len(cases['fine'].records)
    difference=max(abs(a-b)/max(abs(b),1.) for x,y in zip(cases['coarse'].records,cases['fine'].records)
        for i,(a,b) in enumerate(zip(x['sample']['state'],y['sample']['state'])) if i not in excluded)
    audits=all(c.inventory_summary()['maximum_local_continuity_residual_kg_s']<=1e-9
        and abs(c.inventory_summary()['total_mass_change_kg'])<=1e-8
        and c.inventory_summary()['maximum_audits']['energy_relative']<=1e-6
        and c.inventory_summary()['maximum_audits']['mass_relative']<=1e-10
        and c.inventory_summary()['maximum_audits']['boron_relative']<=1e-10 for c in cases.values())
    unchanged=(protocol['source']==signature() and protocol['hot_source']==hot_signature() and protocol['driver_sha256']==sha(__file__))
    passed=bool(unchanged and exact and audits and aligned and difference<=protocol['maximum_relative_refinement']
        and cases['coarse'].phase==cases['fine'].phase=='complete' and cases['strict'].phase=='unprepared'
        and all(not c.report()['canonical_hold_passed'] and not c.report()['stationary_accepted'] for c in cases.values()))
    report=dict(passed=passed,completed=True,source_unchanged=unchanged,restart_exact=exact,
        refinement_relative=difference,strict_refusal=cases['strict'].phase=='unprepared',physical_audits_passed=audits,
        short_hold_complete=cases['coarse'].phase=='complete',duration_s=cases['coarse'].report()['time_s'],
        stationary_accepted=False,r1_accepted=False,closed_tasks=[],
        accepted_scope='Lifecycle transactions, both restart phases and strict refusal; 0.06s permissive fixture only')
    atomic_json(out/'report.json',seal(report))
    if reference is not None:
        expected={p.relative_to(reference) for p in Path(reference).rglob('*.json')}
        actual={p.relative_to(out) for p in out.rglob('*.json')}
        if expected!=actual or any((Path(reference)/p).read_bytes()!=(out/p).read_bytes() for p in actual):
            raise ValueError('Hot source/installed evidence differs')
    if not passed:raise ValueError('Hot lifecycle qualification failed; inspect preserved report')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reference',type=Path)
    args=parser.parse_args();print(json.dumps(run(args.output,args.reference),sort_keys=True))


if __name__=='__main__':main()
