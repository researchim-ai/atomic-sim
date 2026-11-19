"""Prospective six-loop split-range neutral-band experiment, not a stationary hold."""
import argparse
import json
from pathlib import Path

from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_faults import FaultSession
from atomic_model_station_campaign.engine import signature
from atomic_model_station_inventory import capture, summarize
from atomic_model_station_inventory import model as inventory_model
from atomic_model_station_readiness import schema
from scripts.qualify_station_inventory import read


def target(reference, index):
    return reference + (.0001,-.0001,.0001,.0006)[index//10]


def family(band, max_step_s, rtol):
    owner=FaultSession(tick_s=.1,inventory_deadband_m=band,max_step_s=max_step_s,rtol=rtol)
    reference=owner.control.targets['feed_inventory_fraction']
    records=[capture(owner)]; states=[]; results=[]; midpoint=None
    for i in range(40):
        result=owner.step(modes={k:'auto' for k in owner.control.loops} if i==0 else None,
                          targets={'feed_inventory_fraction':target(reference,i)})
        records.append(capture(owner)); results.append(result)
        states.append(owner.control.plant.state.tolist())
        if i==19:midpoint=owner.checkpoint()
    return owner,reference,records,states,results,midpoint


def exchange(records):
    a,b=records[0]['stocks'],records[-1]['stocks']
    return max(0.,a['makeup']['mass_kg']-b['makeup']['mass_kg'])+max(0.,b['drain']['mass_kg']-a['drain']['mass_kg'])


def run(output, reference=None):
    out=Path(output).absolute()
    if any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Symlink output refused')
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(format='station-inventory-neutral/v1',source=signature(),driver_sha256=sha(__file__),inventory_sha256=sha(inventory_model.__file__),
        duration_s=4.,tick_s=.1,inventory_deadband_m=.0002,
        target_offsets_m=[.0001,-.0001,.0001,.0006],ticks_per_stage=10,
        coarse=dict(max_step_s=.1,rtol=1e-7),fine=dict(max_step_s=.05,rtol=2e-8),
        in_band_evaluation_end_s=3.,maximum_neutral_exchange_kg=1e-10,minimum_legacy_exchange_kg=1e-6,
        maximum_in_band_measured_error_m=.0002,refinement_relative=1e-3,
        stationary_accepted=False,closed_tasks=[])
    if reference is not None and read(Path(reference)/'protocol.json')!=protocol:
        raise ValueError('Installed neutral study identity differs')
    atomic_json(out/'protocol.json',seal(protocol))
    legacy=family(0.,**protocol['coarse'])
    neutral=family(protocol['inventory_deadband_m'],**protocol['coarse'])
    fine=family(protocol['inventory_deadband_m'],**protocol['fine'])
    restart=FaultSession.restore(neutral[5]);prefix_exact=restart.checkpoint()==neutral[5]
    for i in range(20,40):
        result=restart.step(targets={'feed_inventory_fraction':target(neutral[1],i)})
        if result!=neutral[4][i] or capture(restart)!=neutral[2][i+1]:
            raise ValueError('Neutral restart suffix differs')
    exact=prefix_exact and restart.checkpoint()==neutral[0].checkpoint()
    excluded=set(schema(neutral[0])['accounting_indices'])
    difference=max(abs(a-b)/max(abs(b),1.) for x,y in zip(neutral[3],fine[3])
                   for j,(a,b) in enumerate(zip(x,y)) if j not in excluded)
    summaries={}
    for name,case in [('legacy',legacy),('neutral',neutral),('fine',fine)]:
        owner,_,records,states,results,_=case
        summaries[name]=summarize(records,maximum_gap_s=.1)
        atomic_json(out/(name+'-trace.json'),seal(dict(records=records,physical_states=states,results=results)))
        atomic_json(out/(name+'-summary.json'),seal(summaries[name]))
        atomic_json(out/(name+'-final.json'),owner.checkpoint())
    atomic_json(out/'midpoint.json',neutral[5]);atomic_json(out/'restart-final.json',restart.checkpoint())
    costs={name:exchange(case[2][:31]) for name,case in [('legacy',legacy),('neutral',neutral),('fine',fine)]}
    error=max(abs(result['observation']['measurements']['feed_level']['value']-target(neutral[1],i))
              for i,result in enumerate(neutral[4][:30]))
    # Last stage intentionally leaves the band: useful automatic action must return.
    active=neutral[2][-1]['stocks']['makeup']['mass_kg']<neutral[2][30]['stocks']['makeup']['mass_kg']
    audits=all(s['sampling_adequate'] and abs(s['total_mass_change_kg'])<=1e-8
        and s['maximum_total_rate_kg_s']<=1e-9 and s['maximum_local_continuity_residual_kg_s']<=1e-9
        and s['maximum_audits']['energy_relative']<=1e-6 and s['maximum_audits']['mass_relative']<=1e-10
        and s['maximum_audits']['boron_relative']<=1e-10 and not s['trip_observed'] for s in summaries.values())
    unchanged=signature()==protocol['source'] and sha(__file__)==protocol['driver_sha256'] and sha(inventory_model.__file__)==protocol['inventory_sha256']
    passed=bool(unchanged and audits and exact and active and difference<=protocol['refinement_relative']
        and error<=protocol['maximum_in_band_measured_error_m']
        and costs['legacy']>protocol['minimum_legacy_exchange_kg']
        and max(costs['neutral'],costs['fine'])<=protocol['maximum_neutral_exchange_kg'])
    report=dict(passed=passed,completed=True,source_unchanged=unchanged,restart_exact=exact,
        in_band_reserve_exchange_kg=costs,maximum_in_band_measured_error_m=error,
        out_of_band_makeup_reactivated=active,refinement_relative=difference,physical_audits_passed=audits,
        duration_s=4.,all_six_loops_automatic=True,closed_tasks=[],stationary_accepted=False,r1_accepted=False,
        accepted_scope='Four-second six-loop neutral-band response; zero reserve turnover claimed only over first three seconds')
    atomic_json(out/'report.json',seal(report))
    if reference is not None:
        expected={p.relative_to(reference) for p in Path(reference).rglob('*.json')}
        actual={p.relative_to(out) for p in out.rglob('*.json')}
        if expected!=actual or any((Path(reference)/p).read_bytes()!=(out/p).read_bytes() for p in actual):
            raise ValueError('Neutral source/installed evidence differs')
    if not passed:raise ValueError('Neutral study failed; original thresholds preserved')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reference',type=Path)
    args=parser.parse_args();print(json.dumps(run(args.output,args.reference),sort_keys=True))


if __name__=='__main__':main()
