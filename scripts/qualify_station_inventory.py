"""Prospective short water-transfer diagnostics; not stationarity qualification."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from atomic_model_multiphysics.provenance import seal, unseal, sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_campaign.engine import signature
from atomic_model_station_faults import FaultSession
from atomic_model_station_inventory import capture, summarize, archive_summary
from atomic_model_station_inventory import model
from atomic_model_station_readiness import schema


def read(path):
    return unseal(json.loads(Path(path).read_text()))


def command(index):
    # Drain, stop, makeup, stop; no reset or hidden stock reassignment.
    return dict(feed_inventory_control=True, feed_inventory_fraction=(.25,.5,.75,.5)[index//10])


def family(max_step_s, rtol):
    owner=FaultSession(tick_s=.05,max_step_s=max_step_s,rtol=rtol)
    rows=[capture(owner)]; states=[]; midpoint=None; saved=None
    for i in range(40):
        owner.step(command(i)); rows.append(capture(owner))
        states.append(dict(time_s=float(owner.control.plant.time),physical_state=owner.control.plant.state.tolist()))
        if i==19:
            midpoint=owner.checkpoint(); saved=deepcopy(owner)
    return owner,rows,states,midpoint,saved


def run(output, reference=None):
    out=Path(output).absolute()
    if any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('Symlink output refused')
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(format='station-inventory-study/v1',source=signature(),driver_sha256=sha(__file__),
                  inventory_sha256=sha(model.__file__),duration_s=2.,tick_s=.05,
                  coarse=dict(max_step_s=.05,rtol=1e-7),fine=dict(max_step_s=.025,rtol=2e-8),
                  maximum_local_continuity_residual_kg_s=1e-9,maximum_total_rate_kg_s=1e-9,
                  maximum_total_mass_change_kg=1e-8,refinement_relative=1e-3,
                  stationary_accepted=False,closed_tasks=[])
    if reference is not None and read(Path(reference)/'protocol.json')!=protocol:
        raise ValueError('Installed diagnostic study identity differs')
    atomic_json(out/'protocol.json',seal(protocol))
    coarse, rows, states, midpoint, saved=family(**protocol['coarse'])
    fine, fine_rows, fine_states, _, _=family(**protocol['fine'])
    restart=FaultSession.restore(midpoint)
    prefix_exact=restart.checkpoint()==saved.checkpoint()
    for i in range(20,40):
        restart.step(command(i))
        if capture(restart)!=rows[i+1]:
            raise ValueError('Diagnostic restart suffix differs')
    restart_exact=prefix_exact and restart.checkpoint()==coarse.checkpoint()
    excluded=set(schema(coarse)['accounting_indices'])
    difference=max(abs(a-b)/max(abs(b),1.) for c,f in zip(states,fine_states)
                   for index,(a,b) in enumerate(zip(c['physical_state'],f['physical_state'])) if index not in excluded)
    summaries={}
    for key, records in [('coarse',rows),('fine',fine_rows)]:
        summary=summarize(records,requested_horizon_s=3600.,maximum_gap_s=.05)
        summaries[key]=summary
        atomic_json(out/(key+'-trace.json'),seal(dict(records=records)))
        atomic_json(out/(key+'-summary.json'),seal(summary))
    archive=archive_summary(states,coarse,window_s=1.)
    atomic_json(out/'archive-trends.json',seal(archive))
    for name,obj in [('midpoint',midpoint),('coarse-final',coarse.checkpoint()),('fine-final',fine.checkpoint()),('restart-final',restart.checkpoint())]:
        atomic_json(out/(name+'.json'),obj)
    audits=all(s['sampling_adequate'] and abs(s['total_mass_change_kg'])<=protocol['maximum_total_mass_change_kg']
               and s['maximum_total_rate_kg_s']<=protocol['maximum_total_rate_kg_s']
               and s['maximum_local_continuity_residual_kg_s']<=protocol['maximum_local_continuity_residual_kg_s']
               and s['maximum_audits']['energy_relative']<=1e-6 and s['maximum_audits']['mass_relative']<=1e-10
               and s['maximum_audits']['boron_relative']<=1e-10 and not s['trip_observed'] for s in summaries.values())
    stocks={r['name']:r for r in summaries['coarse']['stocks']}
    directions=stocks['drain']['change_kg']>0 and stocks['makeup']['change_kg']<0
    unchanged=signature()==protocol['source'] and sha(__file__)==protocol['driver_sha256'] and sha(model.__file__)==protocol['inventory_sha256']
    report=dict(passed=bool(audits and directions and restart_exact and unchanged and difference<=protocol['refinement_relative']),
                source_unchanged=unchanged,restart_exact=restart_exact,refinement_relative=difference,
                local_continuity_accepted=audits,reserve_directions_correct=directions,
                completed=True,duration_s=2.,closed_tasks=[],stationary_accepted=False,r1_accepted=False,
                accepted_scope='Short signed transfers, local mass equations, finite reserves and diagnostic replay only')
    atomic_json(out/'report.json',seal(report))
    if reference is not None:
        expected={p.relative_to(reference) for p in Path(reference).rglob('*.json')}
        actual={p.relative_to(out) for p in out.rglob('*.json')}
        if expected!=actual or any((Path(reference)/p).read_bytes()!=(out/p).read_bytes() for p in actual):
            raise ValueError('Source/installed diagnostic evidence differs')
    if not report['passed']:
        raise ValueError('Inventory diagnostic qualification failed; inspect preserved evidence')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.output,args.reference),sort_keys=True))


if __name__=='__main__':
    main()
