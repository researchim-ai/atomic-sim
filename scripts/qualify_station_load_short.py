"""Short numerical qualification of load scheduling, whole-path comparison and native replay."""
import argparse
import json
from pathlib import Path
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_load import LoadCycle
from scripts.qualify_station_load_cycle import run as execute,identity
from scripts.qualify_station_inventory import read


def run(output,reference=None):
    out=Path(output).absolute()
    if any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Symlink output refused')
    out.mkdir(parents=True,exist_ok=False)
    source=identity();own_sha=sha(__file__)
    cycle=LoadCycle(tick_s=.01,duration_s=.12,points=((0.,1.),(.03,.999),(.06,1.001),(.09,1.)),settling_window_s=.02)
    protocol=dict(source=source,driver_sha256=own_sha,scope='0.12s short transactions only; no long load-cycle acceptance')
    atomic_json(out/'protocol.json',seal(protocol))
    if reference and read(Path(reference)/'protocol.json')!=protocol:raise ValueError('Short load study identity differs')
    results=[]
    for role,step,tolerance in [('coarse',1.,1e-7),('fine',.5,2e-8),('finer',.25,5e-9),('native',1.,1e-7)]:
        result=execute(out/role,cycle,max_step_s=step,rtol=tolerance,
            reference=None if role=='coarse' else out/'coarse',restart=role=='native')
        results.append(result)
    unchanged=source==identity() and own_sha==sha(__file__)
    report=dict(passed=all(r['passed'] for r in results) and unchanged,source_unchanged=unchanged,
        source=source,short_duration_s=.12,native_exact=results[-1]['native_whole_trajectory_exact'],
        maximum_refinement_relative=max(r['refinement']['maximum_relative'] for r in results[1:]),
        closed_tasks=[],r1_accepted=False,validated=False)
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Short load qualification failed')
    if reference:
        reference=Path(reference)
        names={p.relative_to(out) for p in out.rglob('*.json')}
        if names!={p.relative_to(reference) for p in reference.rglob('*.json')} or any((out/p).read_bytes()!=(reference/p).read_bytes() for p in names):
            raise ValueError('Installed short load evidence differs')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();r=run(a.output,a.reference);print(json.dumps(r,sort_keys=True))


if __name__=='__main__':main()
