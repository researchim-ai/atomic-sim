"""Short shutdown lifecycle/native proof; never full thermal acceptance."""
import argparse
from pathlib import Path
from atomic_model_station_shutdown import ShutdownCase
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts import qualify_station_shutdown as study
from scripts.qualify_station_inventory import read


def run(output,reference=None):
    out=Path(output).absolute()
    if any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Symlink output refused')
    protocol=dict(source=study.identity(),driver_sha256=sha(__file__),duration_s=.04,scope='Short lifecycle, not a full shutdown')
    if reference and read(Path(reference)/'protocol.json')!=protocol:raise ValueError('Shutdown short identity')
    out.mkdir(parents=True,exist_ok=False);atomic_json(out/'protocol.json',seal(protocol))
    case=ShutdownCase(tick_s=.01,duration_s=.04)
    normal=study.run(out/'normal',case,max_step_s=.1)
    native=study.run(out/'native',case,max_step_s=.1,restart=True,reference=out/'normal')
    coast=study.run(out/'coastdown',ShutdownCase('coastdown',tick_s=.01,duration_s=.04),max_step_s=.1)
    report=dict(passed=normal['passed'] and native['passed'] and coast['passed'],source=study.identity(),
        native_exact=native['native_whole_path_exact'],closed_tasks=[],npp_11_03_accepted=False,r1_accepted=False,duration_s=.04)
    atomic_json(out/'report.json',seal(report))
    if not report['passed'] or protocol['source']!=study.identity():raise ValueError('Shutdown short proof failed')
    if reference:
        expected={p.relative_to(reference) for p in Path(reference).rglob('*.json')};actual={p.relative_to(out) for p in out.rglob('*.json')}
        if actual!=expected or any((out/p).read_bytes()!=(Path(reference)/p).read_bytes() for p in actual):raise ValueError('Installed short shutdown JSON differs')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();run(a.output,a.reference)


if __name__=='__main__':main()
