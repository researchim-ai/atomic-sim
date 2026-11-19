"""Offline installed-wheel repetition of all forced-momentum comparisons."""
import argparse
from pathlib import Path
from scripts.station_study_delivery import verify

FILES={'protocol.json','spatial.json','dns.json','friction.json','report.json'}


def run(output,source,library):
    return verify(output,source,library,module='scripts.qualify_station_turbulence',package='atomic_model_station_turbulence',scientific_files=FILES,entrypoint=__file__)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','source-evidence','library-evidence'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.output,a.source_evidence,a.library_evidence)

if __name__=='__main__':main()
