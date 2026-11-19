"""Offline cold-wheel repeat of corrected quadrature and full profile matrix."""
import argparse
from pathlib import Path
from scripts.station_study_delivery import verify

FILES={'protocol.json','moments.json','comparison.json','refinement.json','report.json'}


def run(output,source,library):
    return verify(output,source,library,module='scripts.qualify_station_quadrature',package='atomic_model_station_quadrature',scientific_files=FILES,entrypoint=__file__)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','source-evidence','library-evidence'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.output,a.source_evidence,a.library_evidence)

if __name__=='__main__':main()
