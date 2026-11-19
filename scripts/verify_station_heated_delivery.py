"""Offline cold-wheel repetition of the two-constraint heated mean benchmark."""
import argparse
from pathlib import Path
from scripts.station_study_delivery import verify

FILES={'protocol.json','spatial.json','normalizations.json','external.json','report.json'}


def run(output,source,library):
    return verify(output,source,library,module='scripts.qualify_station_heated',package='atomic_model_station_heated',scientific_files=FILES,entrypoint=__file__)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','source-evidence','library-evidence'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.output,a.source_evidence,a.library_evidence)

if __name__=='__main__':main()
