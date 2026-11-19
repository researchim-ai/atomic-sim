"""Run a small station tutorial / Запустить короткий пример станции."""
import argparse
from pathlib import Path
from .model import NAMES,run_example


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example',choices=NAMES)
    parser.add_argument('--output',type=Path,required=True,
        help='New result directory / Новая директория результатов')
    args=parser.parse_args();result=run_example(args.example,args.output)
    print(dict(example=result['example'],passed=result['passed'],simulated_s=result['simulated_s']))


if __name__=='__main__':main()
