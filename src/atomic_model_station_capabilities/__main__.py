"""Station passport and configuration transport / Паспорт и перенос конфигураций."""
import argparse
from pathlib import Path
import json
from dataclasses import asdict
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from atomic_model_station_env import EnvConfig
from .model import passport,dump_config,load_config,migrate_config,CONFIG_FORMAT,RAW_FORMAT


def write_new(path,value):
    path=Path(path).absolute()
    if path.exists() or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('New ordinary output file required')
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(path,value)


def read_document(path):
    with Path(path).open('rb') as stream:data=stream.read(1024*1024+1)
    if len(data)>1024*1024:raise ValueError('Configuration input exceeds 1 MiB')
    return json.loads(data.decode('utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('passport');p.add_argument('--output',type=Path)
    p=sub.add_parser('export-config');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('validate');p.add_argument('input',type=Path)
    p=sub.add_parser('migrate');p.add_argument('input',type=Path)
    p.add_argument('--from-format',choices=[RAW_FORMAT,CONFIG_FORMAT],required=True)
    p.add_argument('--allow-rebind',action='store_true');p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.command=='passport':
            result=seal(passport())
            if args.output:write_new(args.output,result)
        elif args.command=='export-config':
            result=dump_config(EnvConfig());write_new(args.output,result)
        elif args.command=='validate':
            config=load_config(read_document(args.input))
            result=dict(configuration_valid=True,simulation_executed=False,config=asdict(config))
        else:
            config,result=migrate_config(read_document(args.input),
                from_format=args.from_format,allow_rebind=args.allow_rebind)
            write_new(args.output,config)
        print(json.dumps(result,sort_keys=True,allow_nan=False))
    except (ValueError,TypeError,OSError) as error:parser.exit(2,str(error)+'\n')


if __name__=='__main__':main()
