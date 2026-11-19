"""Strict historical-source guard with one explicit, evidenced cache migration.

This does not relabel a historical qualification as a new execution.
"""
from pathlib import Path
from atomic_model_training.spec import read_json
from atomic_model_multiphysics.provenance import unseal,sha

ROOT=Path(__file__).resolve().parents[1]
OLD_WATER='8e85c45065fd6766bbd5a519608ddfe72c0e6aaf29277ba6408b0001dda625fa'
# Exact byte identity from the accepted v104 implementation (not a wildcard).
CACHED_WATER='ac2bc75496cdcd9b88656e9524637b6947a1d084f991c0db020c2a522a2dd574'


def assert_source_compatible(published,current):
    migrations=[]
    def compare(a,b,path=()):
        if a==b:return
        if type(a) is dict and type(b) is dict:
            assert set(a)==set(b),f'Historical source keys changed at {path}'
            for key in sorted(a):compare(a[key],b[key],(*path,key))
            return
        assert path[-3:]==('plant','files','water.py') and a==OLD_WATER and b==CACHED_WATER, f'Unapproved historical source change at {path}'
        migrations.append('/'.join(path))
    compare(published,current)
    if migrations:
        proof=unseal(read_json(ROOT/'docs/benchmarks/station_water_cache_v104.json'))
        assert proof['format']=='station-exact-water-cache/v104'
        r=proof['report']
        assert r['passed'] is True and r['accepted']==r['attempts']==6 and r['failed']==0
        assert len(proof['attempts'])==6 and all(x['checks']['exact_scientific'] is True for x in proof['attempts'])
        def leaves(v):
            if type(v) is dict:
                for k,x in v.items():
                    if k=='water.py':yield x
                    else:yield from leaves(x)
        assert set(leaves(r['source']))=={CACHED_WATER}
        from atomic_model_plant_extension import water
        assert sha(water.__file__)==CACHED_WATER
        from scripts.verify_station_install import accept_nominal
        assert accept_nominal(read_json(ROOT/'docs/benchmarks/station_nominal_requalification_v105.json'))['reused_existing_execution'] is True
    return migrations
