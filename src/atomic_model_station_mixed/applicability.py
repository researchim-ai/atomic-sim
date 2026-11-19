"""Reported experimental boxes from Motegi et al.2024, Table2.

https://onlinelibrary.wiley.com/doi/10.1155/2024/6029412
The paper evaluates opposing (downward heated) flow. Passing a box only makes
comparison to those experiments eligible, never a station/plant qualification.
Re, Pr and Ra MUST use the reference temperatures required by each model.
"""
from math import isfinite

SOURCE = 'https://onlinelibrary.wiley.com/doi/10.1155/2024/6029412'


def assess_opposing(*, direction, reynolds_bulk, prandtl_bulk, reynolds_film,
                    prandtl_film, grashof_film, rayleigh_film):
    if type(direction) is not str or direction not in ('upward', 'downward'):
        raise ValueError('Explicit heated vertical direction required')
    inputs = dict(reynolds_bulk=reynolds_bulk,prandtl_bulk=prandtl_bulk,
                  reynolds_film=reynolds_film,prandtl_film=prandtl_film,
                  grashof_film=grashof_film,rayleigh_film=rayleigh_film)
    if any(type(v) not in (int,float) or not isfinite(v) or v <= 0 for v in inputs.values()):
        raise ValueError('Positive finite groups at declared reference temperatures required')
    # Strict inequalities as reported in Table2, not enlarged extrapolation bounds.
    boxes = {
        'churchill': dict(reynolds_film=(6105.,66835.),grashof_film=(18.89e6,22.52e6),prandtl_film=(1.75,2.09)),
        'jackson_fewster': dict(reynolds_bulk=(1000.,40000.),prandtl_bulk=(2.5,7.)),
        'aicher_martin': dict(reynolds_bulk=(3000.,120000.),rayleigh_film=(3e7,1e9),prandtl_bulk=(.7,5.)),
    }
    result = {}
    for name,box in boxes.items():
        checks = {key:low < inputs[key] < high for key,(low,high) in box.items()}
        checks['opposing_direction'] = direction == 'downward'
        result[name] = dict(eligible_for_reported_box_comparison=all(checks.values()),checks=checks,
                           failed_checks=sorted(key for key,value in checks.items() if not value),
                           reported_box={k:list(v) for k,v in box.items()},station_usable=False,validated=False,
                           source=SOURCE,source_table=2,
                           caveat='Reported marginal ranges are not a rectangular experimental coverage guarantee; geometry, BC, development, uncertainty remain to qualify')
    return result
