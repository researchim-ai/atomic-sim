"""Geometric inventory readouts; no additional inventory or interface dynamics."""
import math


def finite(value):
    if type(value) not in (int,float):return False
    try:return math.isfinite(value)
    except OverflowError:return False


def liquid_level(mass_kg,liquid_density_kg_m3,area_m2):
    values=(mass_kg,liquid_density_kg_m3,area_m2)
    if any(not finite(v) for v in values) or mass_kg<0 or liquid_density_kg_m3<=0 or area_m2<=0:
        raise ValueError('Liquid level geometry/domain')
    result=mass_kg/liquid_density_kg_m3/area_m2
    if not finite(result):raise ValueError('Liquid level finite arithmetic domain')
    return result


def collapsed_level(mass_kg,quality,liquid_density_kg_m3,area_m2,volume_m3):
    if not finite(quality) or not 0<=quality<=1:raise ValueError('Collapsed level quality')
    if not finite(volume_m3) or volume_m3<=0:raise ValueError('Collapsed level volume')
    level=liquid_level(mass_kg,liquid_density_kg_m3,area_m2)*(1-quality)
    if level*area_m2>volume_m3*(1+1e-10):raise ValueError('Liquid volume exceeds declared vessel')
    return level
