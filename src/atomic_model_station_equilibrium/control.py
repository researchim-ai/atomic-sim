"""References derived from a declared solved operating point, before first sample."""
from dataclasses import replace


def references(loop_configs,plant):
    if plant.time!=0 or len(plant.history) or plant.station_equilibrium is None:
        raise ValueError('Solved fresh physical initialization required for PI references')
    snap=plant.snapshot();levels=snap['level_readouts'];cfg=plant.station_equilibrium['config'];flow=plant.feed_terms['feed_flow_kg_s']
    goals=dict(rod_target=cfg['power_w'],steam_bypass=cfg['pressure_pa'],steam_admission=cfg['speed_rad_s'],
        feed_flow_fraction=levels['sg_collapsed_level_m'],condensate_flow_fraction=levels['hotwell_level_m'],feed_inventory_fraction=levels['feed_tank_level_m'])
    biases=dict(rod_target=float(plant.state[12]),steam_bypass=float(plant.state[plant.bypass_index]),
        steam_admission=float(plant.controls['steam_admission']),feed_flow_fraction=float(flow/plant.feed_flow_scale_kg_s),
        condensate_flow_fraction=float(flow/plant.feed_flow_scale_kg_s),feed_inventory_fraction=.5)
    result=[]
    for c in loop_configs:
        changes=dict(target=goals[c.output],bias=biases[c.output])
        if c.feedforward_sensor is not None:changes['feedforward_reference']=float(flow)
        result.append(replace(c,**changes))
    return result
