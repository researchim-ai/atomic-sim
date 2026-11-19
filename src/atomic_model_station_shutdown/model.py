"""Research shutdown invariants; stopped fission does not mean zero decay heat."""
from dataclasses import dataclass
import math
from atomic_model_station_load.model import finite

HORIZONS={'scram':600.,'coastdown':1800.}


@dataclass(frozen=True)
class ShutdownCase:
    family:str='scram'
    tick_s:float=5.
    duration_s:float|None=None
    dhr_requested:bool=True
    film_model:str="dittus_boelter"

    def __post_init__(self):
        from atomic_model_station.film_correlations import validate_model
        validate_model(self.film_model)
        if type(self.family) is not str or self.family not in HORIZONS:raise ValueError('Shutdown family')
        if type(self.dhr_requested) is not bool:raise ValueError('Boolean DHR request')
        if self.duration_s is None:object.__setattr__(self,'duration_s',HORIZONS[self.family])
        if not finite(self.tick_s) or not .01<=self.tick_s<=60:raise ValueError('Shutdown tick')
        if not finite(self.duration_s) or not self.tick_s<=self.duration_s<=HORIZONS[self.family]:raise ValueError('Shutdown horizon')
        ticks=self.duration_s/self.tick_s
        if abs(ticks-round(ticks))>1e-9 or not 1<=round(ticks)<=8192:raise ValueError('Integral bounded shutdown ticks')

    @property
    def ticks(self):return round(self.duration_s/self.tick_s)

    @property
    def full_horizon(self):return self.duration_s==HORIZONS[self.family]

    def command(self):
        p=dict(scram=True,turbine_trip=True,dhr_requested=self.dhr_requested)
        if self.family=='coastdown':p['primary_pump_on']=False
        return p


def capture(owner,result=None):
    plant=owner.control.plant;p=plant.snapshot()
    keys=('time_s','powers','core','electrical','primary','actuator_status','steam_header','feed_train',
          'exact_mode_reduction','energy_relative','mass_relative','boron_relative','gross_generated_j','own_needs_j','net_generated_j',
          'net_exported_j','backup_used_j','dhr_transferred_j','film','primary_pump_speed_rad_s','turbine_speed_rad_s')
    return dict(physical={k:p[k] for k in keys},state=plant.state.tolist(),result=result)


def assess(initial,records,case):
    if type(case) is not ShutdownCase:raise ValueError('ShutdownCase required')
    reasons=set();maximum={k:0. for k in ('energy_relative','mass_relative','boron_relative')}
    if len(records)!=case.ticks:reasons.add('incomplete_horizon')
    start=initial['physical'];previous_speed=start['primary_pump_speed_rad_s']
    arrival=(1-initial['state'][12])/start['actuator_status']['rod_drive']['scram_speed_s']
    for i,row in enumerate(records):
        p=row['physical'];receipt=row['result']['receipt'];act=p['actuator_status'];electrical=p['electrical']
        if (not finite(p['time_s']) or abs(p['time_s']-(i+1)*case.tick_s)>1e-8
            or not finite(receipt['time_s']) or abs(receipt['time_s']-i*case.tick_s)>1e-8):reasons.add('clock')
        if len(row['state'])!=len(initial['state']) or any(not finite(x) for x in row['state']):reasons.add('state')
        if not receipt['trip_latched'] or not act['scram_latched'] or not act['turbine_trip_latched'] or act['effective_generator_connected']:reasons.add('trip_priority')
        if p['time_s']>=arrival and act['rod_position']!=1.:reasons.add('rod_insertion')
        for k in maximum:
            v=p[k]
            if not finite(v) or v<0:reasons.add('invalid_budget')
            else:maximum[k]=max(maximum[k],v)
        if not electrical['grid_available'] or electrical['dhr_from_backup'] or p['backup_used_j']!=start['backup_used_j']:reasons.add('supply_scope')
        if electrical['dhr_active']!=case.dhr_requested:reasons.add('dhr_activation')
        if (not finite(electrical['energy_transfer_residual_w']) or abs(electrical['energy_transfer_residual_w'])>1e-5
            or not finite(electrical['dhr_entropy_generation_w_k']) or electrical['dhr_entropy_generation_w_k']< -1e-6):reasons.add('dhr_balance_or_entropy')
        if not case.dhr_requested and p['powers']['dhr_heat_w']!=0:reasons.add('unrequested_heat')
        heat=[p['powers']['dhr_heat_w'],p['powers']['nuclear_heat_w'],p['core']['fission_total_w']]
        if any(not finite(v) or v<0 for v in heat):reasons.add('invalid_heat')
        if p['feed_train']['makeup_flow_kg_s']!=0 or p['feed_train']['drain_flow_kg_s']!=0:reasons.add('reserve_exchange')
        if case.family=='coastdown':
            speed=p['primary_pump_speed_rad_s']
            if not finite(speed) or not 0<speed<=previous_speed+1e-9 or p['primary']['effective_motor_on']:reasons.add('coastdown')
            previous_speed=speed
        if not finite(p['primary']['momentum_residual_pa']) or abs(p['primary']['momentum_residual_pa'])>1e-5:reasons.add('primary_momentum')
    if any(v>(1e-6 if k=='energy_relative' else 1e-10) for k,v in maximum.items()):reasons.add('budget')
    final=records[-1]['physical'] if records else start
    fission=final['core']['fission_total_w']
    fraction=fission/max(start['core']['fission_total_w'],1.) if finite(fission) else None
    response=bool(fraction is not None and fraction<=1e-3 and finite(final['powers']['nuclear_heat_w']) and final['powers']['nuclear_heat_w']>0)
    if case.full_horizon and not response:reasons.add('shutdown_response')
    dhr=final['dhr_transferred_j']-start['dhr_transferred_j']
    if not finite(dhr):dhr=None;reasons.add('invalid_dhr_energy')
    if records and case.dhr_requested and (dhr is None or dhr<=0):reasons.add('no_dhr_energy')
    return dict(passed=not reasons,reasons=sorted(reasons),completed_ticks=len(records),full_horizon=case.full_horizon and len(records)==case.ticks,required_horizon_configured=case.full_horizon,
        maxima=maximum,fission_fraction=fraction,shutdown_response_reached=response,dhr_transferred_j=dhr,
        final_primary_speed_rad_s=final['primary_pump_speed_rad_s'],closed_tasks=[],npp_11_03_accepted=False,r1_accepted=False,
        scope='Grid-backed DHR and finite water stocks; numerical refinement, native replay and installed qualification are separate')
