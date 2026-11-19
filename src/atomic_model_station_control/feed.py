"""Six-loop hot reference: three process loops and three liquid-inventory loops."""
from atomic_model_station import PWRBypassSession
from atomic_model_station_signals import StationSignals,Sensor
from .pi import PIConfig
from .model import ControlSession,ProtectionConfig


def feed_control_session(*,tick_s=1.,seed=0,power_target_w=3.15e9,pressure_target_pa=6.3e6,protection=None,integration_step_s=1.,rtol=1e-7,tracking_tau_s=10.,inventory_deadband_m=0.):
    plant=PWRBypassSession(max_step_s=integration_step_s,rtol=rtol);snap=plant.snapshot();levels=snap['level_readouts']
    sensors=[Sensor('power','fission_power_w',lag_s=.5),Sensor('pressure','steam_pressure_pa',lag_s=.2),
        Sensor('speed','turbine_speed_rad_s',lag_s=.2),Sensor('sg_level','sg_collapsed_level_m',lag_s=1.),
        Sensor('hotwell_level','hotwell_level_m',lag_s=1.),Sensor('feed_level','feed_tank_level_m',lag_s=1.),
        Sensor('sg_outflow','sg_outflow_kg_s',lag_s=.2),Sensor('feed_flow','feed_flow_kg_s',lag_s=.2),Sensor('condensate_flow','condensate_flow_kg_s',lag_s=.2)]
    tracking=tracking_tau_s
    loops=[PIConfig('power','rod_target',power_target_w,snap['core']['fission_total_w'],kp=.01,ki=.002,direction=-1,bias=snap['controls']['rod_target'],tracking_tau_s=tracking),
        PIConfig('pressure','steam_bypass',pressure_target_pa,1e6,kp=.1,ki=.02,direction=-1,tracking_tau_s=tracking),
        PIConfig('speed','steam_admission',157.,10.,kp=.03,ki=.01,bias=1.,tracking_tau_s=tracking),
        PIConfig('sg_level','feed_flow_fraction',levels['sg_collapsed_level_m'],.1,kp=.1,ki=.005,bias=.5,rate_per_s=.1,tracking_tau_s=tracking,feedforward_sensor='sg_outflow',feedforward_reference=snap['feed_train']['feed_flow_kg_s'],feedforward_scale=snap['feed_train']['flow_scale_kg_s']),
        PIConfig('hotwell_level','condensate_flow_fraction',levels['hotwell_level_m'],.1,kp=.2,ki=.003,bias=.5,direction=-1,rate_per_s=.1,tracking_tau_s=tracking,feedforward_sensor='feed_flow',feedforward_reference=snap['feed_train']['feed_flow_kg_s'],feedforward_scale=snap['feed_train']['flow_scale_kg_s']),
        PIConfig('feed_level','feed_inventory_fraction',levels['feed_tank_level_m'],.1,kp=2.5,ki=.02,bias=.5,rate_per_s=.1,tracking_tau_s=tracking,inventory_deadband_m=inventory_deadband_m)]
    return ControlSession(plant,instruments=StationSignals(sensors,tick_s=tick_s,seed=seed),loops=loops,
        protection=protection if protection is not None else ProtectionConfig())
