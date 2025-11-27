
from dataclasses import dataclass
from typing import Dict

@dataclass
class ReactorConfig:
    # Meta
    name: str
    type_str: str
    
    # Geometry & Specs
    thermal_power: float  # MW
    core_height: float    # meters
    num_assemblies: int
    
    # Neutronics Parameters
    beta_eff: float       # Delayed neutron fraction
    neutron_speed: float  # v [cm/s]
    
    # Reactivity Coefficients (Feedback)
    # Units: delta_k/k per unit
    temp_coeff_fuel: float     # Doppler effect (per deg C)
    temp_coeff_coolant: float  # Moderator temp effect (per deg C)
    void_coeff: float          # Void coeff (per fraction 0..1) - CRITICAL SAFETY PARAM
    
    # Control System
    scram_speed: float         # Control rod drop speed [fraction/sec]
    shim_speed: float          # Normal control speed [fraction/sec]

# === VVER-1000 (Similar to PWR) ===
# Inherently safe: Negative void coefficient.
VVER_1000 = ReactorConfig(
    name="VVER-1000",
    type_str="PWR (Pressurized Water Reactor)",
    
    thermal_power=3000.0,
    core_height=3.53,
    num_assemblies=163,
    
    beta_eff=0.0065,
    neutron_speed=1000.0, # Scaled for numerical stability in explicit solver
    
    temp_coeff_fuel=-2.4e-5,    # Strong Doppler safety
    temp_coeff_coolant=-5.0e-4, # Strong negative moderator feedback
    void_coeff=-0.15,           # NEGATIVE: Boiling -> Power Drop (Safe)
    
    scram_speed=0.25,           # Fast drop (gravity) ~4 sec full insertion
    shim_speed=0.01
)

# === RBMK-1000 (Channel Type) ===
# Unstable at low power: Positive void coefficient.
# Graphite moderated, water cooled.
RBMK_1000 = ReactorConfig(
    name="RBMK-1000",
    type_str="LWGR (Light Water Graphite Reactor)",
    
    thermal_power=3200.0,
    core_height=7.0,      # Very tall core -> susceptible to xenon oscillations
    num_assemblies=1661,
    
    beta_eff=0.0065,
    neutron_speed=1000.0,
    
    temp_coeff_fuel=-1.2e-5,    # Weaker Doppler than VVER
    temp_coeff_coolant=5.0e-5,  # Slightly positive
    void_coeff=0.02,            # POSITIVE: Boiling -> Power Rise (Danger!)
    
    scram_speed=0.05,           # Slow insertion (old design) ~18-20 sec
    shim_speed=0.005
)

# Registry
REACTORS: Dict[str, ReactorConfig] = {
    "vver": VVER_1000,
    "rbmk": RBMK_1000
}

