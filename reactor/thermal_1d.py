"""
1D Thermal-Hydraulic Model with Two-Phase Flow (Boiling)
Моделирует распределение температур и паросодержание (void fraction).
"""

import torch
import torch.nn as nn

class ThermalModel1D(nn.Module):
    """
    1D тепловая модель с учетом кипения теплоносителя.
    """
    
    def __init__(self, num_nodes=50, device='cpu', dtype=torch.float64):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.num_nodes = num_nodes
        
        # === Физические параметры ===
        self.register_buffer('C_fuel', torch.tensor(4.0, device=device, dtype=dtype))
        self.register_buffer('C_clad', torch.tensor(0.4, device=device, dtype=dtype))
        self.register_buffer('C_coolant', torch.tensor(2.0, device=device, dtype=dtype))
        
        self.register_buffer('h_gap', torch.tensor(0.4, device=device, dtype=dtype))
        self.register_buffer('h_film_liquid', torch.tensor(0.6, device=device, dtype=dtype))
        self.register_buffer('h_film_boiling', torch.tensor(3.0, device=device, dtype=dtype)) # Much higher heat transfer
        
        self.register_buffer('w_Cp', torch.tensor(2.0, device=device, dtype=dtype)) 
        self.register_buffer('T_inlet', torch.tensor(280.0, device=device, dtype=dtype))
        
        # === Параметры кипения ===
        self.register_buffer('T_sat', torch.tensor(345.0, device=device, dtype=dtype)) # Saturation temp at 15.5 MPa (PWR)
        self.register_buffer('LatentHeat', torch.tensor(100.0, device=device, dtype=dtype)) # Arbitrary high value for latent heat capacity equivalent
        
        # Обратные связи
        self.register_buffer('alpha_fuel', torch.tensor(-2e-5, device=device, dtype=dtype)) 
        self.register_buffer('alpha_coolant', torch.tensor(-5e-4, device=device, dtype=dtype)) 
        self.register_buffer('alpha_void', torch.tensor(-0.01, device=device, dtype=dtype)) # Strong negative void coefficient
        
        # State
        self.T_fuel = None
        self.T_clad = None
        self.T_coolant = None # This is "bulk" temperature. If > T_sat, it represents enthalpy.
        self.void_fraction = None # 0.0 to 1.0
        self.flow_factor = 1.0
        
        self.reset()
        
    def reset(self):
        z_idx = torch.arange(self.num_nodes, device=self.device, dtype=self.dtype)
        x = z_idx / (self.num_nodes - 1)
        self.T_coolant = 310.0 - 30.0 * x 
        self.T_clad = self.T_coolant + 10.0
        self.T_fuel = self.T_clad + 100.0 
        self.void_fraction = torch.zeros(self.num_nodes, device=self.device, dtype=self.dtype)
        self.flow_factor = 1.0

    def compute_derivatives(self, T_f, T_c, T_co, power_profile):
        # Determine Heat Transfer Coefficient based on regime
        # Simplified boiling curve: if T_clad > T_sat + 5, we have boiling
        # Interpolate h between liquid and boiling
        # Simple switch for now
        
        # Check for boiling conditions (Nucleate boiling)
        is_boiling = (T_c > self.T_sat)
        h_film = torch.where(is_boiling, self.h_film_boiling, self.h_film_liquid)
        
        # 1. Fuel
        Q_gap = self.h_gap * (T_f - T_c)
        dT_f = (power_profile - Q_gap) / self.C_fuel
        
        # 2. Clad
        # Heat transfer to coolant depends on coolant state (T_co)
        # If T_co > T_sat, it's saturation temperature physically, but mathematically we store Enthalpy-like temp
        # Heat transfer is driven by (T_clad - T_coolant_bulk) usually.
        # If boiling, T_bulk ~ T_sat.
        
        T_bulk = torch.minimum(T_co, self.T_sat) # Physical temperature stops at saturation
        Q_film = h_film * (T_c - T_bulk)
        dT_c = (Q_gap - Q_film) / self.C_clad
        
        # 3. Coolant (Enthalpy transport)
        # Advection
        T_in_node = torch.roll(T_co, shifts=-1, dims=0)
        T_in_node[-1] = self.T_inlet
        
        Q_advection = self.w_Cp * self.flow_factor * (T_co - T_in_node)
        
        # If T_co > T_sat, effective heat capacity is much higher (Latent Heat)
        # Or we can just treat T_co as Enthalpy/Cp.
        # Let's keep C_coolant constant but interpret T_co > T_sat as "Quality region"
        dT_co = (Q_film - Q_advection) / self.C_coolant
        
        return dT_f, dT_c, dT_co

    def step(self, dt, power_profile_MW, flow_factor=1.0):
        self.flow_factor = flow_factor
        
        # RK4
        tf0, tc0, tco0 = self.T_fuel.clone(), self.T_clad.clone(), self.T_coolant.clone()
        
        k1_f, k1_c, k1_co = self.compute_derivatives(tf0, tc0, tco0, power_profile_MW)
        k2_f, k2_c, k2_co = self.compute_derivatives(tf0+0.5*dt*k1_f, tc0+0.5*dt*k1_c, tco0+0.5*dt*k1_co, power_profile_MW)
        k3_f, k3_c, k3_co = self.compute_derivatives(tf0+0.5*dt*k2_f, tc0+0.5*dt*k2_c, tco0+0.5*dt*k2_co, power_profile_MW)
        k4_f, k4_c, k4_co = self.compute_derivatives(tf0+dt*k3_f, tc0+dt*k3_c, tco0+dt*k3_co, power_profile_MW)
        
        self.T_fuel = tf0 + (dt/6.0)*(k1_f + 2*k2_f + 2*k3_f + k4_f)
        self.T_clad = tc0 + (dt/6.0)*(k1_c + 2*k2_c + 2*k3_c + k4_c)
        self.T_coolant = tco0 + (dt/6.0)*(k1_co + 2*k2_co + 2*k3_co + k4_co)
        
        # Calculate Void Fraction
        # If T_coolant > T_sat, the excess energy is steam quality
        # delta_T_superheat = T_coolant - T_sat
        # Quality x ~ C_p * delta_T / LatentHeat
        # Void alpha ~ x / (x + (1-x) * (rho_steam/rho_liquid))
        # Simplified: alpha ~ k * (T_coolant - T_sat)
        
        excess_temp = torch.clamp(self.T_coolant - self.T_sat, min=0.0)
        # Simple conversion model: 50 degrees over sat = 100% void (just for simulation effect)
        self.void_fraction = torch.clamp(excess_temp / 50.0, max=1.0)
        
        return self.compute_reactivity_feedback()
        
    def compute_reactivity_feedback(self):
        # Standard temp feedback
        # Use physical temperature (capped at T_sat) for liquid density feedback
        T_physical = torch.minimum(self.T_coolant, self.T_sat)
        
        d_rho_fuel = self.alpha_fuel * (self.T_fuel - 600.0)
        d_rho_coolant = self.alpha_coolant * (T_physical - 290.0)
        
        # Void feedback
        d_rho_void = self.alpha_void * self.void_fraction
        
        return d_rho_fuel + d_rho_coolant + d_rho_void

    def get_state(self):
        return {
            'fuel_temp': self.T_fuel.cpu().numpy(),
            'coolant_temp': self.T_coolant.cpu().numpy(),
            'max_fuel_temp': self.T_fuel.max().item(),
            'max_clad_temp': self.T_clad.max().item(),
            'outlet_temp': torch.minimum(self.T_coolant[0], self.T_sat).item(),
            'inlet_temp': self.T_coolant[-1].item(),
            'max_void_fraction': self.void_fraction.max().item(),
            'avg_void_fraction': self.void_fraction.mean().item()
        }
