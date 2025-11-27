import torch
import torch.nn as nn
import math

class ThermalModel1D(nn.Module):
    def __init__(self, config=None, num_nodes=50, device='cpu', dtype=torch.float64):
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device
        self.dtype = dtype
        
        # --- Physical Constants (Water properties at ~15 MPa for VVER) ---
        # Specific heat capacity (J/kg/K)
        self.cp_fuel = 300.0   # UO2
        self.cp_coolant = 5500.0 # Pressurized water (avg)
        
        # Density (kg/m^3)
        self.rho_coolant_liquid = 750.0 
        self.rho_coolant_vapor = 40.0   # Steam at high pressure
        
        # Geometry
        self.core_height = config.core_height if config else 3.53
        self.node_height = self.core_height / num_nodes
        self.total_flow_area = 4.0 # m^2 (approx for VVER-1000)
        
        # Heat transfer coefficients (W/m^2/K)
        self.h_gap = 5000.0 # Fuel-Clad gap conductance
        self.h_coolant_base = 25000.0 # Dittus-Boelter approx base
        
        # Feedback coefficients
        alpha_f = config.temp_coeff_fuel if config else -2e-5
        alpha_c = config.temp_coeff_coolant if config else -5e-4
        alpha_v = config.void_coeff if config else -0.15 
        
        self.register_buffer('alpha_fuel', torch.tensor(alpha_f, device=device, dtype=dtype))
        self.register_buffer('alpha_coolant', torch.tensor(alpha_c, device=device, dtype=dtype))
        self.register_buffer('alpha_void', torch.tensor(alpha_v, device=device, dtype=dtype))
        
        # --- State Variables ---
        # Temperatures (K) - Start at operating conditions
        self.register_buffer('temp_fuel', torch.ones(num_nodes, device=device, dtype=dtype) * 900.0)
        self.register_buffer('temp_clad', torch.ones(num_nodes, device=device, dtype=dtype) * 600.0)
        self.register_buffer('temp_coolant', torch.linspace(290+273.15, 320+273.15, num_nodes, device=device, dtype=dtype))
        
        # Flow Dynamics State
        # Mass flow rate (kg/s) - Nominal ~18000 kg/s for VVER-1000
        self.nominal_flow = 18000.0
        self.register_buffer('mass_flow', torch.tensor(self.nominal_flow, device=device, dtype=dtype))
        
        # Pump State (0.0 to 1.0)
        self.register_buffer('pump_speed', torch.tensor(1.0, device=device, dtype=dtype))
        
        # Boiling & Void
        self.t_sat = 345.0 + 273.15 # Saturation temp at 15.7 MPa
        self.register_buffer('void_fraction', torch.zeros(num_nodes, device=device, dtype=dtype))

        # Hydraulic Parameters
        self.friction_coeff = 100.0 # High friction at nominal flow (pressure drop)
        self.inertia_coeff = 50.0 # Moderate inertia
        self.buoyancy_coeff = 150.0 # Sufficient to drive ~10% flow, but negligible at 100% flow

    def get_reactivity_feedback(self):
        # Doppler (Fuel Temp)
        delta_tf = self.temp_fuel - 900.0
        rho_fuel = self.alpha_fuel * delta_tf
        
        # Moderator Temp
        delta_tc = self.temp_coolant - (305.0 + 273.15)
        rho_coolant = self.alpha_coolant * delta_tc
        
        # Void (Boiling)
        rho_void = self.alpha_void * self.void_fraction
        
        return rho_fuel + rho_coolant + rho_void

    def compute_hydraulics(self, dt):
        """
        Simplified 1D Momentum Equation:
        d(Flow)/dt = (Pump_Head + Buoyancy_Head - Friction_Head) / Inertia
        """
        # 1. Buoyancy Head (Natural Circulation driver)
        # Proportional to integral of (rho_cold - rho_local) * g * dz
        # We approximate it by the average temperature difference from inlet
        avg_temp = torch.mean(self.temp_coolant)
        inlet_temp = self.temp_coolant[0]
        density_diff_factor = (avg_temp - inlet_temp) / inlet_temp
        buoyancy_force = self.buoyancy_coeff * density_diff_factor * 10000.0 # Scaling factor
        
        # 2. Pump Head
        # Proportional to square of pump speed
        pump_force = (self.pump_speed * self.nominal_flow) * 1.5 # Target force to maintain flow
        if self.pump_speed < 0.1: pump_force = 0.0 # Pump off
        
        # 3. Friction Head (Resistance)
        # Proportional to flow^2
        flow_abs = torch.abs(self.mass_flow)
        friction_force = self.friction_coeff * self.mass_flow * flow_abs / self.nominal_flow
        
        # 4. Momentum Balance
        net_force = pump_force + buoyancy_force - friction_force
        # DEBUG: Print forces if flow is unexpectedly zero
        if self.mass_flow.item() < 1.0:
             # Only print if we expect circulation (buoyancy is significant)
             if buoyancy_force.item() > 1.0:
                 print(f"DEBUG TH: Flow={self.mass_flow.item():.3f} Pump={pump_force.item():.3f} Buoy={buoyancy_force.item():.3f} Fric={friction_force.item():.3f} Net={net_force.item():.3f}")
        
        d_flow = (net_force / self.inertia_coeff) * dt
        
        new_flow = self.mass_flow + d_flow
        # Prevent reverse flow for this simplified model (optional, but safer numerically)
        new_flow = torch.clamp(new_flow, min=0.0)
        
        self.mass_flow.copy_(new_flow)
        
        return new_flow

    def forward(self, power_distribution, dt, inlet_temp_k=None):
        """
        power_distribution: Normalized power shape (sum=1 if treated as probability, but here it's relative)
                            We assume input is in Watts per node.
        inlet_temp_k: Inlet coolant temperature in Kelvin (optional, defaults to 280C)
        """
        # Update Flow Dynamics first
        current_flow = self.compute_hydraulics(dt)
        
        # Convert mass flow to velocity for advection: u = m_dot / (rho * Area)
        # Simplified: u propto mass_flow
        # Nominal velocity ~ 4-5 m/s
        velocity = (current_flow / self.nominal_flow) * 5.0
        velocity = torch.clamp(velocity, min=0.05) # Min velocity for numeric stability (conduction only limit)

        # --- Inlet Boundary Condition ---
        if inlet_temp_k is not None:
            if isinstance(inlet_temp_k, float):
                 val = torch.tensor(inlet_temp_k, device=self.device, dtype=self.dtype)
            else:
                 val = inlet_temp_k
            self.temp_coolant[0] = val
        else:
            self.temp_coolant[0] = 280.0 + 273.15
            
        # --- Heat Transfer ---
        # 1. Fuel to Clad
        q_fuel_gen = power_distribution # Watts generated in fuel
        # Q_gap = h_gap * Area * (T_f - T_c)
        # Simplified lumped parameter approach per node
        q_trans_fuel_clad = self.h_gap * (self.temp_fuel - self.temp_clad)
        
        # Fuel Energy Balance: d(rho*cp*V*T)/dt = Q_gen - Q_out
        # Simplified thermal mass constants
        mass_fuel_node = 500.0 # kg approx per node slice
        mass_clad_node = 100.0
        mass_coolant_node = 200.0 
        
        dT_fuel = (q_fuel_gen - q_trans_fuel_clad) / (mass_fuel_node * self.cp_fuel) * dt
        self.temp_fuel += dT_fuel
        
        # 2. Clad to Coolant
        # Heat transfer coeff depends on flow (Dittus-Boelter) and Boiling
        reynolds_factor = (current_flow / self.nominal_flow)**0.8
        h_coolant = self.h_coolant_base * reynolds_factor
        
        # Check for boiling
        is_boiling = self.temp_clad > self.t_sat
        
        # Nucleate boiling enhancement (simplified Thom correlation logic)
        h_boiling_factor = torch.where(is_boiling, torch.tensor(5.0, device=self.device, dtype=self.dtype), torch.tensor(1.0, device=self.device, dtype=self.dtype))
        h_coolant = h_coolant * h_boiling_factor
        
        q_trans_clad_coolant = h_coolant * (self.temp_clad - self.temp_coolant)
        
        dT_clad = (q_trans_fuel_clad - q_trans_clad_coolant) / (mass_clad_node * 500.0) * dt # cp_clad approx 500
        self.temp_clad += dT_clad
        
        # 3. Coolant Advection + Heating
        # Energy equation: dT/dt + u * dT/dz = Q_in / (rho*cp)
        
        # Heating source term
        source_term = q_trans_clad_coolant / (mass_coolant_node * self.cp_coolant)
        
        # Advection term (Upwind scheme)
        # T[i] new = T[i] - u * dt/dz * (T[i] - T[i-1])
        dz = self.node_height
        courant = velocity * dt / dz
        
        # Shift array for advection
        # Use current T[0] as previous for T[1]
        temp_prev = torch.cat([self.temp_coolant[0].unsqueeze(0), self.temp_coolant[:-1]])
        advection = (velocity / dz) * (self.temp_coolant - temp_prev)
        
        dT_coolant = (source_term - advection) * dt
        
        # Apply update (skip index 0 as it is fixed BC)
        self.temp_coolant[1:] += dT_coolant[1:]
        
        # --- Update Void Fraction ---
        # If T > T_sat, quality x > 0
        # Simple homogeneous equilibrium model for now
        enthalpy_excess = (self.temp_coolant - self.t_sat) * self.cp_coolant
        latent_heat = 1.5e6 # J/kg approx
        
        quality = torch.clamp(enthalpy_excess / latent_heat, min=0.0, max=1.0)
        
        # Void fraction (alpha) from quality (x)
        # alpha = 1 / (1 + ((1-x)/x) * (rho_v/rho_l) * S) 
        # Simplified linear relationship for low quality region
        self.void_fraction = torch.clamp(quality * 10.0, max=1.0) 
        
        # If boiling, clamp temp to Tsat for energy balance (latent heat taken by void gen)
        # In real code, enthalpy is state variable. Here we mimic it.
        # But allow slight superheat for numerics
        self.temp_coolant = torch.min(self.temp_coolant, torch.tensor(self.t_sat + 5.0, device=self.device, dtype=self.dtype))
        
        return self.temp_fuel, self.temp_coolant
