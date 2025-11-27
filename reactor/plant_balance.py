import torch
import math

class PlantBalance:
    """
    Модель баланса станции (Balance of Plant - BOP).
    Включает Парогенератор (ПГ), Турбину и Питательную систему.
    
    Связывает Первый контур (Реактор) и Второй контур (Турбина).
    """
    def __init__(self, config=None, dt=0.05):
        self.dt = dt
        
        # === Steam Generator (SG) ===
        # U-tube SG typical for VVER-1000
        # Primary side: Hot water inside tubes.
        # Secondary side: Boiling water outside tubes.
        
        # State
        self.sg_pressure = 6.4e6 # Pa (64 atm) - Nominal for VVER
        self.sg_level = 0.0 # m (Deviation from nominal), range -1.0 to +1.0
        self.sg_mass = 50000.0 # kg (Water inventory)
        self.steam_flow = 0.0 # kg/s
        
        # Thermo properties
        self.tsat_secondary = 280.0 # C (Saturation temp at 64 atm)
        self.cp_water = 4200.0
        self.h_vaporization = 1.5e6 # J/kg (approx latent heat)
        
        # Heat Transfer
        # UA (Overall Heat Transfer Coeff) ~ 50-100 MW/K total
        self.ua_sg = 150.0 * 1e6 # W/K boosted for efficient heat removal
        
        # === Turbine & Generator ===
        self.turbine_throttle = 1.0 # 0..1 (Governor valve)
        self.electric_power = 1000.0 # MWe
        self.grid_load = 1000.0 # Target MWe
        
        # === Feedwater System ===
        self.feedwater_flow = 1600.0 # kg/s (Nominal)
        self.feedwater_temp = 220.0 # C (Preheated)
        self.feedwater_pump_speed = 1.0 # 0..1
        
        # PID for SG Level Control
        self.level_pid_integral = 0.0
        self.auto_level = True
        
        # PID Parameters
        self.level_pid_kp = 2000.0 
        self.level_pid_ki = 100.0
        self.level_pid_kd = 10.0

    def step(self, primary_inlet_temp, primary_flow_rate, dt):
        """
        primary_inlet_temp: Hot leg temp form reactor (C)
        primary_flow_rate: Coolant flow (kg/s)
        """
        self.dt = dt
        
        # 1. Heat Transfer Primary -> Secondary
        # Q = UA * LMTD (Log Mean Temp Diff)
        # Simplified: Q = UA * (T_prim_avg - T_sec_sat)
        
        # T_prim_out is what we return to reactor (Cold Leg)
        # Energy balance on primary side slice in SG:
        # C_p * m_dot * (T_in - T_out) = Q
        # And Q = UA * ((T_in + T_out)/2 - T_sat)
        
        # Solving for T_out (Cold Leg Temp):
        # Let A = 2 * m_dot * Cp
        # Let B = UA
        # A * (Tin - Tout) = B * (Tin + Tout - 2*Tsat)
        # A*Tin - A*Tout = B*Tin + B*Tout - 2*B*Tsat
        # Tout * (A + B) = Tin*(A - B) + 2*B*Tsat
        # Tout = (Tin*(A-B) + 2*B*Tsat) / (A+B)
        
        A = 2 * primary_flow_rate * self.cp_water
        B = self.ua_sg
        
        # Safety clamp for low flow
        if A < B: A = B * 1.1 
        
        t_sat = self.get_saturation_temp(self.sg_pressure)
        self.tsat_secondary = t_sat
        
        t_out_cold_leg = (primary_inlet_temp * (A - B) + 2 * B * t_sat) / (A + B)
        
        # Heat extracted from Primary (Watts)
        q_trans = primary_flow_rate * self.cp_water * (primary_inlet_temp - t_out_cold_leg)
        
        # 2. Steam Generation (Secondary Side)
        # Energy added to secondary boils water
        # Q_trans = m_steam * h_vap + m_heatup * cp * dT
        # Simplified: All Q goes to boiling (assuming FW is close to sat or accounting for enthalpy)
        
        # Enthalpy diff Feed -> Steam
        # h_feed approx 900 kJ/kg (220C)
        # h_steam approx 2800 kJ/kg
        delta_h = 1.9e6 # J/kg
        
        steam_generated = q_trans / delta_h
        
        # 3. Turbine Consumption
        # Flow depends on Pressure and Valve Position
        # m_steam = K * Throttle * sqrt(P_sg - P_condenser)
        # Assuming P_condenser ~ 0
        nominal_steam_flow = 1600.0 # kg/s for 1000 MW
        self.steam_flow = nominal_steam_flow * self.turbine_throttle * (self.sg_pressure / 6.4e6)
        
        # Electric Power
        # Efficiency ~ 33%
        thermal_to_electric = 0.33
        self.electric_power = (self.steam_flow * delta_h) * thermal_to_electric / 1e6 # MW
        
        # 4. Mass Balance (SG Level)
        # dM/dt = m_feed - m_steam
        
        # Auto Feedwater Control (maintain level 0.0)
        if self.auto_level:
            level_error = 0.0 - self.sg_level
            # Feed forward: Match steam flow
            # PI controller
            self.level_pid_integral += level_error * dt
            kp = self.level_pid_kp
            ki = self.level_pid_ki
            # DEBUG: Print controller internal state
            # print(f"DEBUG BOP: Lvl={self.sg_level:.3f} Err={level_error:.3f} Int={self.level_pid_integral:.3f} Steam={self.steam_flow:.3f} Target={target_flow:.3f} FW={self.feedwater_flow:.3f}")
            
            target_flow = self.steam_flow + kp * level_error + ki * self.level_pid_integral
            
            # Pump limitations
            max_pump_flow = 2000.0 * self.feedwater_pump_speed
            self.feedwater_flow = max(0.0, min(max_pump_flow, target_flow))
            
        mass_delta = (self.feedwater_flow - self.steam_flow) * dt
        self.sg_mass += mass_delta
        
        # Convert Mass to Level (simplified cylinder)
        # Nominal mass 50000 kg. Level range +/- 1m corresponds to +/- 5000 kg
        self.sg_level = (self.sg_mass - 50000.0) / 5000.0
        
        # 5. Pressure Dynamics
        # Pressure depends on energy balance mismatch (production vs consumption)
        # If Q_trans > P_turbine_load, Pressure rises
        power_mismatch = q_trans - (self.steam_flow * delta_h)
        # Pressure heat capacity of the system
        dp_dt = power_mismatch / 1e4 # Scaling factor tuned for time constant ~30s
        self.sg_pressure += dp_dt * dt
        
        # Safety valves
        if self.sg_pressure > 8.0e6: # 80 atm relief
             self.sg_pressure = 8.0e6
             self.steam_flow += 100.0 # Venting
             
        return t_out_cold_leg # Return T_cold to reactor inlet

    def get_saturation_temp(self, pressure_pa):
        # Antoine-like approximation for Water 50-100 atm
        # P (bar) vs T (C)
        # 64 bar -> ~280 C
        p_bar = pressure_pa / 1e5
        # Power law fit: T ~ 100 * P^0.25
        return 100.0 * (p_bar ** 0.25)

