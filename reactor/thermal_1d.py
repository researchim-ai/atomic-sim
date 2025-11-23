"""
1D Thermal-Hydraulic Model
Моделирует распределение температур топлива, оболочки и теплоносителя по высоте канала.
"""

import torch
import torch.nn as nn

class ThermalModel1D(nn.Module):
    """
    1D тепловая модель (single channel average).
    
    Сетка соответствует нейтронной модели (0 = TOP, N = BOTTOM).
    Теплоноситель течет снизу вверх (от N-1 к 0).
    """
    
    def __init__(self, num_nodes=50, device='cpu', dtype=torch.float64):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.num_nodes = num_nodes
        
        # === Физические параметры ===
        # Теплоемкости (МДж/К на узел)
        # Нормированы на один узел. Если весь реактор C_total, то здесь C_total/N
        self.register_buffer('C_fuel', torch.tensor(4.0, device=device, dtype=dtype))
        self.register_buffer('C_clad', torch.tensor(0.4, device=device, dtype=dtype))
        self.register_buffer('C_coolant', torch.tensor(2.0, device=device, dtype=dtype))
        
        # Коэффициенты теплопередачи (МВт/К на узел)
        self.register_buffer('h_gap', torch.tensor(0.4, device=device, dtype=dtype)) # Fuel -> Clad
        self.register_buffer('h_film', torch.tensor(0.6, device=device, dtype=dtype)) # Clad -> Coolant
        
        # Параметры потока
        # Advection time scale = Volume / FlowRate
        self.register_buffer('tau_flow_nominal', torch.tensor(0.5, device=device, dtype=dtype)) # seconds to pass one node? No, total transit time.
        # Transit time ~ 1-2 seconds for core.
        # Let's say transit time is 1.0s. Then per node dt = 1.0 / 50 = 0.02s.
        # Flow rate coefficient for advection term: m_dot * Cp
        self.register_buffer('w_Cp', torch.tensor(2.0, device=device, dtype=dtype)) # MW/K (flow heat capacity rate)
        
        self.register_buffer('T_inlet', torch.tensor(280.0, device=device, dtype=dtype)) # Inlet at bottom
        
        # Температурные коэффициенты реактивности (для обратной связи)
        # Теперь это не просто скаляр, а локальный эффект
        self.register_buffer('alpha_fuel', torch.tensor(-2e-5, device=device, dtype=dtype)) # per K
        self.register_buffer('alpha_coolant', torch.tensor(-5e-4, device=device, dtype=dtype)) # per K (stronger)
        
        # State vectors (size N)
        self.T_fuel = None
        self.T_clad = None
        self.T_coolant = None
        self.flow_factor = 1.0
        
        self.reset()
        
    def reset(self):
        # Initial profile
        # Linear rise for coolant, roughly cosine for fuel (following flux)
        z_idx = torch.arange(self.num_nodes, device=self.device, dtype=self.dtype)
        
        # Coolant: 280 at bottom (idx N-1) -> 310 at top (idx 0)
        # Linear interpolation
        # normalized height x from 0 (top) to 1 (bottom)
        x = z_idx / (self.num_nodes - 1)
        self.T_coolant = 310.0 - 30.0 * x # 310 at top, 280 at bottom
        
        # Fuel and Clad slightly hotter
        self.T_clad = self.T_coolant + 10.0
        self.T_fuel = self.T_clad + 100.0 # Centerline
        
        self.flow_factor = 1.0

    def compute_derivatives(self, T_f, T_c, T_co, power_profile):
        """
        Args:
            T_f, T_c, T_co: Temperature vectors (N,)
            power_profile: Power deposition vector (MW per node) (N,)
        """
        # 1. Fuel Equation
        # dT_f/dt = (Power - Heat_to_clad) / C_f
        Q_gap = self.h_gap * (T_f - T_c)
        dT_f = (power_profile - Q_gap) / self.C_fuel
        
        # 2. Clad Equation
        # dT_c/dt = (Heat_from_fuel - Heat_to_coolant) / C_c
        Q_film = self.h_film * (T_c - T_co)
        dT_c = (Q_gap - Q_film) / self.C_clad
        
        # 3. Coolant Equation (with Advection)
        # dT_co/dt = (Heat_from_clad - Advection) / C_co
        
        # Advection term: w * Cp * (T_out - T_in) for the node
        # Flow is Bottom (N-1) -> Top (0)
        # T_in for node i is T_{i+1}. For node N-1 is T_inlet.
        
        # Create shifted vector for T_in
        # T_co indices: 0 1 ... N-1
        # T_in indices: 1 2 ... N-1 Inlet
        
        T_in_node = torch.roll(T_co, shifts=-1, dims=0) # shift left (up in index)
        T_in_node[-1] = self.T_inlet # Set bottom boundary condition
        
        # Energy balance per node:
        # C_node * dT/dt = Q_film - w*Cp * (T_node - T_in_node)
        # Note: Advection is usually dH/dx. Here finite volume.
        # Heat removal by flow = flow_rate * (T_out - T_in)
        # Here T_out = T_node (well-mixed assumption) or T_edge.
        # Using Upwind Difference Scheme: T_out = T_i, T_in = T_{i+1}
        
        Q_advection = self.w_Cp * self.flow_factor * (T_co - T_in_node)
        
        dT_co = (Q_film - Q_advection) / self.C_coolant
        
        return dT_f, dT_c, dT_co

    def step(self, dt, power_profile_MW, flow_factor=1.0):
        self.flow_factor = flow_factor
        
        # RK4 Integration
        tf0, tc0, tco0 = self.T_fuel.clone(), self.T_clad.clone(), self.T_coolant.clone()
        
        k1_f, k1_c, k1_co = self.compute_derivatives(tf0, tc0, tco0, power_profile_MW)
        
        k2_f, k2_c, k2_co = self.compute_derivatives(
            tf0 + 0.5*dt*k1_f, tc0 + 0.5*dt*k1_c, tco0 + 0.5*dt*k1_co, power_profile_MW
        )
        
        k3_f, k3_c, k3_co = self.compute_derivatives(
            tf0 + 0.5*dt*k2_f, tc0 + 0.5*dt*k2_c, tco0 + 0.5*dt*k2_co, power_profile_MW
        )
        
        k4_f, k4_c, k4_co = self.compute_derivatives(
            tf0 + dt*k3_f, tc0 + dt*k3_c, tco0 + dt*k3_co, power_profile_MW
        )
        
        self.T_fuel = tf0 + (dt/6.0)*(k1_f + 2*k2_f + 2*k3_f + k4_f)
        self.T_clad = tc0 + (dt/6.0)*(k1_c + 2*k2_c + 2*k3_c + k4_c)
        self.T_coolant = tco0 + (dt/6.0)*(k1_co + 2*k2_co + 2*k3_co + k4_co)
        
        return self.compute_reactivity_feedback()
        
    def compute_reactivity_feedback(self):
        """
        Calculates reactivity feedback profile (delta Sigma_a or delta rho).
        Returns: vector (N,) of reactivity changes.
        """
        # Simplification: linearized feedback around some reference
        # We assume reference is roughly the initial state (reset state).
        # Let's take 300C and 600C as arbitrary base for now or calculate deviation.
        
        # For simplicity in this iteration: returns raw reactivity delta
        d_rho_fuel = self.alpha_fuel * (self.T_fuel - 600.0) # ref 600
        d_rho_coolant = self.alpha_coolant * (self.T_coolant - 290.0) # ref 290
        
        return d_rho_fuel + d_rho_coolant

    def get_state(self):
        return {
            'max_fuel_temp': self.T_fuel.max().item(),
            'max_clad_temp': self.T_clad.max().item(),
            'outlet_temp': self.T_coolant[0].item(), # Top node
            'inlet_temp': self.T_coolant[-1].item(),
            'avg_fuel_temp': self.T_fuel.mean().item()
        }

