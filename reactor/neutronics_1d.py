
"""
1D Multigroup Neutron Diffusion Model
Реализует пространственную кинетику реактора в 1D геометрии (осевое распределение).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Neutronics1D(nn.Module):
    """
    1D модель нейтронной кинетики на основе уравнения диффузии.
    
    Уравнение (1 группа):
    1/v * dφ/dt = d/dz(D * dφ/dz) + (νΣf * (1-β) - Σa) * φ + Σ λi * Ci
    dCi/dt = βi * νΣf * φ - λi * Ci
    
    Пространственная дискретизация: Finite Difference Method
    """
    
    def __init__(self, config, num_nodes=50, device='cpu', dtype=torch.float64):
        """
        Args:
            config: ReactorConfig object containing physical params
            num_nodes: количество узлов по высоте
            device: устройство (cpu/cuda)
            dtype: тип данных
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.num_nodes = num_nodes
        
        # Geometry (config has meters, we work in cm usually, or SI meters?
        # Let's work in cm for cross-sections stability as per tradition, or adapt.
        # Code uses cm^-1 generally.
        self.H = config.core_height * 100.0 # m -> cm
        self.dz = self.H / num_nodes
        self.dz_sq = self.dz ** 2
        
        # === Физические константы ===
        # Velocity
        self.register_buffer('v', torch.tensor(config.neutron_speed, device=device, dtype=dtype))
        
        # Delayed Neutrons (Approximate 6-group data, scaled to match beta_eff)
        # Standard parameters for U-235 thermal fission
        beta_fractions = torch.tensor(
            [0.033, 0.219, 0.196, 0.395, 0.115, 0.042], device=device, dtype=dtype
        ) # Relative fractions sum to 1.0
        
        self.register_buffer('beta_i', beta_fractions * config.beta_eff)
        
        self.register_buffer('lambda_i', torch.tensor(
            [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01], 
            device=device, dtype=dtype
        ))
        self.register_buffer('beta_total', torch.sum(self.beta_i))
        
        # === Сечения (Профили по высоте) ===
        # D: Коэф диффузии ~ 1.0 - 1.5 см
        self.register_buffer('D_base', torch.ones(num_nodes, device=device, dtype=dtype) * 1.2)
        
        # Sigma_a: Сечение поглощения ~ 0.01 - 0.02 см^-1 (без стержней)
        self.register_buffer('Sigma_a_base', torch.ones(num_nodes, device=device, dtype=dtype) * 0.015)
        
        # nu*Sigma_f: Criticality search approximation
        # B_g^2 = (pi/H)^2
        geometric_buckling = (math.pi / self.H) ** 2
        # k_inf = 1 + M^2 * B^2 => nu_Sigma_f / Sigma_a = 1 + L^2 * B^2
        # Approx: nu_Sigma_f = Sigma_a + D * B^2
        critical_production = 0.015 + 1.2 * geometric_buckling
        
        # Add excess reactivity for burnup/control (e.g. +5-10%)
        initial_excess = 1.08 
        self.register_buffer('nu_Sigma_f_base', torch.ones(num_nodes, device=device, dtype=dtype) * (critical_production * initial_excess))
        
        # Параметры Бора
        self.register_buffer('sigma_a_boron', torch.tensor(5.0e-6, device=device, dtype=dtype)) # per ppm
        
        # Текущие сечения
        self.D = self.D_base.clone()
        self.Sigma_a = self.Sigma_a_base.clone()
        self.nu_Sigma_f = self.nu_Sigma_f_base.clone()
        
        # Состояние
        self.phi = None
        self.C = None
        
        self.rod_position = 0.0 
        self.rod_worth_total = 0.02
        
        # Source Term
        self.register_buffer('source_term', torch.ones(num_nodes, device=device, dtype=dtype) * 1e5)
        
        self.reset()

    def reset(self):
        """Инициализация потока формой косинуса и равновесных предшественников"""
        z = torch.linspace(0, torch.pi, self.num_nodes, device=self.device, dtype=self.dtype)
        
        # Начальный поток (номинал ~ 1e13)
        self.phi = torch.sin(z) * 1e13
        self.phi = torch.clamp(self.phi, min=1e7)
        
        self.D = self.D_base.clone()
        self.Sigma_a = self.Sigma_a_base.clone()
        self.nu_Sigma_f = self.nu_Sigma_f_base.clone()
        
        production_rate = self.nu_Sigma_f * self.phi
        coeff = self.beta_i.unsqueeze(1) / self.lambda_i.unsqueeze(1)
        self.C = coeff * production_rate.unsqueeze(0)
        
        self.boron_concentration = 0.0

    def update_cross_sections(self, rod_pos, boron_ppm=0.0, temp_feedback=None, xenon_absorption=None, fuel_feedback=None):
        self.rod_position = rod_pos
        self.boron_concentration = boron_ppm
        
        self.Sigma_a = self.Sigma_a_base.clone()
        self.nu_Sigma_f = self.nu_Sigma_f_base.clone() 
        
        # === Эффект Выгорания ===
        if fuel_feedback is not None:
            delta_Sigma_a, delta_nu_Sigma_f = fuel_feedback
            self.Sigma_a += delta_Sigma_a
            self.nu_Sigma_f += delta_nu_Sigma_f
        
        # === Эффект Бора ===
        boron_absorption = self.sigma_a_boron * boron_ppm
        self.Sigma_a += boron_absorption
        
        # === Эффект Ксенона ===
        if xenon_absorption is not None:
            self.Sigma_a += xenon_absorption
        
        # === Эффект стержней ===
        insertion_idx = int(rod_pos * self.num_nodes)
        if insertion_idx > 0:
            self.Sigma_a[:insertion_idx] += self.rod_worth_total
            
        # === Температурная Обратная Связь ===
        if temp_feedback is not None:
            # temp_feedback is delta_rho (reactivity change)
            # delta_rho = (k - k0)/k ~ -delta_Sigma_a / nu_Sigma_f
            # So delta_Sigma_a = -delta_rho * nu_Sigma_f
            
            # Note: We approximate using base nu_Sigma_f scale
            delta_Sigma_a = -temp_feedback * self.nu_Sigma_f_base
            self.Sigma_a += delta_Sigma_a

    def compute_diffusion_term(self, phi):
        phi_padded = F.pad(phi.unsqueeze(0).unsqueeze(0), (1, 1), mode='constant', value=0).squeeze()
        D_padded = F.pad(self.D.unsqueeze(0).unsqueeze(0), (1, 1), mode='replicate').squeeze()
        D_edges = 0.5 * (D_padded[:-1] + D_padded[1:]) 
        dphi = (phi_padded[1:] - phi_padded[:-1]) / self.dz
        J = -D_edges * dphi
        leakage = -(J[1:] - J[:-1]) / self.dz
        return leakage

    def compute_derivatives(self, phi, C):
        leakage = self.compute_diffusion_term(phi)
        prompt_production = (1.0 - self.beta_total) * self.nu_Sigma_f * phi
        absorption = self.Sigma_a * phi
        delayed_source = torch.sum(self.lambda_i.unsqueeze(1) * C, dim=0)
        
        term = leakage + prompt_production - absorption + delayed_source + self.source_term
        dphi_dt = self.v * term
        
        fission_rate = self.nu_Sigma_f * phi
        precursor_production = self.beta_i.unsqueeze(1) * fission_rate.unsqueeze(0)
        precursor_decay = self.lambda_i.unsqueeze(1) * C
        
        dC_dt = precursor_production - precursor_decay
        
        return dphi_dt, dC_dt

    def step(self, dt, rod_pos=None, boron_ppm=None):
        if rod_pos is not None:
            bpm = boron_ppm if boron_ppm is not None else self.boron_concentration
            self.update_cross_sections(rod_pos, boron_ppm=bpm)
            
        phi0 = self.phi.clone()
        C0 = self.C.clone()
        
        k1_phi, k1_C = self.compute_derivatives(phi0, C0)
        
        phi_mid = torch.clamp(phi0 + 0.5 * dt * k1_phi, min=1e-10)
        C_mid = C0 + 0.5 * dt * k1_C
        k2_phi, k2_C = self.compute_derivatives(phi_mid, C_mid)
        
        phi_mid = torch.clamp(phi0 + 0.5 * dt * k2_phi, min=1e-10)
        C_mid = C0 + 0.5 * dt * k2_C
        k3_phi, k3_C = self.compute_derivatives(phi_mid, C_mid)
        
        phi_end = torch.clamp(phi0 + dt * k3_phi, min=1e-10)
        C_end = C0 + dt * k3_C
        k4_phi, k4_C = self.compute_derivatives(phi_end, C_end)
        
        self.phi = torch.clamp(phi0 + (dt / 6.0) * (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi), min=1e-10)
        self.C = torch.clamp(C0 + (dt / 6.0) * (k1_C + 2*k2_C + 2*k3_C + k4_C), min=1e-10)
        
        return self.phi.mean()
    
    def get_state(self):
        return {
            'flux_profile': self.phi.cpu().numpy(),
            'avg_flux': self.phi.mean().item(),
            'peak_flux': self.phi.max().item(),
            'axial_offset': self.compute_axial_offset(),
            'boron_ppm': self.boron_concentration
        }
        
    def compute_axial_offset(self):
        mid = self.num_nodes // 2
        p_top = torch.sum(self.phi[:mid])
        p_bottom = torch.sum(self.phi[mid:])
        if (p_top + p_bottom) > 1e-6:
            return ((p_top - p_bottom) / (p_top + p_bottom)).item()
        return 0.0
