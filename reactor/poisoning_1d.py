"""
1D Xenon-Iodine Poisoning Model
Моделирует пространственное распределение Ксенона-135 и Йода-135.
"""

import torch
import torch.nn as nn

class XenonIodine1D(nn.Module):
    """
    Модель отравления Ксеноном-135 в 1D геометрии.
    
    Уравнения для каждого узла z:
    dI/dt = gamma_I * Sigma_f * phi - lambda_I * I
    dXe/dt = gamma_Xe * Sigma_f * phi + lambda_I * I - lambda_Xe * Xe - sigma_a_Xe * phi * Xe
    
    Это позволяет моделировать ксеноновые волны (axial xenon oscillations).
    """
    
    def __init__(self, num_nodes=50, device='cpu', dtype=torch.float64):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.num_nodes = num_nodes
        
        # === Физические константы ===
        # Yields (выходы продуктов деления)
        self.register_buffer('gamma_I', torch.tensor(0.061, device=device, dtype=dtype))
        self.register_buffer('gamma_Xe', torch.tensor(0.002, device=device, dtype=dtype))
        
        # Decay constants (1/s)
        # I-135 half-life ~ 6.57 hours
        self.register_buffer('lambda_I', torch.tensor(2.9e-5, device=device, dtype=dtype))
        # Xe-135 half-life ~ 9.14 hours
        self.register_buffer('lambda_Xe', torch.tensor(2.1e-5, device=device, dtype=dtype))
        
        # Microscopic absorption cross-section for Xe-135 (cm^2)
        # Huge value (~2.6e6 barns) -> ~2.6e-18 cm^2
        # WARNING: In diffusion equation, we use macroscopic Sigma. 
        # Here we compute number density N_Xe (atoms/cm^3).
        # Macroscopic Sigma_Xe = sigma_a_Xe * N_Xe
        self.register_buffer('sigma_a_Xe', torch.tensor(2.6e-18, device=device, dtype=dtype))
        
        # State vectors (N,) - концентрации [atoms/cm^3]
        self.I = None
        self.Xe = None
        
        # Initial approximate flux for equilibrium calc
        self.nominal_flux = 1e13 
        self.nu_sigma_f_ref = 0.018 # Reference macroscopic fission xs (SYNC WITH NEUTRONICS)
        
        self.reset()
        
    def reset(self, flux_profile=None):
        """
        Сброс к равновесному состоянию для заданного профиля потока.
        Если flux_profile=None, используется косинусное распределение.
        """
        if flux_profile is None:
            z = torch.linspace(0, torch.pi, self.num_nodes, device=self.device, dtype=self.dtype)
            flux_profile = torch.sin(z) * self.nominal_flux
            flux_profile = torch.clamp(flux_profile, min=1e8)
            
        # Equilibrium Iodine: dI/dt = 0
        # I_eq = (gamma_I * Sigma_f * phi) / lambda_I
        # Note: Sigma_f is macroscopic fission cross section. Assume roughly const or passed.
        # We use a reference value here for initialization.
        
        source_term = self.nu_sigma_f_ref * flux_profile # Approximation: nu*Sigma_f ~ Sigma_f * 2.4
        # Let's just use Sigma_f_ref approx = nu_Sigma_f / 2.43
        sigma_f_approx = self.nu_sigma_f_ref / 2.43
        
        self.I = (self.gamma_I * sigma_f_approx * flux_profile) / self.lambda_I
        
        # Equilibrium Xenon: dXe/dt = 0
        # Xe_eq = (gamma_Xe * Sigma_f * phi + lambda_I * I) / (lambda_Xe + sigma_a_Xe * phi)
        numerator = self.gamma_Xe * sigma_f_approx * flux_profile + self.lambda_I * self.I
        denominator = self.lambda_Xe + self.sigma_a_Xe * flux_profile
        
        self.Xe = numerator / denominator
        
    def compute_derivatives(self, I, Xe, flux, sigma_f):
        """
        Вычисление производных dI/dt, dXe/dt.
        Args:
            I, Xe: текущие концентрации (N,)
            flux: профиль потока (N,)
            sigma_f: макроскопическое сечение деления (N,)
        """
        # Iodine
        dI = self.gamma_I * sigma_f * flux - self.lambda_I * I
        
        # Xenon
        # Production from fission + Decay of Iodine - Decay of Xenon - Burnup of Xenon
        production = self.gamma_Xe * sigma_f * flux + self.lambda_I * I
        loss = self.lambda_Xe * Xe + self.sigma_a_Xe * flux * Xe
        
        dXe = production - loss
        
        return dI, dXe
        
    def step(self, dt, flux, nu_sigma_f):
        """
        Шаг интегрирования по времени.
        Args:
            dt: шаг времени (с)
            flux: вектор потока нейтронов (N,)
            nu_sigma_f: вектор сечения генерации (N,)
        """
        # Convert nu_sigma_f to sigma_f approx
        sigma_f = nu_sigma_f / 2.43
        
        # RK4 Integration
        I0 = self.I.clone()
        Xe0 = self.Xe.clone()
        
        k1_I, k1_Xe = self.compute_derivatives(I0, Xe0, flux, sigma_f)
        
        k2_I, k2_Xe = self.compute_derivatives(
            I0 + 0.5*dt*k1_I, Xe0 + 0.5*dt*k1_Xe, flux, sigma_f
        )
        
        k3_I, k3_Xe = self.compute_derivatives(
            I0 + 0.5*dt*k2_I, Xe0 + 0.5*dt*k2_Xe, flux, sigma_f
        )
        
        k4_I, k4_Xe = self.compute_derivatives(
            I0 + dt*k3_I, Xe0 + dt*k3_Xe, flux, sigma_f
        )
        
        self.I = I0 + (dt/6.0)*(k1_I + 2*k2_I + 2*k3_I + k4_I)
        self.Xe = Xe0 + (dt/6.0)*(k1_Xe + 2*k2_Xe + 2*k3_Xe + k4_Xe)
        
        # Ensure non-negative
        self.I = torch.clamp(self.I, min=0.0)
        self.Xe = torch.clamp(self.Xe, min=0.0)
        
        return self.get_absorption_cross_section()
        
    def get_absorption_cross_section(self):
        """
        Возвращает макроскопическое сечение поглощения ксенона (Sigma_Xe).
        Sigma_Xe = sigma_a_Xe * N_Xe
        Returns: Vector (N,) [cm^-1]
        """
        return self.sigma_a_Xe * self.Xe

    def get_state(self):
        return {
            'Xe_concentration': self.Xe.cpu().numpy(),
            'I_concentration': self.I.cpu().numpy(),
            'Xe_poisoning_reactivity': 0.0 # TODO: convert to rho if needed
        }

