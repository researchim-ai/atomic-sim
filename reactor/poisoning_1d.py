import torch
import torch.nn as nn

class XenonIodine1D(nn.Module):
    """
    Модель отравления Ксеноном-135 в 1D геометрии.
    """
    
    def __init__(self, num_nodes=50, device='cpu', dtype=torch.float64):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.num_nodes = num_nodes
        
        # Yields
        self.register_buffer('gamma_I', torch.tensor(0.061, device=device, dtype=dtype))
        self.register_buffer('gamma_Xe', torch.tensor(0.002, device=device, dtype=dtype))
        
        # Decay constants (1/s)
        self.register_buffer('lambda_I', torch.tensor(2.9e-5, device=device, dtype=dtype))
        self.register_buffer('lambda_Xe', torch.tensor(2.1e-5, device=device, dtype=dtype))
        
        # Microscopic absorption cross-section for Xe-135
        self.register_buffer('sigma_a_Xe', torch.tensor(2.6e-18, device=device, dtype=dtype))
        
        # State vectors
        self.I = None
        self.Xe = None
        
        self.nominal_flux = 1e13 
        self.nu_sigma_f_ref = 0.018 
        
        self.reset()
        
    def reset(self, flux_profile=None):
        if flux_profile is None:
            z = torch.linspace(0, torch.pi, self.num_nodes, device=self.device, dtype=self.dtype)
            flux_profile = torch.sin(z) * self.nominal_flux
            flux_profile = torch.clamp(flux_profile, min=1e8)
            
        sigma_f_approx = self.nu_sigma_f_ref / 2.43
        
        self.I = (self.gamma_I * sigma_f_approx * flux_profile) / self.lambda_I
        
        numerator = self.gamma_Xe * sigma_f_approx * flux_profile + self.lambda_I * self.I
        denominator = self.lambda_Xe + self.sigma_a_Xe * flux_profile
        
        self.Xe = numerator / denominator
        
    def compute_derivatives(self, I, Xe, flux, sigma_f):
        dI = self.gamma_I * sigma_f * flux - self.lambda_I * I
        production = self.gamma_Xe * sigma_f * flux + self.lambda_I * I
        loss = self.lambda_Xe * Xe + self.sigma_a_Xe * flux * Xe
        dXe = production - loss
        return dI, dXe
        
    def step(self, dt, flux, nu_sigma_f=None):
        if nu_sigma_f is None:
            if isinstance(flux, torch.Tensor):
                # Use ref scalar or vector
                sigma_f = self.nu_sigma_f_ref / 2.43
            else:
                sigma_f = 0.01
        else:
            sigma_f = nu_sigma_f / 2.43
        
        I0 = self.I.clone()
        Xe0 = self.Xe.clone()
        
        k1_I, k1_Xe = self.compute_derivatives(I0, Xe0, flux, sigma_f)
        k2_I, k2_Xe = self.compute_derivatives(I0 + 0.5*dt*k1_I, Xe0 + 0.5*dt*k1_Xe, flux, sigma_f)
        k3_I, k3_Xe = self.compute_derivatives(I0 + 0.5*dt*k2_I, Xe0 + 0.5*dt*k2_Xe, flux, sigma_f)
        k4_I, k4_Xe = self.compute_derivatives(I0 + dt*k3_I, Xe0 + dt*k3_Xe, flux, sigma_f)
        
        self.I = I0 + (dt/6.0)*(k1_I + 2*k2_I + 2*k3_I + k4_I)
        self.Xe = Xe0 + (dt/6.0)*(k1_Xe + 2*k2_Xe + 2*k3_Xe + k4_Xe)
        
        self.I = torch.clamp(self.I, min=0.0)
        self.Xe = torch.clamp(self.Xe, min=0.0)
        
        return self.get_absorption_cross_section()

    def forward(self, flux, dt, nu_sigma_f=None):
        """PyTorch module forward pass."""
        return self.step(dt, flux, nu_sigma_f)
        
    def get_absorption_cross_section(self):
        return self.sigma_a_Xe * self.Xe

    def get_state(self):
        return {
            'Xe_concentration': self.Xe.cpu().numpy(),
            'I_concentration': self.I.cpu().numpy(),
        }
