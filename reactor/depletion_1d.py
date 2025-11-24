import torch
import torch.nn as nn

class FuelDepletion1D(nn.Module):
    def __init__(self, num_nodes=50, device='cpu', dtype=torch.float64):
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device
        self.dtype = dtype
        
        # === Физические константы (микроскопические сечения в барнах) ===
        # 1 барн = 1e-24 см^2
        barn = 1e-24
        
        # U-235 (Основное топливо)
        self.sigma_a_U235 = 680.0 * barn
        self.sigma_f_U235 = 580.0 * barn
        self.nu_U235 = 2.43
        
        # U-238 (Сырье для плутония)
        self.sigma_a_U238 = 2.7 * barn # Capture only mainly
        self.sigma_f_U238 = 0.0 # Fast fission ignored in 1-group thermal
        
        # Pu-239 (Наработанное топливо)
        self.sigma_a_Pu239 = 1011.0 * barn
        self.sigma_f_Pu239 = 740.0 * barn
        self.nu_Pu239 = 2.87
        
        # Начальные концентрации ядер (ядер/см^3)
        # ВАЖНО: Нормировка под макроскопические сечения нейтроники.
        # В модели нейтроники Sigma ~ 0.02 см^-1.
        # Реальная плотность UO2 дает Sigma ~ 2.0 см^-1 (в 100 раз больше).
        # Для согласования уменьшаем эффективную концентрацию N_total.
        N_total = 2.3e20 # Было 2.3e22
        enrichment = 0.045
        
        self.register_buffer('N_U235', torch.ones(num_nodes, device=device, dtype=dtype) * (N_total * enrichment))
        self.register_buffer('N_U238', torch.ones(num_nodes, device=device, dtype=dtype) * (N_total * (1 - enrichment)))
        self.register_buffer('N_Pu239', torch.zeros(num_nodes, device=device, dtype=dtype))
        
        # Начальные макроскопические сечения (сохраняем для delta расчета)
        self._update_macroscopic()
        self.Sigma_a_initial = self.Sigma_a_fuel.clone()
        self.nu_Sigma_f_initial = self.nu_Sigma_f_fuel.clone()
        
        # Время в секундах, прошедшее с начала кампании (EFPS - Effective Full Power Seconds)
        self.time_burned = 0.0

    def _update_macroscopic(self):
        """Пересчитывает макроскопические сечения топлива на основе концентраций."""
        # Sigma_a = sum(N_i * sigma_a_i)
        self.Sigma_a_fuel = (
            self.N_U235 * self.sigma_a_U235 +
            self.N_U238 * self.sigma_a_U238 +
            self.N_Pu239 * self.sigma_a_Pu239
        )
        
        # nu*Sigma_f = sum(N_i * sigma_f_i * nu_i)
        self.nu_Sigma_f_fuel = (
            self.N_U235 * self.sigma_f_U235 * self.nu_U235 +
            self.N_Pu239 * self.sigma_f_Pu239 * self.nu_Pu239
        )

    def step(self, flux, dt_seconds):
        """
        Выполняет шаг выгорания.
        flux: нейтронный поток [н/см^2/с]
        dt_seconds: шаг времени [с]
        """
        # Уравнения Бейтмена (упрощенные)
        # dN_235 = -sigma_a_235 * phi * N_235 * dt
        loss_U235 = self.sigma_a_U235 * flux * self.N_U235 * dt_seconds
        self.N_U235 -= loss_U235
        
        # dN_238 = -sigma_a_238 * phi * N_238 * dt (убыль мала, но есть)
        loss_U238 = self.sigma_a_U238 * flux * self.N_U238 * dt_seconds
        self.N_U238 -= loss_U238
        
        # dN_Pu239 = (sigma_c_238 * phi * N_238) - (sigma_a_Pu239 * phi * N_Pu239)
        # sigma_c_U238 ~ sigma_a_U238 (так как деления нет)
        production_Pu = self.sigma_a_U238 * flux * self.N_U238 * dt_seconds
        loss_Pu = self.sigma_a_Pu239 * flux * self.N_Pu239 * dt_seconds
        self.N_Pu239 += (production_Pu - loss_Pu)
        
        self.time_burned += dt_seconds
        self._update_macroscopic()

    def get_cross_section_changes(self):
        """Возвращает ИЗМЕНЕНИЯ сечений относительно начала кампании."""
        # Это нужно, чтобы добавить к базовым сечениям в нейтронике
        delta_Sigma_a = self.Sigma_a_fuel - self.Sigma_a_initial
        delta_nu_Sigma_f = self.nu_Sigma_f_fuel - self.nu_Sigma_f_initial
        return delta_Sigma_a, delta_nu_Sigma_f

    def get_state(self):
        return {
            'burnup_days': self.time_burned / 86400.0,
            'N_U235_avg': self.N_U235.mean().item(),
            'N_Pu239_avg': self.N_Pu239.mean().item(),
            'fuel_reactivity_change': (self.nu_Sigma_f_fuel.mean() - self.Sigma_a_fuel.mean()).item()
        }

