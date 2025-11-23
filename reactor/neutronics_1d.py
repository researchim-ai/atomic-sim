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
    
    def __init__(self, num_nodes=50, core_height=350.0, device='cpu', dtype=torch.float64):
        """
        Args:
            num_nodes: количество узлов по высоте
            core_height: высота активной зоны (см)
            device: устройство (cpu/cuda)
            dtype: тип данных
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.num_nodes = num_nodes
        self.H = core_height
        self.dz = core_height / num_nodes
        self.dz_sq = self.dz ** 2
        
        # === Физические константы (Пример для PWR) ===
        # Скорость нейтронов (тепловая группа) ~ 2200 м/с, но для 1-группового приближения 
        # берется усредненная скорость спектра ~ 10^5 - 10^6 см/с
        self.register_buffer('v', torch.tensor(1.0e5, device=device, dtype=dtype))
        
        # Константы запаздывающих нейтронов (6 групп)
        self.register_buffer('beta_i', torch.tensor(
            [0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273], 
            device=device, dtype=dtype
        ))
        self.register_buffer('lambda_i', torch.tensor(
            [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01], 
            device=device, dtype=dtype
        ))
        self.register_buffer('beta_total', torch.sum(self.beta_i))
        
        # === Сечения (Профили по высоте) ===
        # Инициализируем однородными, потом можно менять
        # D: Коэф диффузии ~ 1.0 - 1.5 см
        self.register_buffer('D_base', torch.ones(num_nodes, device=device, dtype=dtype) * 1.2)
        
        # Sigma_a: Сечение поглощения ~ 0.01 - 0.02 см^-1 (без стержней)
        self.register_buffer('Sigma_a_base', torch.ones(num_nodes, device=device, dtype=dtype) * 0.015)
        
        # nu*Sigma_f: Сечение генерации. Должно быть чуть больше Sigma_a + утечка для критичности
        # Для 350см реактора геометрический лапласиан B_g^2 ~ (pi/H)^2 ~ (3.14/350)^2 ~ 8e-5
        # k_eff = nu*Sigma_f / (Sigma_a + D*B^2) ~ 1
        # nu*Sigma_f ~ Sigma_a + D*B^2 ~ 0.015 + 1.2 * 8e-5 ~ 0.0151
        self.register_buffer('nu_Sigma_f_base', torch.ones(num_nodes, device=device, dtype=dtype) * 0.01515)
        
        # Текущие сечения (могут меняться от температуры и стержней)
        self.D = self.D_base.clone()
        self.Sigma_a = self.Sigma_a_base.clone()
        self.nu_Sigma_f = self.nu_Sigma_f_base.clone()
        
        # Состояние
        # phi: Поток нейтронов [n/cm^2/s] (вектор N)
        self.phi = None
        # C: Концентрации предшественников [atom/cm^3] (матрица 6 x N)
        self.C = None
        
        # Позиции стержней (0.0 - извлечены, 1.0 - полностью внутри)
        # Можно задать один стержень на весь реактор или банк стержней
        self.rod_position = 0.0 
        # Эффективность стержней (добавка к Sigma_a при полном погружении)
        self.rod_worth_total = 0.005 # ~0.5% dRho ~ 5 beta
        
        self.reset()

    def reset(self):
        """Инициализация потока формой косинуса и равновесных предшественников"""
        z = torch.linspace(0, torch.pi, self.num_nodes, device=self.device, dtype=self.dtype)
        
        # Начальный поток (номинал ~ 1e13)
        self.phi = torch.sin(z) * 1e13
        self.phi = torch.clamp(self.phi, min=1e7) # Не даем уйти в 0 на границах для стабильности
        
        # Сброс сечений к базовым
        self.D = self.D_base.clone()
        self.Sigma_a = self.Sigma_a_base.clone()
        self.nu_Sigma_f = self.nu_Sigma_f_base.clone()
        
        # Равновесные концентрации предшественников
        # dC/dt = 0 => lambda * C = beta * nu * Sigma_f * phi
        # C = (beta / lambda) * nu * Sigma_f * phi
        
        # (6, 1) * (1, N) -> (6, N)
        production_rate = self.nu_Sigma_f * self.phi
        coeff = self.beta_i.unsqueeze(1) / self.lambda_i.unsqueeze(1)
        self.C = coeff * production_rate.unsqueeze(0)

    def update_cross_sections(self, rod_pos, temp_feedback=None, xenon_absorption=None):
        """
        Обновление сечений на основе положения стержней, температур и отравления.
        Args:
            rod_pos: 0.0 (top/out) to 1.0 (bottom/in). Стержни входят СВЕРХУ.
            temp_feedback: (опционально) корректировка сечений от температуры
            xenon_absorption: (опционально) вектор (N,) макроскопического сечения ксенона
        """
        self.rod_position = rod_pos
        
        # Сброс к базе
        self.Sigma_a = self.Sigma_a_base.clone()
        
        # === Эффект Ксенона ===
        if xenon_absorption is not None:
            # Добавляем поглощение ксенона к базовому поглощению
            self.Sigma_a += xenon_absorption
        
        # === Эффект стержней ===
        # Стержни входят сверху (индекс 0) вниз (индекс N)
        # rod_pos = 0.5 означает, что стержни закрывают верхнюю половину (0 .. N/2)
        
        insertion_idx = int(rod_pos * self.num_nodes)
        
        if insertion_idx > 0:
            # Добавляем сечение поглощения в зону, где есть стержень
            # Профиль поглощения стержня считаем равномерным
            self.Sigma_a[:insertion_idx] += self.rod_worth_total
            
        # TODO: Добавить температурные эффекты (Доплер)
        if temp_feedback is not None:
            # temp_feedback - это вектор изменения реактивности (rho)
            # rho ~ -delta_Sigma_a / Sigma_a  => delta_Sigma_a ~ - rho * Sigma_a
            # Отрицательная реактивность увеличивает сечение поглощения
            
            delta_Sigma_a = -temp_feedback * self.Sigma_a_base
            self.Sigma_a += delta_Sigma_a

    def compute_diffusion_term(self, phi):
        """
        Вычисляет утечку: ∇·D∇φ с помощью конечных разностей.
        Возвращает вектор той же длины, что и phi.
        Граничные условия: φ=0 на границах (Vacuum boundary conditions аппроксимация)
        """
        # Создаем паддинг нулями (граничные условия)
        # (1, 1, N) -> (1, 1, N+2)
        phi_padded = F.pad(phi.unsqueeze(0).unsqueeze(0), (1, 1), mode='constant', value=0).squeeze()
        
        # Вычисляем потоки через грани ячеек (Current J = -D * dphi/dz)
        # J_{i+1/2} = - D_{i+1/2} * (phi_{i+1} - phi_i) / dz
        
        # D интерполируем на границы ячеек. 
        # D_padded для удобства
        D_padded = F.pad(self.D.unsqueeze(0).unsqueeze(0), (1, 1), mode='replicate').squeeze()
        
        # D на границах ячеек (среднее арифметическое)
        # D_edges[i] соответствует D_{i-1/2}
        D_edges = 0.5 * (D_padded[:-1] + D_padded[1:]) 
        
        # Производные dphi/dz на границах
        dphi = (phi_padded[1:] - phi_padded[:-1]) / self.dz
        
        # Токи J
        J = -D_edges * dphi
        
        # Дивергенция тока -dJ/dz = (J_in - J_out) / dz
        # leakage = (J_{i-1/2} - J_{i+1/2}) / dz
        # Обратите внимание на индексы. J имеет размер N+1 (границы N ячеек)
        
        leakage = -(J[1:] - J[:-1]) / self.dz
        
        return leakage

    def compute_derivatives(self, phi, C):
        """
        Вычисляет dphi/dt и dC/dt
        """
        # 1. Диффузия (Leakage)
        leakage = self.compute_diffusion_term(phi)
        
        # 2. Генерация и поглощение
        # Prompt production: (1 - beta) * nu * Sigma_f * phi
        prompt_production = (1.0 - self.beta_total) * self.nu_Sigma_f * phi
        
        # Absorption: Sigma_a * phi
        absorption = self.Sigma_a * phi
        
        # 3. Запаздывающие нейтроны (сумма по группам)
        # Source from precursors: sum(lambda_i * C_i)
        # C: (6, N) -> sum(dim=0) -> (N,)
        delayed_source = torch.sum(self.lambda_i.unsqueeze(1) * C, dim=0)
        
        # === Уравнение для phi ===
        # 1/v * dphi/dt = Leakage + PromptProd - Abs + DelayedSrc
        dphi_dt = self.v * (leakage + prompt_production - absorption + delayed_source)
        
        # === Уравнение для C ===
        # dC_i/dt = beta_i * nu * Sigma_f * phi - lambda_i * C_i
        # production: (N,) -> expand to (6, N)
        fission_rate = self.nu_Sigma_f * phi
        precursor_production = self.beta_i.unsqueeze(1) * fission_rate.unsqueeze(0)
        precursor_decay = self.lambda_i.unsqueeze(1) * C
        
        dC_dt = precursor_production - precursor_decay
        
        return dphi_dt, dC_dt

    def step(self, dt, rod_pos=None):
        """
        Шаг по времени (RK4)
        """
        if rod_pos is not None:
            self.update_cross_sections(rod_pos)
            
        # RK4 Integration
        phi0 = self.phi.clone()
        C0 = self.C.clone()
        
        # k1
        k1_phi, k1_C = self.compute_derivatives(phi0, C0)
        
        # k2
        phi_mid = phi0 + 0.5 * dt * k1_phi
        C_mid = C0 + 0.5 * dt * k1_C
        # Защита от отрицательного потока (физически невозможно)
        phi_mid = torch.clamp(phi_mid, min=1e-10)
        k2_phi, k2_C = self.compute_derivatives(phi_mid, C_mid)
        
        # k3
        phi_mid = phi0 + 0.5 * dt * k2_phi
        C_mid = C0 + 0.5 * dt * k2_C
        phi_mid = torch.clamp(phi_mid, min=1e-10)
        k3_phi, k3_C = self.compute_derivatives(phi_mid, C_mid)
        
        # k4
        phi_end = phi0 + dt * k3_phi
        C_end = C0 + dt * k3_C
        phi_end = torch.clamp(phi_end, min=1e-10)
        k4_phi, k4_C = self.compute_derivatives(phi_end, C_end)
        
        # Update
        self.phi = phi0 + (dt / 6.0) * (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)
        self.C = C0 + (dt / 6.0) * (k1_C + 2*k2_C + 2*k3_C + k4_C)
        
        # Final clamp
        self.phi = torch.clamp(self.phi, min=1e-10)
        self.C = torch.clamp(self.C, min=1e-10)
        
        # Возвращаем полную мощность (сумма по объему)
        # Power ~ proportional to sum(phi * dz)
        # Для совместимости с интерфейсом возвращаем нормированную мощность
        # Предположим, что 100 МВт соответствует среднему потоку 1e13 * объемный фактор
        # Вернем просто средний поток для начала или сумму
        return self.phi.mean()
    
    def get_state(self):
        return {
            'flux_profile': self.phi.cpu().numpy(),
            'avg_flux': self.phi.mean().item(),
            'peak_flux': self.phi.max().item(),
            'axial_offset': self.compute_axial_offset()
        }
        
    def compute_axial_offset(self):
        """
        Вычисляет аксиальный оффсет (Ao):
        Ao = (P_top - P_bottom) / (P_top + P_bottom)
        Где P_top - мощность в верхней половине, P_bottom - в нижней.
        Важный параметр для управления PWR.
        """
        mid = self.num_nodes // 2
        p_top = torch.sum(self.phi[:mid])
        p_bottom = torch.sum(self.phi[mid:])
        
        if (p_top + p_bottom) > 1e-6:
            return ((p_top - p_bottom) / (p_top + p_bottom)).item()
        return 0.0

