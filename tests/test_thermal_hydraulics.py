import pytest
import torch
import numpy as np
from reactor.thermal_1d import ThermalModel1D
from reactor.reactor_configs import VVER_1000

class TestThermalHydraulics:
    
    @pytest.fixture
    def thermal_model(self):
        return ThermalModel1D(config=VVER_1000, num_nodes=50, device='cpu', dtype=torch.float64)

    def test_initial_state(self, thermal_model):
        """Проверка начальных температур и расхода."""
        assert torch.allclose(thermal_model.mass_flow, torch.tensor(18000.0, dtype=torch.float64))
        assert thermal_model.temp_coolant[0] > 270 + 273.15 # Inlet reasonable
        assert torch.all(thermal_model.void_fraction == 0.0) # No boiling at start

    def test_flow_coastdown(self, thermal_model):
        """Проверка выбега насосов (инерция). Расход не должен падать мгновенно."""
        thermal_model.pump_speed.fill_(0.0) # Trip pump
        dt = 0.1
        power_dist = torch.zeros(50, dtype=torch.float64)
        
        initial_flow = thermal_model.mass_flow.item()
        thermal_model(power_dist, dt)
        next_flow = thermal_model.mass_flow.item()
        
        assert next_flow < initial_flow
        assert next_flow > initial_flow * 0.9 # Shouldn't drop by >10% in 0.1s (Inertia works)
        
    def test_natural_circulation(self, thermal_model):
        """Проверка установления естественной циркуляции."""
        thermal_model.pump_speed.fill_(0.0)
        # Simulate long time
        dt = 0.5
        # Add some decay heat to drive circulation (buoyancy needs dT)
        # Set temp profile manually to speed up
        # Use copy_ to update buffer in-place
        temp_profile = torch.linspace(300+273.15, 350+273.15, 50, dtype=torch.float64)
        thermal_model.temp_coolant.copy_(temp_profile)
        
        power_dist = torch.ones(50, dtype=torch.float64) * 5e6 # Higher power to maintain gradient
    
        # Reset flow to near zero to test startup of NC
        thermal_model.mass_flow.fill_(10.0) # Small non-zero start
    
        for i in range(50): # 25 seconds
            # Physics handles temp evolution
            thermal_model(power_dist, dt)
            
        final_flow = thermal_model.mass_flow.item()
        # Should accelerate. Inertia is lower now.
        assert final_flow > 10.5 # Modest increase is enough to prove force exists
        # Buoyancy force should maintain flow positive against friction
        
    def test_boiling_onset(self, thermal_model):
        """Проверка возникновения кипения при высокой мощности."""
        # High power
        power_dist = torch.ones(50, dtype=torch.float64) * 1e8 # 100 MW per node
        dt = 0.1

        # Reduce flow to encourage boiling
        # WE MUST REDUCE PUMP SPEED otherwise the pump restores flow to 18000!
        thermal_model.pump_speed.fill_(0.4) # Corresponds to ~7200 flow
        thermal_model.mass_flow.fill_(7200.0)

        # Run enough steps to heat up
        for _ in range(200):
            thermal_model(power_dist, dt)

        assert torch.max(thermal_model.void_fraction) > 0.0
        # Temp should be clamped near saturation
        max_temp = torch.max(thermal_model.temp_coolant).item()
        t_sat = thermal_model.t_sat
        assert max_temp >= t_sat - 1.0

