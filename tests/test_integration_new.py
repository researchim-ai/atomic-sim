import pytest
import torch
from reactor.simulator_1d import ReactorSimulator1D

class TestIntegrationNew:
    def test_full_simulation_step(self):
        """Проверка полного цикла симуляции."""
        sim = ReactorSimulator1D(config_name='vver')
        
        # Run one step
        state = sim.step(dt=0.1)
        
        # Check keys exist
        assert 'turbine_power_mw' in state
        assert 'sg_pressure' in state
        assert 'mass_flow' in state
        
        # Check values reasonable
        # At t=0 power might be ramping up, just check no NaN
        assert not torch.isnan(torch.tensor(state['thermal_power_mw']))
        assert 5.0e6 < state['sg_pressure'] < 8.0e6
        
    def test_scram_scenario(self):
        """Сценарий аварийной защиты (SCRAM)."""
        sim = ReactorSimulator1D(config_name='vver')
        
        # Steady state run
        for _ in range(10): sim.step(dt=0.1)
        p_init = sim.neutronics.phi.mean().item()
        
        # SCRAM
        sim.control.rod_position = 1.0 
        
        # Run 2 seconds
        for _ in range(20): sim.step(dt=0.1)
        
        p_final = sim.neutronics.phi.mean().item()
        
        # Power should drop significantly
        assert p_final < p_init * 0.5

