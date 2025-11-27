import pytest
import torch
from reactor.neutronics_1d import Neutronics1D
from reactor.reactor_configs import VVER_1000

class TestNeutronics1D:
    @pytest.fixture
    def neutronics(self):
        return Neutronics1D(config=VVER_1000, num_nodes=50, device='cpu')

    def test_criticality(self, neutronics):
        """Реактор должен быть близок к критичности в чистом состоянии."""
        # Rods out, no boron
        neutronics.step(dt=0.01)
        
        # Flux shouldn't change rapidly
        initial_flux = neutronics.phi.sum().item()
        
        # Run for 1 second
        for _ in range(10):
            neutronics.step(dt=0.1)
            
        final_flux = neutronics.phi.sum().item()
        
        # Allow some drift (supercritical at start), but not infinite
        ratio = final_flux / initial_flux
        assert 0.5 < ratio < 10.0

    def test_rod_insertion(self, neutronics):
        """Ввод стержней должен снижать мощность."""
        initial_flux = neutronics.phi.sum().item()
        
        # Insert rods 50%
        # Need enough time for kinetics to respond
        for _ in range(20):
            neutronics.step(dt=0.05, rod_pos=0.5)
            
        final_flux = neutronics.phi.sum().item()
        assert final_flux < initial_flux

    def test_flux_shape(self, neutronics):
        """Форма потока должна быть косинусоидальной (максимум в центре)."""
        neutronics.reset()
        phi = neutronics.phi.cpu().numpy()
        
        center = len(phi) // 2
        edge = 0
        
        assert phi[center] > phi[edge]
        assert phi[center] > phi[-1]

