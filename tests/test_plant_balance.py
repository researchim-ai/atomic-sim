import pytest
from reactor.plant_balance import PlantBalance

class TestPlantBalance:
    
    @pytest.fixture
    def plant(self):
        return PlantBalance(dt=0.1)

    def test_initial_steady_state(self, plant):
        """Проверка, что при номинальных параметрах система стабильна."""
        # Nominal VVER: Hot Leg ~320C, Flow ~18000 kg/s
        hot_leg = 320.0 
        flow = 18000.0
        
        cold_leg = plant.step(hot_leg, flow, dt=0.1)
        
        # Delta T should be around 30C (320 -> 290)
        # 3000 MW = 18000 * 4200 * dT => dT = 3000e6 / (75.6e6) = 39.6 C
        # So Cold Leg ~ 320 - 40 = 280
        assert 270.0 < cold_leg < 310.0
        
        # Steam pressure should be stable around 6.4 MPa
        assert 6.0e6 < plant.sg_pressure < 7.0e6
        
        # Electric power ~ 1000 MW
        # Efficiency 33% of 3000 MW th
        assert 900.0 < plant.electric_power < 1100.0

    def test_turbine_load_rejection(self, plant):
        """Сброс нагрузки турбины должен повышать давление."""
        hot_leg = 320.0 
        flow = 18000.0
        
        # Close turbine valve
        plant.turbine_throttle = 0.0
        
        initial_pressure = plant.sg_pressure
        plant.step(hot_leg, flow, dt=1.0)
        
        # Energy is entering SG but not leaving -> P rises
        assert plant.sg_pressure > initial_pressure
        assert plant.steam_flow == 0.0
        assert plant.electric_power == 0.0

    def test_feedwater_control(self, plant):
        """Автоматика уровня должна работать."""
        plant.sg_level = -0.5 # Low level
        plant.level_pid_integral = 0.0 # Start fresh
    
        # Run a few steps
        for _ in range(10):
            plant.step(320.0, 18000.0, dt=0.1)
    
        # Controller should increase feedwater flow > steam flow to fill up
        # Nominal steam flow ~ 1600
        # Due to transient, values might fluctuate, but FW should be higher than Steam eventually
        # Check the trend or just that it's responding positively
        assert plant.feedwater_flow > plant.steam_flow # Just check direction

