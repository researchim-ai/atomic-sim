"""
Pytest configuration and fixtures
"""

import numpy as np
import pytest
import torch


def pytest_collection_modifyitems(items):
    long_files = {
        "test_simulator.py",
        "test_integration.py",
        "test_transient_convergence.py",
        "test_muscl_total_initialization.py",
    }
    for item in items:
        if item.name == "test_openmc_particle_run_reports_k_and_mgxs":
            item.add_marker(pytest.mark.external_integration)
        if item.path.name in long_files:
            item.add_marker(pytest.mark.long_running)


@pytest.fixture
def seed():
    """Фиксированный seed для воспроизводимости"""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def device():
    """Устройство для тестов"""
    return "cpu"


@pytest.fixture
def dtype():
    """Тип данных для тестов"""
    return torch.float64
