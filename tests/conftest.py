"""
Pytest configuration and fixtures
"""

import numpy as np
import pytest
import torch


def pytest_collection_modifyitems(items):
    from atomic_model_release.lanes import classify

    marker = {
        "openmc": "external_integration",
        "hardware": "hardware_specific",
        "long": "long_running",
    }
    for item in items:
        lane = classify(item.path, item.name)
        if lane in marker:
            item.add_marker(getattr(pytest.mark, marker[lane]))


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
