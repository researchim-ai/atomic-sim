"""
Pytest configuration and fixtures
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def seed():
    """Фиксированный seed для воспроизводимости"""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def device():
    """Устройство для тестов"""
    return 'cpu'


@pytest.fixture
def dtype():
    """Тип данных для тестов"""
    return torch.float64

