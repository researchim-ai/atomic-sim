"""
Нелинейная кривая ценности управляющих стержней
Более реалистичная модель зависимости реактивности от позиции
"""

import torch


def rod_worth_curve(position: torch.Tensor, shape: float = 2.2) -> torch.Tensor:
    """
    S-образная кривая worth для управляющих стержней
    
    В реальных реакторах кривая ценности стержней нелинейна:
    - Наибольшая ценность в средней части активной зоны
    - Меньшая эффективность на краях
    
    Args:
        position: позиция стержня (0 = вставлен, 1 = извлечен)
        shape: параметр формы кривой (> 1 для S-образной)
    
    Returns:
        относительная ценность: 0 (полностью извлечен) до 1 (полностью вставлен)
    """
    # f(1) = 0 (извлечен - нет реактивности)
    # f(0) = 1 (вставлен - максимальная отрицательная реактивность)
    return 1.0 - torch.pow(position, shape)


def differential_worth(position: torch.Tensor, shape: float = 2.2) -> torch.Tensor:
    """
    Дифференциальная ценность (производная worth по позиции)
    
    Показывает эффективность движения стержня в данной точке
    
    Args:
        position: позиция стержня
        shape: параметр формы кривой
    
    Returns:
        дифференциальная ценность
    """
    return -shape * torch.pow(position, shape - 1.0)


def integral_worth(position: torch.Tensor, shape: float = 2.2) -> torch.Tensor:
    """
    Интегральная ценность стержня от полностью вставленного до текущей позиции
    
    Args:
        position: позиция стержня
        shape: параметр формы
    
    Returns:
        интегральная ценность
    """
    # Интеграл от 0 до position функции (1 - x^shape)
    return position - torch.pow(position, shape + 1.0) / (shape + 1.0)

