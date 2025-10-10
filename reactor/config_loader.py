"""
Загрузчик конфигураций из YAML файлов
Позволяет настраивать параметры симулятора из конфигов
"""

import yaml
from pathlib import Path
import torch


class Config:
    """Класс конфигурации симулятора"""
    
    def __init__(self, config_dict=None):
        """
        Args:
            config_dict: словарь с конфигурацией
        """
        self.config = config_dict or {}
    
    def get(self, path, default=None):
        """
        Получить значение по пути (например, 'simulator.device')
        
        Args:
            path: путь к параметру (через точку)
            default: значение по умолчанию
        
        Returns:
            значение параметра
        """
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    @classmethod
    def from_yaml(cls, filepath):
        """
        Загрузить конфигурацию из YAML файла
        
        Args:
            filepath: путь к YAML файлу
        
        Returns:
            Config объект
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)
    
    @classmethod
    def default(cls):
        """
        Загрузить конфигурацию по умолчанию
        
        Returns:
            Config объект
        """
        # Поиск default.yaml
        default_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        
        if default_path.exists():
            return cls.from_yaml(default_path)
        else:
            # Возвращаем пустую конфигурацию
            return cls({})
    
    def to_dict(self):
        """Преобразовать в словарь"""
        return self.config
    
    def __repr__(self):
        return f"Config({self.config})"


def load_config(filepath=None):
    """
    Удобная функция для загрузки конфигурации
    
    Args:
        filepath: путь к файлу конфигурации (None = default)
    
    Returns:
        Config объект
    """
    if filepath is None:
        return Config.default()
    else:
        return Config.from_yaml(filepath)

