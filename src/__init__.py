"""Görevi: Python'a "Bu klasör rastgele bir klasör değil,
 projemizin kalbi olan bir modüldür, içindeki dosyalar birbiriyle konuşabilir" der"""
# src/__init__.py
from .preprocessing import DataPreprocessor
from .inference import ModelPredictor

__version__ = "2.0.0"