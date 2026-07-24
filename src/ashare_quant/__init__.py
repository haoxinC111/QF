"""A-share multi-factor research backtester."""

from .config import AppConfig
from .data import MarketDataBundle
from .pit_data import PointInTimeDataBundle

__all__ = ["AppConfig", "MarketDataBundle", "PointInTimeDataBundle"]
__version__ = "2.0.0a12"
