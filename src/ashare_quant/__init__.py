"""A-share multi-factor research backtester."""

from .config import AppConfig
from .data import MarketDataBundle

__all__ = ["AppConfig", "MarketDataBundle"]
__version__ = "1.6.0"
