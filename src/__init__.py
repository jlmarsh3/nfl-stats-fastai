"""Utility package exposing feature engineering and splits.
Ensures local imports work when launching apps from subdirectories.
"""
from .features import build_features  # noqa: F401
from .splits import time_splits  # noqa: F401
