"""Compatibility stub for Philadelphia legacy config module."""

from src.city_config import get_city_runtime_config

_CFG = get_city_runtime_config("phl")
globals().update(vars(_CFG))
