"""Philadelphia pipeline tests — backward-compatible wrapper.

Imports all parameterized tests from test_city_pipeline.py which covers
CHI, PHL, and ATL. Running this file directly or via pytest still works.

To run Philadelphia-specific tests only:
    pytest tests/test_city_pipeline.py -k "phl"
"""
from tests.test_city_pipeline import *  # noqa: F401,F403
