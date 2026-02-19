#!/usr/bin/env python3
"""Austin Synthesis Calibration — backward-compatible wrapper.

Delegates to the unified ``run_synthesis_calibration.py --city aus``.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.argv = [sys.argv[0], "--city", "aus"]
from scripts.run_synthesis_calibration import main
if __name__ == "__main__":
    main()
