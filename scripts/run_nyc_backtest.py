#!/usr/bin/env python3
"""NYC Backtest — backward-compatible wrapper.

Delegates to the unified ``run_backtest.py --city nyc``.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.argv = [sys.argv[0], "--city", "nyc"]
from scripts.run_backtest import main
if __name__ == "__main__":
    main()
