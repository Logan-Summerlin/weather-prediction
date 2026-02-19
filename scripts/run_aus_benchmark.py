#!/usr/bin/env python3
"""Austin Model Benchmark — backward-compatible wrapper.

Delegates to the unified ``run_benchmark.py --city aus``.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.argv = [sys.argv[0], "--city", "aus"]
from scripts.run_benchmark import main
if __name__ == "__main__":
    main()
