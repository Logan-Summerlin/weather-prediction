#!/usr/bin/env python3
"""Philadelphia Preprocessing Pipeline — backward-compatible wrapper.

Delegates to the unified ``run_preprocessing.py --city phl``.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.argv = [sys.argv[0], "--city", "phl"]
from scripts.run_preprocessing import main
if __name__ == "__main__":
    main()
