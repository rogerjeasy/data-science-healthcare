#!/usr/bin/env python3
"""Write docs/results.md from the step 3-6 tables.

    python scripts/write_results.py
"""
import sys

from hidden_risk.summary import main

if __name__ == "__main__":
    sys.exit(main())
