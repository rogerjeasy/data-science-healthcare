#!/usr/bin/env python3
"""Step 5: test-set metrics, bootstrap tests of H1-H4, screen-negative analysis, fairness.

    python scripts/evaluate_models.py
"""
import sys

from hidden_risk.evaluation import main

if __name__ == "__main__":
    sys.exit(main())
