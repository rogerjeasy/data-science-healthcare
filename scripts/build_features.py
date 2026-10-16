#!/usr/bin/env python3
"""Step 2: write data/processed/nhanes_2005_2018_features.parquet.

    python scripts/build_features.py
"""
import sys

from hidden_risk.features import main

if __name__ == "__main__":
    sys.exit(main())