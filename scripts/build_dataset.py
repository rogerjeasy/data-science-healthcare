#!/usr/bin/env python3
"""Step 1: merge and harmonise the raw NHANES files into data/processed/.

    python scripts/build_dataset.py

Run `python scripts/fetch_data.py` first. The logic lives in src/hidden_risk/.
"""
import sys

from hidden_risk.build import main

if __name__ == "__main__":
    sys.exit(main())
