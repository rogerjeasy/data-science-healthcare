#!/usr/bin/env python3
"""Step 4: tune and fit every model on 2005-2014, predict 2015-2018.

    python scripts/train_models.py
"""
import sys

from hidden_risk.models import main

if __name__ == "__main__":
    sys.exit(main())
