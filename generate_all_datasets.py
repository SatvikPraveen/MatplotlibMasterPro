#!/usr/bin/env python3
"""Regenerate the bundled CSV datasets (thin wrapper around ``mplmasterpro.datasets``).

Usage::

    python generate_all_datasets.py            # writes to ./datasets
    python generate_all_datasets.py --out tmp  # custom directory
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mplmasterpro.datasets import _main

if __name__ == "__main__":
    raise SystemExit(_main())
