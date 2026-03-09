"""Test package initialization."""

# This file makes the tests directory a Python package
# and can be used for shared test fixtures or configuration

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
