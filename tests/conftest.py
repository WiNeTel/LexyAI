"""Pytest configuration for Lexy AI."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `import lexy_core` works.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
