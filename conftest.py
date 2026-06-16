"""Ensure the repository root is importable so `import src...` works under pytest
regardless of how pytest is invoked (`pytest`, `python -m pytest`, from any cwd).
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
