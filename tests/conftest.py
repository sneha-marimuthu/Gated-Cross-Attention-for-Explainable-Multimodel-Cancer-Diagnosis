"""
Pytest configuration and environment fixtures for GCAEMMCD.
Ensures repository root is always on sys.path and working directory is root.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)
