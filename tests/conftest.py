"""Shared pytest fixtures and path setup."""
import sys
import os

# Make fe_jax importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def yaml_1dshell_0():
    path = os.path.join(DATA_DIR, "1Dshell_0.yaml")
    if not os.path.exists(path):
        pytest.skip(f"Test data not found: {path}")
    return path
