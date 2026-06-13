"""Shared pytest fixtures and path setup."""
import sys
import os

# fe_jax lives under opensg_jax/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "opensg_jax"))

import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _yaml_fixture(name):
    """Return the path to a Shell_1DSG YAML file, skip if missing."""
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"Test data not found: {path}")
    return path


@pytest.fixture
def yaml_1dshell_0():
    return _yaml_fixture("1Dshell_0.yaml")


yaml_1dshell_1  = pytest.fixture(name="yaml_1dshell_1" )(lambda: _yaml_fixture("1Dshell_1.yaml"))
yaml_1dshell_2  = pytest.fixture(name="yaml_1dshell_2" )(lambda: _yaml_fixture("1Dshell_2.yaml"))
yaml_1dshell_3  = pytest.fixture(name="yaml_1dshell_3" )(lambda: _yaml_fixture("1Dshell_3.yaml"))
yaml_1dshell_4  = pytest.fixture(name="yaml_1dshell_4" )(lambda: _yaml_fixture("1Dshell_4.yaml"))
yaml_1dshell_5  = pytest.fixture(name="yaml_1dshell_5" )(lambda: _yaml_fixture("1Dshell_5.yaml"))
yaml_1dshell_6  = pytest.fixture(name="yaml_1dshell_6" )(lambda: _yaml_fixture("1Dshell_6.yaml"))
yaml_1dshell_7  = pytest.fixture(name="yaml_1dshell_7" )(lambda: _yaml_fixture("1Dshell_7.yaml"))
yaml_1dshell_8  = pytest.fixture(name="yaml_1dshell_8" )(lambda: _yaml_fixture("1Dshell_8.yaml"))
yaml_1dshell_9  = pytest.fixture(name="yaml_1dshell_9" )(lambda: _yaml_fixture("1Dshell_9.yaml"))
yaml_1dshell_10 = pytest.fixture(name="yaml_1dshell_10")(lambda: _yaml_fixture("1Dshell_10.yaml"))
yaml_1dshell_11 = pytest.fixture(name="yaml_1dshell_11")(lambda: _yaml_fixture("1Dshell_11.yaml"))
yaml_1dshell_12 = pytest.fixture(name="yaml_1dshell_12")(lambda: _yaml_fixture("1Dshell_12.yaml"))
yaml_1dshell_13 = pytest.fixture(name="yaml_1dshell_13")(lambda: _yaml_fixture("1Dshell_13.yaml"))
yaml_1dshell_14 = pytest.fixture(name="yaml_1dshell_14")(lambda: _yaml_fixture("1Dshell_14.yaml"))
yaml_1dshell_15 = pytest.fixture(name="yaml_1dshell_15")(lambda: _yaml_fixture("1Dshell_15.yaml"))
yaml_1dshell_16 = pytest.fixture(name="yaml_1dshell_16")(lambda: _yaml_fixture("1Dshell_16.yaml"))
yaml_1dshell_17 = pytest.fixture(name="yaml_1dshell_17")(lambda: _yaml_fixture("1Dshell_17.yaml"))
yaml_1dshell_18 = pytest.fixture(name="yaml_1dshell_18")(lambda: _yaml_fixture("1Dshell_18.yaml"))
yaml_1dshell_19 = pytest.fixture(name="yaml_1dshell_19")(lambda: _yaml_fixture("1Dshell_19.yaml"))
yaml_1dshell_20 = pytest.fixture(name="yaml_1dshell_20")(lambda: _yaml_fixture("1Dshell_20.yaml"))
yaml_1dshell_21 = pytest.fixture(name="yaml_1dshell_21")(lambda: _yaml_fixture("1Dshell_21.yaml"))
yaml_1dshell_22 = pytest.fixture(name="yaml_1dshell_22")(lambda: _yaml_fixture("1Dshell_22.yaml"))
yaml_1dshell_23 = pytest.fixture(name="yaml_1dshell_23")(lambda: _yaml_fixture("1Dshell_23.yaml"))
yaml_1dshell_24 = pytest.fixture(name="yaml_1dshell_24")(lambda: _yaml_fixture("1Dshell_24.yaml"))
yaml_1dshell_25 = pytest.fixture(name="yaml_1dshell_25")(lambda: _yaml_fixture("1Dshell_25.yaml"))
yaml_1dshell_26 = pytest.fixture(name="yaml_1dshell_26")(lambda: _yaml_fixture("1Dshell_26.yaml"))
yaml_1dshell_27 = pytest.fixture(name="yaml_1dshell_27")(lambda: _yaml_fixture("1Dshell_27.yaml"))
yaml_1dshell_28 = pytest.fixture(name="yaml_1dshell_28")(lambda: _yaml_fixture("1Dshell_28.yaml"))
yaml_1dshell_29 = pytest.fixture(name="yaml_1dshell_29")(lambda: _yaml_fixture("1Dshell_29.yaml"))
