from pathlib import Path

import pytest


@pytest.fixture
def get_test_data_directory() -> Path:
    return Path(__file__).parent / "data"
