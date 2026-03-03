from pathlib import Path

import pytest


def _get_test_data_directory() -> Path:
    return Path(__file__).parent / "data"

@pytest.fixture
def get_test_data_directory() -> Path:
    return _get_test_data_directory()
