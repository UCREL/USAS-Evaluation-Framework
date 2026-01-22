import re

import usas_evaluation_framework


def test_version() -> None:
    version = usas_evaluation_framework.__version__
    assert isinstance(version, str)
    assert re.search(r"\d+\.\d+\.\d+$", version) is not None
