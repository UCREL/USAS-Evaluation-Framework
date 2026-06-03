from pathlib import Path

import pytest

from tests.utils_test import get_test_data_directory  # noqa: F401
from usas_evaluation_framework.parsers.spanish_wikipedia import SpanishWikipediaUSAS
from usas_evaluation_framework.parsers.wikipedia import WikipediaUSAS

EXPECTED_DATASET_NAME = "Medical Wikipedia"
EXPECTED_LANGUAGE = "Spanish"


class TestSpanishWikipediaUSASDefaults:
    """
    Verifies that SpanishWikipediaUSAS pre-fills dataset_name and language
    with the correct Spanish corpus defaults.  Full parsing behaviour is
    covered by TestWikipediaUSAS in test_wikipedia.py.
    """

    @pytest.fixture
    def get_test_directory(self, get_test_data_directory: Path) -> Path:  # noqa: F811
        return get_test_data_directory / "parsers" / "spanish_wikipedia"

    def test_is_wikipedia_usas_subclass(self) -> None:
        """SpanishWikipediaUSAS must be a subclass of WikipediaUSAS."""
        assert issubclass(SpanishWikipediaUSAS, WikipediaUSAS)

    def test_default_dataset_name(self, get_test_directory: Path) -> None:
        """Default dataset_name is 'Medical Wikipedia'."""
        dataset = SpanishWikipediaUSAS.parse(
            get_test_directory / "spanish_wikipedia_empty.csv"
        )
        assert dataset.name == EXPECTED_DATASET_NAME

    def test_default_language(self, get_test_directory: Path) -> None:
        """Default language is 'Spanish'."""
        dataset = SpanishWikipediaUSAS.parse(
            get_test_directory / "spanish_wikipedia_empty.csv"
        )
        assert dataset.language == EXPECTED_LANGUAGE

    def test_defaults_can_be_overridden(self, get_test_directory: Path) -> None:
        """Caller-supplied dataset_name and language override the defaults."""
        dataset = SpanishWikipediaUSAS.parse(
            get_test_directory / "spanish_wikipedia_empty.csv",
            dataset_name="Custom Name",
            language="Catalan",
        )
        assert dataset.name == "Custom Name"
        assert dataset.language == "Catalan"
