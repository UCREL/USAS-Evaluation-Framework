from pathlib import Path

import pytest

from tests.utils_test import get_test_data_directory  # noqa: F401
from usas_evaluation_framework.data_utils import load_usas_mapper
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


    @pytest.fixture
    def get_wikipedia_test_directory(self, get_test_data_directory: Path) -> Path:  # noqa: F811
        # Wikipedia data contains Spanish data and we have also use this to
        # generally test the functionality of Wikipedia USAS data.
        return get_test_data_directory / "parsers" / "wikipedia"

    def test_is_wikipedia_usas_subclass(self) -> None:
        """SpanishWikipediaUSAS must be a subclass of WikipediaUSAS."""
        assert issubclass(SpanishWikipediaUSAS, WikipediaUSAS)

    def test_default_dataset_name(self, get_wikipedia_test_directory: Path) -> None:
        """Default dataset_name is 'Medical Wikipedia'."""
        dataset = SpanishWikipediaUSAS.parse(
            get_wikipedia_test_directory / "wikipedia_empty.csv"
        )
        assert dataset.name == EXPECTED_DATASET_NAME

    def test_default_language(self, get_wikipedia_test_directory: Path) -> None:
        """Default language is 'Spanish'."""
        dataset = SpanishWikipediaUSAS.parse(
            get_wikipedia_test_directory / "wikipedia_empty.csv"
        )
        assert dataset.language == EXPECTED_LANGUAGE

    def test_defaults_can_be_overridden(self, get_wikipedia_test_directory: Path) -> None:
        """Caller-supplied dataset_name and language override the defaults."""
        dataset = SpanishWikipediaUSAS.parse(
            get_wikipedia_test_directory / "wikipedia_empty.csv",
            dataset_name="Custom Name",
            language="Catalan",
        )
        assert dataset.name == "Custom Name"
        assert dataset.language == "Catalan"


    # ------------------------------------------------------------------
    # Full corpus smoke test
    # ------------------------------------------------------------------

    def test_parse_full_dataset(self, get_test_directory: Path) -> None:
        """Parse the full spanish.csv corpus and verify sentence/token counts."""
        data_file = get_test_directory / "spanish_wikipedia_corpus.csv"
        usas_mapper = load_usas_mapper(None, None)
        valid_usas_tags = set(usas_mapper.keys())
        tags_to_filter: set[str] = set()
        dataset = WikipediaUSAS.parse(
            data_file,
            valid_usas_tags,
            tags_to_filter,
            dataset_name="Medical Wikipedia",
            language="Spanish",
        )

        assert dataset.name == "Medical Wikipedia"
        assert dataset.language == "Spanish"
        assert dataset.text_level == "sentence"
        # 21 cancer + 6 chemotherapy + 3 melanoma sentences
        assert len(dataset.texts) == 30

        token_count = 0
        semantic_tag_count = 0
        multi_tag_count = 0
        for text in dataset.texts:
            assert text.tokens is not None
            assert text.lemmas is not None
            assert text.pos_tags is not None
            assert text.semantic_tags is not None
            assert text.mwe_indexes is not None
            token_count += len(text.tokens)
            for tag_list in text.semantic_tags:
                assert len(tag_list) == 1

                if tag_list[0] and tag_list[0] != "Z9":
                    semantic_tag_count += 1
                
                if "/" in tag_list[0]:
                    multi_tag_count += 1

        assert token_count == 1533
        assert semantic_tag_count == 1390
        assert multi_tag_count == 56
