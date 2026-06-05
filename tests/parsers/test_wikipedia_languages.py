from pathlib import Path
from typing import Type

import pytest

from tests.utils_test import get_test_data_directory  # noqa: F401
from usas_evaluation_framework.data_utils import load_usas_mapper
from usas_evaluation_framework.parsers.danish_wikipedia import DanishWikipediaUSAS
from usas_evaluation_framework.parsers.dutch_wikipedia import DutchWikipediaUSAS
from usas_evaluation_framework.parsers.english_wikipedia import EnglishWikipediaUSAS
from usas_evaluation_framework.parsers.hindi_wikipedia import HindiWikipediaUSAS
from usas_evaluation_framework.parsers.spanish_wikipedia import SpanishWikipediaUSAS
from usas_evaluation_framework.parsers.wikipedia import WikipediaUSAS

# (parser_class, language, corpus_subdir, corpus_filename,
#  expected_sentences, expected_tokens, expected_semantic_tags, expected_multi_tags)
LANGUAGE_PARAMS = [
    pytest.param(
        EnglishWikipediaUSAS, "English",
        "english_wikipedia", "english_wikipedia_corpus.csv",
        166, 4036, 3505, 0,
        id="English",
    ),
    pytest.param(
        SpanishWikipediaUSAS, "Spanish",
        "spanish_wikipedia", "spanish_wikipedia_corpus.csv",
        30, 1533, 1390, 56,
        id="Spanish",
    ),
    pytest.param(
        DutchWikipediaUSAS, "Dutch",
        "dutch_wikipedia", "dutch_wikipedia_corpus.csv",
        58, 1088, 944, 32,
        id="Dutch"
    ),
    pytest.param(
        DanishWikipediaUSAS, "Danish",
        "danish_wikipedia", "danish_wikipedia_corpus.csv",
        58, 1104, 949, 5,
        id="Danish"
    ),
    pytest.param(
        HindiWikipediaUSAS, "Hindi",
        "hindi_wikipedia", "hindi_wikipedia_corpus.csv",
        80, 2013, 1810, 2,
        id="Hindi"
    )
]


class TestLanguageSpecificWikipediaUSAS:
    """
    Verifies that each language-specific WikipediaUSAS subclass:
      - is a subclass of WikipediaUSAS
      - pre-fills dataset_name and language with the correct defaults
      - allows those defaults to be overridden by the caller
      - parses its full corpus with the expected sentence/token counts
    """

    @pytest.fixture
    def wikipedia_dir(self, get_test_data_directory: Path) -> Path:  # noqa: F811
        return get_test_data_directory / "parsers" / "wikipedia"

    # ------------------------------------------------------------------
    # Subclass / default / override checks (no corpus data needed)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("parser_cls,language,_sd,_cf,_s,_t,_st,_mt", LANGUAGE_PARAMS)
    def test_is_wikipedia_usas_subclass(
        self, parser_cls: Type[WikipediaUSAS], language: str,
        _sd, _cf, _s, _t, _st, _mt,
    ) -> None:
        assert issubclass(parser_cls, WikipediaUSAS)

    @pytest.mark.parametrize("parser_cls,language,_sd,_cf,_s,_t,_st,_mt", LANGUAGE_PARAMS)
    def test_default_dataset_name(
        self, parser_cls: Type[WikipediaUSAS], language: str,
        _sd, _cf, _s, _t, _st, _mt,
        wikipedia_dir: Path,
    ) -> None:
        dataset = parser_cls.parse(wikipedia_dir / "wikipedia_empty.csv")
        assert dataset.name == "Medical Wikipedia"

    @pytest.mark.parametrize("parser_cls,language,_sd,_cf,_s,_t,_st,_mt", LANGUAGE_PARAMS)
    def test_default_language(
        self, parser_cls: Type[WikipediaUSAS], language: str,
        _sd, _cf, _s, _t, _st, _mt,
        wikipedia_dir: Path,
    ) -> None:
        dataset = parser_cls.parse(wikipedia_dir / "wikipedia_empty.csv")
        assert dataset.language == language

    @pytest.mark.parametrize("parser_cls,language,_sd,_cf,_s,_t,_st,_mt", LANGUAGE_PARAMS)
    def test_defaults_can_be_overridden(
        self, parser_cls: Type[WikipediaUSAS], language: str,
        _sd, _cf, _s, _t, _st, _mt,
        wikipedia_dir: Path,
    ) -> None:
        dataset = parser_cls.parse(
            wikipedia_dir / "wikipedia_empty.csv",
            dataset_name="Custom Name",
            language="Catalan",
        )
        assert dataset.name == "Custom Name"
        assert dataset.language == "Catalan"

    # ------------------------------------------------------------------
    # Full corpus smoke tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "parser_cls,language,corpus_subdir,corpus_filename,"
        "expected_sentences,expected_tokens,expected_semantic_tags,expected_multi_tags",
        LANGUAGE_PARAMS,
    )
    def test_parse_full_dataset(
        self,
        parser_cls: Type[WikipediaUSAS],
        language: str,
        corpus_subdir: str,
        corpus_filename: str,
        expected_sentences: int,
        expected_tokens: int,
        expected_semantic_tags: int,
        expected_multi_tags: int,
        get_test_data_directory: Path,  # noqa: F811
    ) -> None:
        data_file = get_test_data_directory / "parsers" / corpus_subdir / corpus_filename
        usas_mapper = load_usas_mapper(None, None)
        valid_usas_tags = set(usas_mapper.keys())
        tags_to_filter: set[str] = set()

        dataset = parser_cls.parse(data_file, valid_usas_tags, tags_to_filter)

        assert dataset.name == "Medical Wikipedia"
        assert dataset.language == language
        assert dataset.text_level == "sentence"
        assert len(dataset.texts) == expected_sentences

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

        assert token_count == expected_tokens
        assert semantic_tag_count == expected_semantic_tags
        assert multi_tag_count == expected_multi_tags
