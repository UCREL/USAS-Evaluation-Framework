from pathlib import Path

import pytest

from tests.utils_test import get_test_data_directory  # noqa: F401
from usas_evaluation_framework.data_utils import load_usas_mapper
from usas_evaluation_framework.dataset import (
    EvaluationDataset,
    TextLevel,
)
from usas_evaluation_framework.parsers.naacl_2015 import NAACL2015USAS


class TestNAACL2015PortugueseUSAS:

    @pytest.fixture
    def get_test_directory(self, get_test_data_directory: Path) -> Path:  # noqa: F811
        return get_test_data_directory / "parsers" / "naacl_2015_portuguese"


    # ------------------------------------------------------------------
    # Full corpus smoke tests
    # ------------------------------------------------------------------

    def test_parse_full_dataset(self, get_test_directory: Path) -> None:
        data_file = get_test_directory / "naacl_2015_portuguese_corpus.csv"
        usas_mapper = load_usas_mapper(None, None)
        valid_usas_tags = set(usas_mapper.keys())

        dataset = NAACL2015USAS.parse(
            data_file,
            label_validation=valid_usas_tags,
            label_filter=set(),
            language="Portuguese",
        )

        assert isinstance(dataset, EvaluationDataset)
        assert dataset.name == "NAACL 2015"
        assert dataset.language == "Portuguese"
        assert dataset.text_level == TextLevel.sentence
        assert len(dataset.texts) == 39

        token_count = 0
        mwe_token_count = 0
        for text in dataset.texts:
            assert text.tokens is not None
            assert text.lemmas is None
            assert text.pos_tags is None
            assert text.semantic_tags is not None
            assert text.mwe_indexes is not None
            assert len(text.mwe_indexes) == len(text.tokens)
            token_count += len(text.tokens)
            for fs in text.mwe_indexes:
                if fs:
                    mwe_token_count += 1

        assert token_count == 1232
        assert mwe_token_count == 48 # 12

        # MWE Groups

        # Fiction MWEs

        # Newspaper MWEs
        sentence_4 = dataset.texts[23]
        assert sentence_4.mwe_indexes[0] == frozenset({1})
        assert sentence_4.mwe_indexes[1] == frozenset({1})
        assert sentence_4.mwe_indexes[2] == frozenset({1})

        sentence_6 = dataset.texts[25]
        assert sentence_6.mwe_indexes[14] == frozenset({1})
        assert sentence_6.mwe_indexes[15] == frozenset({1})
        assert sentence_6.mwe_indexes[16] == frozenset({1})

        sentence_14 = dataset.texts[33]
        assert sentence_14.mwe_indexes[27] == frozenset({1})
        assert sentence_14.mwe_indexes[28] == frozenset({1})
        assert sentence_14.mwe_indexes[29] == frozenset({1})

        sentence_16 = dataset.texts[35]
        assert sentence_16.mwe_indexes[17] == frozenset({1})
        assert sentence_16.mwe_indexes[18] == frozenset({1})
        assert sentence_16.mwe_indexes[19] == frozenset({1})
