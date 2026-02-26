import copy
from pathlib import Path

import pytest

from tests.utils_test import get_test_data_directory  # noqa: F401
from usas_evaluation_framework.data_utils import load_usas_mapper
from usas_evaluation_framework.dataset import (
    EvaluationDataset,
    EvaluationTexts,
    TextLevel,
)
from usas_evaluation_framework.parsers.corcencc import CorcenccParser


class TestCorcenccParser:

    @pytest.fixture
    def get_test_corcencc_directory(self, get_test_data_directory: Path) -> Path:  # noqa: F811
        return get_test_data_directory / "parsers" / "corcencc"

    def test_parse_one_token(self, get_test_corcencc_directory: Path) -> None:
        one_token_data_file = get_test_corcencc_directory / "corcencc_one_token.txt"
        dataset = CorcenccParser.parse(one_token_data_file)
        assert dataset.name == "Corcencc"
        assert dataset.text_level == "sentence"

        assert len(dataset.texts) == 1
        assert dataset.texts[0].text == "Ceisiwch"
        assert dataset.texts[0].tokens == ["Ceisiwch"]
        assert dataset.texts[0].lemmas is None
        assert dataset.texts[0].pos_tags is None
        assert dataset.texts[0].semantic_tags == ["X8"]
        assert dataset.texts[0].mwe_indexes == [frozenset({})]

    def test_parse_wrong_format(self, get_test_corcencc_directory: Path) -> None:
        wrong_format_data_file = get_test_corcencc_directory / "corcencc_wrong_format.txt"
        with pytest.raises(ValueError):
            CorcenccParser.parse(wrong_format_data_file)

    @pytest.fixture(params=[None, set(["A5.4"])])
    def small_example_expected_data_with_label_filter(self, request: pytest.FixtureRequest) -> tuple[EvaluationDataset, None | set[str]]:
        semantic_tags: list[list[str]] = [
            ["A5.4", "Z5", "A5.1", "Q2.1", "PUNCT"],
            ["X6", "N5.1/A5.4", "Z5", "PUNCT"]
        ]
        if request.param is not None:
            semantic_tags = [
            ["", "Z5", "A5.1", "Q2.1", "PUNCT"],
            ["X6", "N5.1/A5.4", "Z5", "PUNCT"]
        ]
        
        evaluation_texts: list[EvaluationTexts] = [
            EvaluationTexts(
                text="Mae 'n dda sgwrsio !",
                tokens=["Mae", "'n", "dda", "sgwrsio", "!"],
                lemmas=None,
                pos_tags=None,
                semantic_tags=semantic_tags[0],
                mwe_indexes=[frozenset({})] * 5
            ),
            EvaluationTexts(
                text="Dewiswch templed arall .",
                tokens=["Dewiswch", "templed", "arall", "."],
                lemmas=None,
                pos_tags=None,
                semantic_tags=semantic_tags[1],
                mwe_indexes=[frozenset({})] * 4
            )
        ]
        return EvaluationDataset(
            name="Corcencc",
            text_level=TextLevel.sentence,
            labels_removed=request.param,
            texts=evaluation_texts
        ), request.param

    def test_parse_small_example(self,
                                 get_test_corcencc_directory: Path,
                                 small_example_expected_data_with_label_filter: tuple[EvaluationDataset, None | set[str]]) -> None:
        data_file_name = "corcencc_small_example.txt"
        data_file = get_test_corcencc_directory / data_file_name
        small_example_expected_data, label_filter = small_example_expected_data_with_label_filter
        dataset = CorcenccParser.parse(data_file, label_filter=label_filter)

        expected_number_of_texts = 2
        assert len(dataset.texts) == expected_number_of_texts
        for text_index in range(expected_number_of_texts):
            assert dataset.texts[text_index].text == small_example_expected_data.texts[text_index].text
            assert dataset.texts[text_index].tokens == small_example_expected_data.texts[text_index].tokens
            assert dataset.texts[text_index].lemmas == small_example_expected_data.texts[text_index].lemmas
            assert dataset.texts[text_index].pos_tags == small_example_expected_data.texts[text_index].pos_tags
            assert dataset.texts[text_index].semantic_tags == small_example_expected_data.texts[text_index].semantic_tags
            assert dataset.texts[text_index].mwe_indexes == small_example_expected_data.texts[text_index].mwe_indexes
        assert dataset.labels_removed == label_filter

    @pytest.fixture(params=[False, True])
    def small_label_validation_and_error(self, request: pytest.FixtureRequest) -> tuple[set[str], bool]:
        all_labels = set([
            "A5.4", "Z5", "A5.1", "Q2.1", "X6", "N5.1" 
        ])
        if request.param:
            return all_labels, False
        else:
            one_less_label = copy.deepcopy(all_labels)
            one_less_label.remove("N5.1")
            return one_less_label, True

    @pytest.mark.parametrize("label_filter", [None, set(["N5.1"])])
    def test_parse_label_validation(self,
                                    get_test_corcencc_directory: Path,
                                    label_filter: None | set[str],
                                    small_label_validation_and_error: tuple[set[str], bool]) -> None:
        data_file = get_test_corcencc_directory / "corcencc_small_example.txt"
        validation_labels, to_error = small_label_validation_and_error
        if to_error:
            with pytest.raises(ValueError):
                CorcenccParser.parse(data_file, label_validation=validation_labels, label_filter=label_filter)
        else:
            dataset = CorcenccParser.parse(data_file, label_validation=validation_labels, label_filter=label_filter)
            assert len(dataset.texts) == 2
            assert dataset.labels_removed == label_filter

    def test_parse_empty_text(self, get_test_corcencc_directory: Path) -> None:
        text_as_label_file = get_test_corcencc_directory / "corcencc_empty_token.txt"
        with pytest.raises(ValueError):
            CorcenccParser.parse(text_as_label_file)
    
    def test_parse_text_is_usas_label(self, get_test_corcencc_directory: Path) -> None:
        text_as_label_file = get_test_corcencc_directory / "corcencc_text_as_label.txt"
        with pytest.raises(ValueError):
            CorcenccParser.parse(text_as_label_file)

    def test_parse_usas_label_is_valid(self, get_test_corcencc_directory: Path) -> None:
        text_as_label_file = get_test_corcencc_directory / "corcencc_usas_label_is_valid.txt"
        with pytest.raises(ValueError):
            CorcenccParser.parse(text_as_label_file)

    def test_parse_full_dataset(self, get_test_corcencc_directory: Path) -> None:
        data_file = get_test_corcencc_directory / "corcencc_corpus.txt"
        usas_mapper = load_usas_mapper(None, None)
        valid_usas_tags = set(usas_mapper.keys())
        tags_to_filter = set({"Z99"})
        dataset = CorcenccParser.parse(data_file, valid_usas_tags, tags_to_filter)
        number_texts = len(dataset.texts)
        assert number_texts == 611
        assert dataset.labels_removed == tags_to_filter

        token_count = 0
        semantic_tags = 0
        multi_tag = 0
        for text in dataset.texts:
            token_count += len(text.tokens)
            assert text.semantic_tags is not None
            for semantic_tag in text.semantic_tags:
                if semantic_tag != "PUNCT" and semantic_tag:
                    semantic_tags += 1
                if "/" in semantic_tag:
                    multi_tag += 1
        assert token_count == 14876
        assert semantic_tags == 12803
        assert multi_tag == 1314

