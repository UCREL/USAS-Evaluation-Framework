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
from usas_evaluation_framework.parsers.torch import TorchParser


class TestTorchParser:

    @pytest.fixture
    def get_test_torch_directory(self, get_test_data_directory: Path) -> Path:  # noqa: F811
        return get_test_data_directory / "parsers" / "torch"

    def test_parse_one_token(self, get_test_torch_directory: Path) -> None:
        one_token_data_file = get_test_torch_directory / "torch_one_token.csv"
        dataset = TorchParser.parse(one_token_data_file)
        assert dataset.name == "Torch"
        assert dataset.text_level == "sentence"

        assert len(dataset.texts) == 1
        assert dataset.texts[0].text == "新华社"
        assert dataset.texts[0].tokens == ["新华社"]
        assert dataset.texts[0].lemmas is None
        assert dataset.texts[0].pos_tags is None
        assert dataset.texts[0].semantic_tags == ["Z3"]
        assert dataset.texts[0].mwe_indexes == [frozenset({})]

    def test_parse_wrong_format(self, get_test_torch_directory: Path) -> None:
        wrong_format_data_file = get_test_torch_directory / "torch_wrong_format.csv"
        with pytest.raises(ValueError):
            TorchParser.parse(wrong_format_data_file)

    @pytest.fixture(params=[None, set(["T1"])])
    def small_example_expected_data_with_label_filter(self, request: pytest.FixtureRequest) -> tuple[EvaluationDataset, None | set[str]]:
        semantic_tags: list[list[str]] = [
            ["Z3", "PUNCT"],
            ["T1", "I1.1/I3.1"]
        ]
        if request.param is not None:
            semantic_tags = [
            ["Z3", "PUNCT"],
            ["", "I1.1/I3.1"]
        ]
        
        evaluation_texts: list[EvaluationTexts] = [
            EvaluationTexts(
                text="新华社 。",
                tokens=["新华社", "。"],
                lemmas=None,
                pos_tags=None,
                semantic_tags=semantic_tags[0],
                mwe_indexes=[frozenset({})] * 2
            ),
            EvaluationTexts(
                text="18日 记者",
                tokens=["18日", "记者"],
                lemmas=None,
                pos_tags=None,
                semantic_tags=semantic_tags[1],
                mwe_indexes=[frozenset({})] * 2
            )
        ]
        return EvaluationDataset(
            name="Torch",
            text_level=TextLevel.sentence,
            labels_removed=request.param,
            texts=evaluation_texts
        ), request.param

    def test_parse_small_example(self,
                                 get_test_torch_directory: Path,
                                 small_example_expected_data_with_label_filter: tuple[EvaluationDataset, None | set[str]]) -> None:
        data_file_name = "torch_small_example.csv"
        data_file = get_test_torch_directory / data_file_name
        small_example_expected_data, label_filter = small_example_expected_data_with_label_filter
        dataset = TorchParser.parse(data_file, label_filter=label_filter)

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
            "Z3", "T1", "I1.1", "I3.1" 
        ])
        if request.param:
            return all_labels, False
        else:
            one_less_label = copy.deepcopy(all_labels)
            one_less_label.remove("I1.1")
            return one_less_label, True

    @pytest.mark.parametrize("label_filter", [None, set(["I1.1"])])
    def test_parse_label_validation(self,
                                    get_test_torch_directory: Path,
                                    label_filter: None | set[str],
                                    small_label_validation_and_error: tuple[set[str], bool]) -> None:
        data_file = get_test_torch_directory / "torch_small_example.csv"
        validation_labels, to_error = small_label_validation_and_error
        if to_error:
            with pytest.raises(ValueError):
                TorchParser.parse(data_file, label_validation=validation_labels, label_filter=label_filter)
        else:
            dataset = TorchParser.parse(data_file, label_validation=validation_labels, label_filter=label_filter)
            assert len(dataset.texts) == 2
            assert dataset.labels_removed == label_filter

    def test_parse_empty_text(self, get_test_torch_directory: Path) -> None:
        text_as_label_file = get_test_torch_directory / "torch_empty_token.csv"
        with pytest.raises(ValueError):
            TorchParser.parse(text_as_label_file)
    
    def test_parse_text_is_usas_label(self, get_test_torch_directory: Path) -> None:
        text_as_label_file = get_test_torch_directory / "torch_text_as_label.csv"
        with pytest.raises(ValueError):
            TorchParser.parse(text_as_label_file)

    def test_parse_usas_label_is_valid(self, get_test_torch_directory: Path) -> None:
        text_as_label_file = get_test_torch_directory / "torch_usas_label_is_valid.csv"
        with pytest.raises(ValueError):
            TorchParser.parse(text_as_label_file)

    def test_parse_no_correct_use_predicted(self, get_test_torch_directory: Path) -> None:
        no_correct_use_predicted_file = get_test_torch_directory / "torch_predicted_punct_tag.csv"
        dataset = TorchParser.parse(no_correct_use_predicted_file)
        assert len(dataset.texts) == 1
        assert dataset.texts[0].text == "新华社 。"
        assert dataset.texts[0].lemmas is None
        assert dataset.texts[0].pos_tags is None
        assert dataset.texts[0].tokens == ["新华社", "。"]
        assert dataset.texts[0].semantic_tags == ["Z3", "PUNCT"]
        assert dataset.texts[0].mwe_indexes == [frozenset({})] * 2

    def test_parse_full_dataset(self, get_test_torch_directory: Path) -> None:
        data_file = get_test_torch_directory / "torch_corpus.csv"
        usas_mapper = load_usas_mapper(None, None)
        valid_usas_tags = set(usas_mapper.keys())
        tags_to_filter = set({"Z99"})
        dataset = TorchParser.parse(data_file, valid_usas_tags, tags_to_filter)
        number_texts = len(dataset.texts)
        assert number_texts == 46
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
        assert token_count == 2312
        assert semantic_tags == 1756
        assert multi_tag == 1

