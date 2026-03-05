import copy
from pathlib import Path

import pytest

from tests.utils_test import (  # noqa: F401
    _get_test_data_directory,
    get_test_data_directory,
)
from usas_evaluation_framework.data_utils import load_usas_mapper
from usas_evaluation_framework.dataset import (
    EvaluationDataset,
    EvaluationTexts,
    TextLevel,
)
from usas_evaluation_framework.parsers.icc_irish import ICCIrishParser


def downloaded_irish_datasets() -> bool:
    """
    Returns True if all the Irish human annotated datasets have been downloaded to
    the directory `tests/test_data/parsers/icc_irish`, the dataset file names
    should be:
    
    * ICC-GA-WPH-001-the_wire.tsv
    * ICC-GA-WPH-003-george_orwell.tsv
    * ICC-GA-WR0-021-tuairisc.tsv
    """

    test_data_directory = _get_test_data_directory()
    irish_dataset_directory = test_data_directory / "parsers" / "icc_irish"

    data_file_names = [
        "ICC-GA-WPH-001-the_wire.tsv",
        "ICC-GA-WPH-003-george_orwell.tsv",
        "ICC-GA-WR0-021-tuairisc.tsv"
    ]
    data_file_path_exist = [
        (irish_dataset_directory / data_file_name).exists() for data_file_name in data_file_names
    ]

    return all(data_file_path_exist)


class TestICCIrishParser:

    @pytest.fixture
    def get_test_icc_irish_directory(self, get_test_data_directory: Path) -> Path:  # noqa: F811
        return get_test_data_directory / "parsers" / "icc_irish"

    def test_parse_one_token(self, get_test_icc_irish_directory: Path) -> None:
        one_token_data_file = get_test_icc_irish_directory / "icc_one_token.tsv"
        dataset = ICCIrishParser.parse(one_token_data_file)
        assert dataset.name == "ICCIrish"
        assert dataset.text_level == "paragraph"

        assert len(dataset.texts) == 1
        assert dataset.texts[0].text == "Is"
        assert dataset.texts[0].tokens == ["Is"]
        assert dataset.texts[0].lemmas is None
        assert dataset.texts[0].pos_tags is None
        assert dataset.texts[0].semantic_tags == [["Z5"]]
        assert dataset.texts[0].mwe_indexes == [frozenset({})]

    def test_parse_wrong_format(self, get_test_icc_irish_directory: Path) -> None:
        wrong_format_data_file = get_test_icc_irish_directory / "icc_wrong_format.tsv"
        with pytest.raises(ValueError):
            ICCIrishParser.parse(wrong_format_data_file)

    @pytest.fixture(params=[None, set(["Q4.3"])])
    def small_example_expected_data_with_label_filter(self, request: pytest.FixtureRequest) -> tuple[EvaluationDataset, None | set[str]]:
        semantic_tags: list[list[list[str]]] = [
            [["Z5"], ["Q4.3/N4"], ["Q4.3"], ["Z8"], ["Z0"], ["Z0"], ["PUNCT"]]
        ]
        if request.param is not None:
            semantic_tags: list[list[list[str]]] = [
            [["Z5"], ["Q4.3/N4"], [""], ["Z8"], ["Z0"], ["Z0"], ["PUNCT"]]
        ]
        
        evaluation_texts: list[EvaluationTexts] = [
            EvaluationTexts(
                text="Is sraith theilifíse í The Simpsons .",
                tokens=["Is", "sraith", "theilifíse", "í", "The", "Simpsons", "."],
                lemmas=None,
                pos_tags=None,
                semantic_tags=semantic_tags[0],
                mwe_indexes=[frozenset({})] * 4 + [frozenset({1}), frozenset({1}), frozenset({})]
            )
        ]
        return EvaluationDataset(
            name="ICCIrish",
            text_level=TextLevel.paragraph,
            labels_removed=request.param,
            texts=evaluation_texts
        ), request.param

    def test_parse_small_example(self,
                                 get_test_icc_irish_directory: Path,
                                 small_example_expected_data_with_label_filter: tuple[EvaluationDataset, None | set[str]]) -> None:
        data_file_name = "icc_small_example.tsv"
        data_file = get_test_icc_irish_directory / data_file_name
        small_example_expected_data, label_filter = small_example_expected_data_with_label_filter
        dataset = ICCIrishParser.parse(data_file, label_filter=label_filter)

        expected_number_of_texts = 1
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
            "Q4.3", "Z5", "N4", "Z8", "Z0", "Z0"
        ])
        if request.param:
            return all_labels, False
        else:
            one_less_label = copy.deepcopy(all_labels)
            one_less_label.remove("Q4.3")
            return one_less_label, True

    @pytest.mark.parametrize("label_filter", [None, set(["N5.1"])])
    def test_parse_label_validation(self,
                                    get_test_icc_irish_directory: Path,
                                    label_filter: None | set[str],
                                    small_label_validation_and_error: tuple[set[str], bool]) -> None:
        data_file = get_test_icc_irish_directory / "icc_small_example.tsv"
        validation_labels, to_error = small_label_validation_and_error
        if to_error:
            with pytest.raises(ValueError):
                ICCIrishParser.parse(data_file, label_validation=validation_labels, label_filter=label_filter)
        else:
            dataset = ICCIrishParser.parse(data_file, label_validation=validation_labels, label_filter=label_filter)
            assert len(dataset.texts) == 1
            assert dataset.labels_removed == label_filter

    def test_parse_empty_text(self, get_test_icc_irish_directory: Path) -> None:
        empty_token_file = get_test_icc_irish_directory / "icc_empty_token.tsv"
        with pytest.raises(ValueError):
            ICCIrishParser.parse(empty_token_file)
    
    def test_parse_text_is_usas_label(self, get_test_icc_irish_directory: Path) -> None:
        text_as_label_file = get_test_icc_irish_directory / "icc_text_as_label.tsv"
        with pytest.raises(ValueError):
            ICCIrishParser.parse(text_as_label_file)

    def test_parse_usas_label_is_valid(self, get_test_icc_irish_directory: Path) -> None:
        usas_label_is_valid_file = get_test_icc_irish_directory / "icc_usas_label_is_valid.tsv"
        with pytest.raises(ValueError):
            ICCIrishParser.parse(usas_label_is_valid_file)
    
    def test_parse_non_z9_punct(self, get_test_icc_irish_directory: Path) -> None:
        non_z9_punct_file = get_test_icc_irish_directory / "icc_non_z9_punct.tsv"
        with pytest.raises(ValueError):
            ICCIrishParser.parse(non_z9_punct_file)

    def test_parse_mwe_errors(self, get_test_icc_irish_directory: Path) -> None:
        non_ascending_mwe_file = get_test_icc_irish_directory / "icc_non_ascending_mwe.tsv"
        with pytest.raises(ValueError):
            ICCIrishParser.parse(non_ascending_mwe_file)

        negative_mwe_file = get_test_icc_irish_directory / "icc_negative_mwe.tsv"
        with pytest.raises(ValueError):
            ICCIrishParser.parse(negative_mwe_file)

        duplicate_mwe_index_file = get_test_icc_irish_directory / "icc_duplicate_mwe_index.tsv"
        with pytest.raises(ValueError):
            ICCIrishParser.parse(duplicate_mwe_index_file)

        duplicate_mwe_index_file = get_test_icc_irish_directory / "icc_duplicate_mwe_index_2.tsv"
        with pytest.raises(ValueError):
            ICCIrishParser.parse(duplicate_mwe_index_file)

        incorrect_mwe_format_file = get_test_icc_irish_directory / "icc_incorrect_mwe_format.tsv"
        with pytest.raises(ValueError):
            ICCIrishParser.parse(incorrect_mwe_format_file)

    @pytest.mark.skipif(not downloaded_irish_datasets(), reason="Irish datasets not downloaded")
    def test_parse_full_dataset(self, get_test_icc_irish_directory: Path) -> None:
        data_files = [
            get_test_icc_irish_directory / "ICC-GA-WPH-001-the_wire.tsv",
            get_test_icc_irish_directory / "ICC-GA-WPH-003-george_orwell.tsv",
            get_test_icc_irish_directory / "ICC-GA-WR0-021-tuairisc.tsv"
        ]
        tokens_counts = [
            222,
            375,
            110
        ]
        semantic_tag_counts = [
            194,
            322,
            102
        ]
        multi_tag_counts = [
            23,
            24,
            13
        ]
        
        usas_mapper = load_usas_mapper(None, None)
        valid_usas_tags = set(usas_mapper.keys())
        tags_to_filter = set({"Z99"})
        for data_file_index, data_file in enumerate(data_files):
            dataset = ICCIrishParser.parse(data_file, valid_usas_tags, tags_to_filter)
            number_texts = len(dataset.texts)
            assert number_texts == 1
            assert dataset.labels_removed == tags_to_filter

            token_count = 0
            semantic_tags = 0
            multi_tag = 0
            for text in dataset.texts:
                token_count += len(text.tokens)
                assert text.semantic_tags is not None
                for semantic_tag_list in text.semantic_tags:
                    assert len(semantic_tag_list) == 1
                    semantic_tag = semantic_tag_list[0]
                    if semantic_tag != "PUNCT" and semantic_tag:
                        semantic_tags += 1
                    if "/" in semantic_tag:
                        multi_tag += 1
            assert token_count == tokens_counts[data_file_index]
            assert semantic_tags == semantic_tag_counts[data_file_index]
            assert multi_tag == multi_tag_counts[data_file_index]
