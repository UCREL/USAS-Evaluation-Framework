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
from usas_evaluation_framework.parsers.benedict import EnglishBenedict, FinnishBenedict


class TestEnglishBenedict:

    @pytest.fixture
    def get_test_english_benedict_directory(self, get_test_data_directory: Path) -> Path:  # noqa: F811
        return get_test_data_directory / "parsers" / "benedict" / "english"


    def test_get_mwe_indexes_no_mwes(self) -> None:
        test_string = "Coffee_F2"
        expected_output: list[frozenset[int]] = [frozenset({})]
        assert EnglishBenedict.get_mwe_indexes(test_string) == expected_output

        test_string = "The_Z5 history_T1.1.1 of_Z5 coffee_F2"
        expected_output: list[frozenset[int]] = [frozenset({})] * 4
        assert EnglishBenedict.get_mwe_indexes(test_string) == expected_output

        # No text
        test_string = ""
        expected_output: list[frozenset[int]] = []
        assert EnglishBenedict.get_mwe_indexes(test_string) == expected_output

    def test_get_mwe_indexes_one_mwe(self) -> None:
        test_string = ("Turkish_F2/O4.5[i86.2.1 grind_F2/O4.5[i86.2.2 -_- "
                       "extremely_A13.3 finely_O4.5 ground_A1.1.1 ,_PUNC dust-like_O4.1")
        expected_output: list[frozenset[int]] = [frozenset({1}), frozenset({1}), frozenset({}), frozenset({}), frozenset({}), frozenset({}), frozenset({}), frozenset({})]
        assert EnglishBenedict.get_mwe_indexes(test_string) == expected_output

    def test_get_mwe_indexes_two_mwe(self) -> None:
        test_string = ("Vac_F2/O2[i136.2.1 pot_F2/O2[i136.2.2 is_A3+ by_A13.3[i137.2.1 far_A13.3[i137.2.2")
        expected_output: list[frozenset[int]] = [frozenset({1}), frozenset({1}), frozenset({}), frozenset({2}), frozenset({2})]
        assert EnglishBenedict.get_mwe_indexes(test_string) == expected_output

    def test_get_mwe_indexes_discontinuous(self) -> None:
        test_string = ("Vac_F2/O2[i136.3.1 pot_F2/O2[i136.3.2 is_A3+ by_A13.3[i136.3.3 far_A13.3")
        expected_output: list[frozenset[int]] = [frozenset({1}), frozenset({1}), frozenset({}), frozenset({1}), frozenset({})]
        assert EnglishBenedict.get_mwe_indexes(test_string) == expected_output

    def test_get_mwe_indexes_edge_case_token(self) -> None:
        # The string contains the same starting format as the MWE index, i.e. [i
        test_string = ("Vac[i_F2/O2[i136.3.1 pot_F2/O2[i136.3.2 is_A3+ by_A13.3[i136.3.3 far_A13.3")
        expected_output: list[frozenset[int]] = [frozenset({1}), frozenset({1}), frozenset({}), frozenset({1}), frozenset({})]
        assert EnglishBenedict.get_mwe_indexes(test_string) == expected_output

        # The string contains the same starting format as the MWE index, i.e. [i
        test_string = ("Vac[i_F2/O2[i136.3.1 pot_F2/O2[i136.3.2 is[i_A3+ by_A13.3[i136.3.3 far_A13.3")
        expected_output: list[frozenset[int]] = [frozenset({1}), frozenset({1}), frozenset({}), frozenset({1}), frozenset({})]
        assert EnglishBenedict.get_mwe_indexes(test_string) == expected_output

    def test_get_mwe_indexes_format_error(self) -> None:
        # No underscore between the token text and the USAS tag MWE information
        with pytest.raises(ValueError):
            test_string = ("VacF2/O2[i136.3.1 pot_F2/O2[i136.3.2 is_A3+ by_A13.3[i136.3.3 far_A13.3")
            EnglishBenedict.get_mwe_indexes(test_string)
        
        # Number of tokens doesn't match number of token indexes for that MWE
        with pytest.raises(ValueError):
            test_string = ("Vac_F2/O2[i136.2.1 pot_F2/O2[i136.2.2 is_A3+ by_A13.3[i137.2.3 far_A13.3")
            EnglishBenedict.get_mwe_indexes(test_string)
        
        # More than one [i for a token, the first token cannot have two MWE assigned to it
        # This is a form of nesting which is not supported
        with pytest.raises(ValueError):
            test_string = ("Vac_F2/O2[i136.2.1[i137.2.1 pot_F2/O2[i136.2.2 is_A3+ by_A13.3[i137.2.2 far_A13.3")
            EnglishBenedict.get_mwe_indexes(test_string)

        # Not enough token indexes for the number of tokens
        with pytest.raises(ValueError):
            test_string = ("Vac_F2/O2[i136.2.1 pot_F2/O2 is_A3+ by_A13.3 far_A13.3")
            EnglishBenedict.get_mwe_indexes(test_string)

        # Correct format check, e.g. [i\d+.\d+.\d+
        with pytest.raises(ValueError):
            test_string = ("Vac_F2/O2[iA1.2.1 pot_F2/O2[iA1.2.2 is_A3+ by_A13.3 far_A13.3")
            EnglishBenedict.get_mwe_indexes(test_string)

        # Correct format check, e.g. [i\d+.\d+.\d+
        with pytest.raises(ValueError):
            test_string = ("Vac_F2/O2[i.2.1 pot_F2/O2[i.2.2 is_A3+ by_A13.3 far_A13.3")
            EnglishBenedict.get_mwe_indexes(test_string)

        # Correct format check, e.g. [i\d+.\d+.\d+
        with pytest.raises(ValueError):
            test_string = ("Vac_F2/O2[i1.g.1 pot_F2/O2[i1.g.2 is_A3+ by_A13.3 far_A13.3")
            EnglishBenedict.get_mwe_indexes(test_string)

        # Correct format check, e.g. [i\d+.\d+.\d+
        with pytest.raises(ValueError):
            test_string = ("Vac_F2/O2[i1.2.g pot_F2/O2[i1.2.g is_A3+ by_A13.3 far_A13.3")
            EnglishBenedict.get_mwe_indexes(test_string)

        # Correct format check, e.g. [i\d+.\d+.\d+ with 3 digits
        with pytest.raises(ValueError):
            test_string = ("Vac_F2/O2[i1.2 pot_F2/O2[i1.2.2 is_A3+ by_A13.3 far_A13.3")
            EnglishBenedict.get_mwe_indexes(test_string)

    def test_validate_text_string_format_empty_string(self) -> None:
        with pytest.raises(ValueError):
            EnglishBenedict.validate_text_string_format("")
        
        with pytest.raises(ValueError):
            EnglishBenedict.validate_text_string_format(" ")
    
    def test_validate_text_string_format_expected_string(self) -> None:
        # Only USAS tags
        test_string = ("Vac_F2/O2 pot_F2/O2 is_A3+ by_A13.3 far_A13.3")
        expected_output = ("Vac_F2/O2 pot_F2/O2 is_A3+ by_A13.3 far_A13.3",
                           ["Vac", "pot", "is", "by", "far"],
                           ["F2/O2", "F2/O2", "A3", "A13.3", "A13.3"])
        assert EnglishBenedict.validate_text_string_format(test_string) == expected_output

        # With MWEs
        test_string = ("Vac_F2/O2[i136.3.1 pot_F2/O2[i136.3.2 is_A3+ by_A13.3[i136.3.3 far_A13.3")
        expected_output = ("Vac_F2/O2[i136.3.1 pot_F2/O2[i136.3.2 is_A3+ by_A13.3[i136.3.3 far_A13.3",
                           ["Vac", "pot", "is", "by", "far"],
                           ["F2/O2", "F2/O2", "A3", "A13.3", "A13.3"])
        assert EnglishBenedict.validate_text_string_format(test_string) == expected_output

        # With the various USAS tags that should be converted
        test_string = ("-_- ._. a_! ._PUNC another_,")
        expected_output = ("-_- ._. a_! ._PUNC another_,",
                           ["-", ".", "a", ".", "another"],
                           ["PUNCT", "PUNCT", "PUNCT", "PUNCT", "PUNCT"])
        assert EnglishBenedict.validate_text_string_format(test_string) == expected_output

        # The USAS tags that should be converted with MWE information
        test_string = ("Vac_F2/O2[i136.3.1 !_![i136.3.2 is_A3+ by_A13.3[i136.3.3 far_A13.3")
        expected_output = ("Vac_F2/O2[i136.3.1 !_![i136.3.2 is_A3+ by_A13.3[i136.3.3 far_A13.3",
                           ["Vac", "!", "is", "by", "far"],
                           ["F2/O2", "PUNCT", "A3", "A13.3", "A13.3"])
        assert EnglishBenedict.validate_text_string_format(test_string) == expected_output

    def test_validate_text_string_format_incorrect_format(self) -> None:
        # No underscore or USAS tags
        with pytest.raises(ValueError):
            test_string = "Vac pot is by far"
            EnglishBenedict.validate_text_string_format(test_string)
        
        # No underscore
        with pytest.raises(ValueError):
            test_string = "VacF2/O2 potF2/O2 isA3+ byA13.3 farA13.3"
            EnglishBenedict.validate_text_string_format(test_string)
        
        # Too many underscores
        with pytest.raises(ValueError):
            test_string = "Vac_F2/O2_F2/O2 pot_F2/O2 is_A3+ by_A13.3 far_A13.3"
            EnglishBenedict.validate_text_string_format(test_string)

        # no text token
        with pytest.raises(ValueError):
            test_string = "_F2/O2 pot_F2/O2 is_A3+ by_A13.3 far_A13.3"
            EnglishBenedict.validate_text_string_format(test_string)

        # Incorrect USAS tags
        with pytest.raises(ValueError):
            test_string = "Vac_ZX2"
            EnglishBenedict.validate_text_string_format(test_string)

        # No USAS tag after underscore
        with pytest.raises(ValueError):
            test_string = "Vac_ pot_"
            EnglishBenedict.validate_text_string_format(test_string)

        # Only MWE information and no USAS tag
        with pytest.raises(ValueError):
            test_string = "Vac_[i136.3.1 pot_[i136.3.2"
            EnglishBenedict.validate_text_string_format(test_string)



    def test_parse_empty_text(self, get_test_english_benedict_directory: Path) -> None:
        empty_english_data_file = get_test_english_benedict_directory / "benedict_english_empty.txt"
        dataset = EnglishBenedict.parse(empty_english_data_file)
        assert isinstance(dataset, EvaluationDataset)
        assert dataset.name == "Benedict English"
        assert dataset.text_level == "sentence"
        assert len(dataset.texts) == 0
        assert dataset.labels_removed is None

    def test_parse_one_token(self, get_test_english_benedict_directory: Path) -> None:
        english_one_token_data_file = get_test_english_benedict_directory / "benedict_english_one_token.txt"
        dataset = EnglishBenedict.parse(english_one_token_data_file)
        assert len(dataset.texts) == 1
        assert dataset.texts[0].text == "Coffee_F2"
        assert dataset.texts[0].tokens == ["Coffee"]
        assert dataset.texts[0].lemmas is None
        assert dataset.texts[0].pos_tags is None
        assert dataset.texts[0].semantic_tags == ["F2"]
        assert dataset.texts[0].mwe_indexes == [frozenset({})]

    def test_parse_wrong_format(self, get_test_english_benedict_directory: Path) -> None:
        english_wrong_format_data_file = get_test_english_benedict_directory / "benedict_english_wrong_format.txt"
        with pytest.raises(ValueError):
            EnglishBenedict.parse(english_wrong_format_data_file)

    @pytest.fixture(params=[None, set(["F2"])])
    def small_english_example_expected_data_with_label_filter(self, request: pytest.FixtureRequest) -> tuple[EvaluationDataset, None | set[str]]:
        semantic_tags: list[list[str]] = [
            ["F2"],
            ["Z5", "T1.1.1", "Z5", "F2"],
            ["F2/O4.5", "F2/O4.5", "PUNCT", "A13.3", "O4.5", "A1.1.1", "PUNCT", "O4.1"],
            ["F2/O4.5", "F2/O4.5", "PUNCT", "Z5", "O4.5", "F2/O4.5", "PUNCT", "A13.4", "O4.1", "O1.1"]
        ]
        if request.param is not None:
            semantic_tags = [
                [""],
                ["Z5", "T1.1.1", "Z5", ""],
                ["F2/O4.5", "F2/O4.5", "PUNCT", "A13.3", "O4.5", "A1.1.1", "PUNCT", "O4.1"],
                ["F2/O4.5", "F2/O4.5", "PUNCT", "Z5", "O4.5", "F2/O4.5", "PUNCT", "A13.4", "O4.1", "O1.1"]
            ]
        
        evaluation_texts: list[EvaluationTexts] = [
            EvaluationTexts(
                text="Coffee_F2",
                tokens=["Coffee"],
                lemmas=None,
                pos_tags=None,
                semantic_tags=semantic_tags[0],
                mwe_indexes=[frozenset({})]
            ),
            EvaluationTexts(
                text="The_Z5 history_T1.1.1 of_Z5 coffee_F2",
                tokens=["The", "history", "of", "coffee"],
                lemmas=None,
                pos_tags=None,
                semantic_tags=semantic_tags[1],
                mwe_indexes=[frozenset({})] * 4
            ),
            EvaluationTexts(
                text="Turkish_F2/O4.5[i86.2.1 grind_F2/O4.5[i86.2.2 -_- extremely_A13.3 finely_O4.5 ground_A1.1.1 ,_PUNC dust-like_O4.1",
                tokens=["Turkish", "grind", "-", "extremely", "finely", "ground", ",", "dust-like"],
                lemmas=None,
                pos_tags=None,
                semantic_tags=semantic_tags[2],
                mwe_indexes=[frozenset({1}), frozenset({1})] + [frozenset({})] * 6
            ),
            EvaluationTexts(
                text="Espresso_F2/O4.5[i87.2.1 grind_F2/O4.5[i87.2.2 -_- a_Z5 fine_O4.5 grind_F2/O4.5 ,_PUNC almost_A13.4 dust-like_O4.1 powder_O1.1",
                tokens=["Espresso", "grind", "-", "a", "fine", "grind", ",", "almost", "dust-like", "powder"],
                lemmas=None,
                pos_tags=None,
                semantic_tags=semantic_tags[3],
                mwe_indexes=[frozenset({1}), frozenset({1})] + [frozenset({})] * 8
            )
        ]
        return EvaluationDataset(
            name="Benedict English",
            text_level=TextLevel.sentence,
            labels_removed=request.param,
            texts=evaluation_texts
        ), request.param

    @pytest.mark.parametrize("data_file_name", ["benedict_english_small_example.txt", "benedict_english_small_with_empty_line.txt"])
    def test_parse_small_example(self,
                                 get_test_english_benedict_directory: Path,
                                 data_file_name: str,
                                 small_english_example_expected_data_with_label_filter: tuple[EvaluationDataset, None | set[str]]) -> None:
        english_data_file = get_test_english_benedict_directory / data_file_name
        small_english_example_expected_data, label_filter = small_english_example_expected_data_with_label_filter
        dataset = EnglishBenedict.parse(english_data_file, label_filter=label_filter)

        expected_number_of_texts = 4
        assert len(dataset.texts) == expected_number_of_texts
        for text_index in range(expected_number_of_texts):
            assert dataset.texts[text_index].text == small_english_example_expected_data.texts[text_index].text
            assert dataset.texts[text_index].tokens == small_english_example_expected_data.texts[text_index].tokens
            assert dataset.texts[text_index].lemmas == small_english_example_expected_data.texts[text_index].lemmas
            assert dataset.texts[text_index].pos_tags == small_english_example_expected_data.texts[text_index].pos_tags
            assert dataset.texts[text_index].semantic_tags == small_english_example_expected_data.texts[text_index].semantic_tags
            assert dataset.texts[text_index].mwe_indexes == small_english_example_expected_data.texts[text_index].mwe_indexes
        assert dataset.labels_removed == label_filter

    @pytest.fixture(params=[False, True])
    def small_label_validation_and_error(self, request: pytest.FixtureRequest) -> tuple[set[str], bool]:
        all_labels = set([
            "Z5", "F2", "T1.1.1", "O4.5", "A13.3", "A1.1.1", "O4.1", "A13.4", "O1.1" 
        ])
        if request.param:
            return all_labels, False
        else:
            one_less_label = copy.deepcopy(all_labels)
            one_less_label.remove("O4.5")
            return one_less_label, True

    @pytest.mark.parametrize("label_filter", [None, set(["F2"])])
    def test_parse_label_validation(self,
                                    get_test_english_benedict_directory: Path,
                                    label_filter: None | set[str],
                                    small_label_validation_and_error: tuple[set[str], bool]) -> None:
        english_data_file = get_test_english_benedict_directory / "benedict_english_small_example.txt"
        validation_labels, to_error = small_label_validation_and_error
        if to_error:
            with pytest.raises(ValueError):
                EnglishBenedict.parse(english_data_file, label_validation=validation_labels, label_filter=label_filter)
        else:
            dataset = EnglishBenedict.parse(english_data_file, label_validation=validation_labels, label_filter=label_filter)
            assert len(dataset.texts) == 4
            assert dataset.labels_removed == label_filter

    def test_parse_text_is_usas_label(self, get_test_english_benedict_directory: Path) -> None:
        english_text_as_label_file = get_test_english_benedict_directory / "benedict_english_text_as_label.txt"
        with pytest.raises(ValueError):
            EnglishBenedict.parse(english_text_as_label_file)


    def test_parse_full_dataset(self, get_test_english_benedict_directory: Path) -> None:
        english_data_file = get_test_english_benedict_directory / "benedict_english_corpus.txt"
        usas_mapper = load_usas_mapper(None, None)
        valid_usas_tags = set(usas_mapper.keys())
        tags_to_filter = set({"Z99"})
        dataset = EnglishBenedict.parse(english_data_file, valid_usas_tags, tags_to_filter)
        number_texts = len(dataset.texts)
        assert number_texts == 73
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
        assert token_count == 3899
        assert semantic_tags == 3468
        assert multi_tag == 212
        

class TestFinnishBenedict:

    @pytest.fixture
    def get_test_finnish_benedict_directory(self, get_test_data_directory: Path) -> Path:  # noqa: F811
        return get_test_data_directory / "parsers" / "benedict" / "finnish"

    def test_validate_text_string_format_empty_string(self) -> None:
        with pytest.raises(ValueError):
            FinnishBenedict.validate_text_string_format("")
        
        with pytest.raises(ValueError):
            FinnishBenedict.validate_text_string_format(" ")
    
    def test_validate_text_string_format_expected_string(self) -> None:
        # Only USAS tags
        test_string = ("Vac_F2/O2 pot_F2/O2 is_A3+")
        expected_output = EvaluationTexts(text="Vac_F2/O2 pot_F2/O2 is_A3+",
                                          lemmas=None,
                                          pos_tags=None,
                                          tokens=["Vac", "pot", "is"],
                                          semantic_tags=["F2/O2", "F2/O2", "A3"],
                                          mwe_indexes=[frozenset({})] * 3)
        assert FinnishBenedict.validate_text_string_format(test_string) == expected_output

        # With MWEs
        test_string = ("Vac_F2/O2_i pot_F2/O2_i is_A3+")
        expected_output = EvaluationTexts(text="Vac_F2/O2_i pot_F2/O2_i is_A3+",
                                          lemmas=None,
                                          pos_tags=None,
                                          tokens=["Vac", "pot", "is"],
                                          semantic_tags=["F2/O2", "F2/O2", "A3"],
                                          mwe_indexes=[frozenset({1}), frozenset({1}), frozenset({})])
        assert FinnishBenedict.validate_text_string_format(test_string) == expected_output

        # With the various USAS tags that should be converted
        test_string = ('- , ! : ( ) ? " .')
        expected_output = EvaluationTexts(text='- , ! : ( ) ? " .',
                           tokens=['-', ',', '!', ':', '(', ')', '?', '"', '.'],
                           semantic_tags=["PUNCT"] * 9,
                           mwe_indexes=[frozenset({})] * 9,
                           lemmas=None,
                           pos_tags=None)
        assert FinnishBenedict.validate_text_string_format(test_string) == expected_output

        # Test the MWE indexing can handle multiple MWEs
        test_string = ("Vac_F2/O2_i pot_F2/O2_i is_A3+ good_A13.3_i day_A13.3_i")
        expected_output = EvaluationTexts(text="Vac_F2/O2_i pot_F2/O2_i is_A3+ good_A13.3_i day_A13.3_i",
                                          lemmas=None,
                                          pos_tags=None,
                                          tokens=["Vac", "pot", "is", "good", "day"],
                                          semantic_tags=["F2/O2", "F2/O2", "A3", "A13.3", "A13.3"],
                                          mwe_indexes=[frozenset({1}), frozenset({1}), frozenset({}), frozenset({2}), frozenset({2})])
        assert FinnishBenedict.validate_text_string_format(test_string) == expected_output


    def test_validate_text_string_format_incorrect_format(self) -> None:
        # No underscore or USAS tags
        with pytest.raises(ValueError):
            test_string = "Vac pot is by far"
            FinnishBenedict.validate_text_string_format(test_string)
        
        # No underscore
        with pytest.raises(ValueError):
            test_string = "VacF2/O2 potF2/O2 isA3+ byA13.3 farA13.3"
            FinnishBenedict.validate_text_string_format(test_string)
        
        # The underscore where an `i` should be contains the wrong character
        with pytest.raises(ValueError):
            test_string = "Vac_F2/O2_g pot_F2/O2 is_A3+ by_A13.3 far_A13.3"
            FinnishBenedict.validate_text_string_format(test_string)

        # Too many underscores
        with pytest.raises(ValueError):
            test_string = "Vac_F2/O2_i_ pot_F2/O2_i_ is_A3+ by_A13.3 far_A13.3"
            FinnishBenedict.validate_text_string_format(test_string)

        # no text token
        with pytest.raises(ValueError):
            test_string = "_F2/O2 pot_F2/O2 is_A3+ by_A13.3 far_A13.3"
            FinnishBenedict.validate_text_string_format(test_string)

        # Incorrect USAS tags
        with pytest.raises(ValueError):
            test_string = "Vac_ZX2"
            FinnishBenedict.validate_text_string_format(test_string)

        # Incorrect placement of the `i` MWE tag
        with pytest.raises(ValueError):
            test_string = "Vac_i pot_i"
            FinnishBenedict.validate_text_string_format(test_string)

    def test_parse_empty_text(self, get_test_finnish_benedict_directory: Path) -> None:
        empty_finnish_data_file = get_test_finnish_benedict_directory / "benedict_finnish_empty.txt"
        dataset = FinnishBenedict.parse(empty_finnish_data_file)
        assert isinstance(dataset, EvaluationDataset)
        assert dataset.name == "Benedict Finnish"
        assert dataset.text_level == "sentence"
        assert len(dataset.texts) == 0
        assert dataset.labels_removed is None

    def test_parse_one_token(self, get_test_finnish_benedict_directory: Path) -> None:
        finnish_one_token_data_file = get_test_finnish_benedict_directory / "benedict_finnish_one_token.txt"
        dataset = FinnishBenedict.parse(finnish_one_token_data_file)
        assert len(dataset.texts) == 1
        assert dataset.texts[0].text == "Kahvi_F2"
        assert dataset.texts[0].tokens == ["Kahvi"]
        assert dataset.texts[0].lemmas is None
        assert dataset.texts[0].pos_tags is None
        assert dataset.texts[0].semantic_tags == ["F2"]
        assert dataset.texts[0].mwe_indexes == [frozenset({})]

    @pytest.fixture(params=[None, set(["F2"])])
    def small_finnish_example_expected_data_with_label_filter(self, request: pytest.FixtureRequest) -> tuple[EvaluationDataset, None | set[str]]:
        semantic_tags: list[list[str]] = [
            ["A1.5.2", "PUNCT"],
            ["N3.1/F2/O4.3"],
            ["F2/L3", "F2/L3"],
            ["F2"]
        ]
        if request.param is not None:
            semantic_tags = [
                ["A1.5.2", "PUNCT"],
                ["N3.1/F2/O4.3"],
                ["F2/L3", "F2/L3"],
                [""]
            ]
        
        evaluation_texts: list[EvaluationTexts] = [
            EvaluationTexts(
                text="käyttökelvottomia_A1.5.2- .",
                tokens=["käyttökelvottomia", "."],
                lemmas=None,
                pos_tags=None,
                semantic_tags=semantic_tags[0],
                mwe_indexes=[frozenset({})] * 2
            ),
            EvaluationTexts(
                text="Paahtoasteet_N3.1/F2/O4.3",
                tokens=["Paahtoasteet"],
                lemmas=None,
                pos_tags=None,
                semantic_tags=semantic_tags[1],
                mwe_indexes=[frozenset({})]
            ),
            EvaluationTexts(
                text="Coffea_F2/L3_i arabica_F2/L3_i",
                tokens=["Coffea", "arabica"],
                lemmas=None,
                pos_tags=None,
                semantic_tags=semantic_tags[2],
                mwe_indexes=[frozenset({1}), frozenset({1})]
            ),
            EvaluationTexts(
                text="Coffea_F2",
                tokens=["Coffea"],
                lemmas=None,
                pos_tags=None,
                semantic_tags=semantic_tags[3],
                mwe_indexes=[frozenset({})]
            )
        ]
        return EvaluationDataset(
            name="Benedict Finnish",
            text_level=TextLevel.sentence,
            labels_removed=request.param,
            texts=evaluation_texts
        ), request.param

    @pytest.mark.parametrize("data_file_name", ["benedict_finnish_small_example.txt", "benedict_finnish_small_with_empty_line.txt"])
    def test_parse_small_example(self,
                                 get_test_finnish_benedict_directory: Path,
                                 data_file_name: str,
                                 small_finnish_example_expected_data_with_label_filter: tuple[EvaluationDataset, None | set[str]]) -> None:
        finnish_data_file = get_test_finnish_benedict_directory / data_file_name
        small_finnish_example_expected_data, label_filter = small_finnish_example_expected_data_with_label_filter
        dataset = FinnishBenedict.parse(finnish_data_file, label_filter=label_filter)

        expected_number_of_texts = 4
        assert len(dataset.texts) == expected_number_of_texts
        for text_index in range(expected_number_of_texts):
            assert dataset.texts[text_index].text == small_finnish_example_expected_data.texts[text_index].text
            assert dataset.texts[text_index].tokens == small_finnish_example_expected_data.texts[text_index].tokens
            assert dataset.texts[text_index].lemmas == small_finnish_example_expected_data.texts[text_index].lemmas
            assert dataset.texts[text_index].pos_tags == small_finnish_example_expected_data.texts[text_index].pos_tags
            assert dataset.texts[text_index].semantic_tags == small_finnish_example_expected_data.texts[text_index].semantic_tags
            assert dataset.texts[text_index].mwe_indexes == small_finnish_example_expected_data.texts[text_index].mwe_indexes
        assert dataset.labels_removed == label_filter

    @pytest.fixture(params=[False, True])
    def small_label_validation_and_error(self, request: pytest.FixtureRequest) -> tuple[set[str], bool]:
        all_labels = set([
            "A1.5.2", "N3.1", "F2", "O4.3", "L3"
        ])
        if request.param:
            return all_labels, False
        else:
            one_less_label = copy.deepcopy(all_labels)
            one_less_label.remove("L3")
            return one_less_label, True

    @pytest.mark.parametrize("label_filter", [None, set(["F2"])])
    def test_parse_label_validation(self,
                                    get_test_finnish_benedict_directory: Path,
                                    label_filter: None | set[str],
                                    small_label_validation_and_error: tuple[set[str], bool]) -> None:
        finnish_data_file = get_test_finnish_benedict_directory / "benedict_finnish_small_example.txt"
        validation_labels, to_error = small_label_validation_and_error
        if to_error:
            with pytest.raises(ValueError):
                FinnishBenedict.parse(finnish_data_file, label_validation=validation_labels, label_filter=label_filter)
        else:
            dataset = FinnishBenedict.parse(finnish_data_file, label_validation=validation_labels, label_filter=label_filter)
            assert len(dataset.texts) == 4
            assert dataset.labels_removed == label_filter

    def test_parse_text_is_usas_label(self, get_test_finnish_benedict_directory: Path) -> None:
        finnish_text_as_label_file = get_test_finnish_benedict_directory / "benedict_finnish_text_as_label.txt"
        with pytest.raises(ValueError):
            FinnishBenedict.parse(finnish_text_as_label_file)


    def test_parse_full_dataset(self, get_test_finnish_benedict_directory: Path) -> None:
        finnish_data_file = get_test_finnish_benedict_directory / "benedict_finnish_corpus.txt"
        usas_mapper = load_usas_mapper(None, None)
        valid_usas_tags = set(usas_mapper.keys())
        tags_to_filter = set({"Z99"})
        dataset = FinnishBenedict.parse(finnish_data_file, valid_usas_tags, tags_to_filter)
        number_texts = len(dataset.texts)
        assert number_texts == 72

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
        assert token_count == 2439
        assert semantic_tags == 2068
        assert multi_tag == 254
