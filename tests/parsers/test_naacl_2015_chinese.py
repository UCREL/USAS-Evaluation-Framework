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
from usas_evaluation_framework.parsers.naacl_2015_chinese import NAACL2015ChineseUSAS


class TestNAACL2015ChineseUSAS:

    @pytest.fixture
    def get_test_directory(self, get_test_data_directory: Path) -> Path:  # noqa: F811
        return get_test_data_directory / "parsers" / "naacl_2015_chinese"

    # ------------------------------------------------------------------
    # dataset_name parameter
    # ------------------------------------------------------------------

    def test_parse_default_dataset_name(self, get_test_directory: Path) -> None:
        """Without dataset_name the returned dataset is named 'NAACL 2015'."""
        dataset = NAACL2015ChineseUSAS.parse(get_test_directory / "naacl_2015_chinese_empty.csv")
        assert dataset.name == "NAACL 2015"

    def test_parse_none_dataset_name_uses_default(self, get_test_directory: Path) -> None:
        """Passing dataset_name=None falls back to 'NAACL 2015'."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_chinese_empty.csv",
            dataset_name=None,
        )
        assert dataset.name == "NAACL 2015"

    @pytest.mark.parametrize("dataset_name", [
        "NAACL 2015 Custom",
        "Chinese USAS Corpus",
    ])
    def test_parse_custom_dataset_name(
        self, get_test_directory: Path, dataset_name: str
    ) -> None:
        """dataset_name is stored on the returned EvaluationDataset."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_chinese_empty.csv",
            dataset_name=dataset_name,
        )
        assert dataset.name == dataset_name

    # ------------------------------------------------------------------
    # language parameter
    # ------------------------------------------------------------------

    def test_parse_default_language(self, get_test_directory: Path) -> None:
        """Without language the returned dataset has language='Chinese'."""
        dataset = NAACL2015ChineseUSAS.parse(get_test_directory / "naacl_2015_chinese_empty.csv")
        assert dataset.language == "Chinese"

    @pytest.mark.parametrize("language", ["Mandarin", "Cantonese", None])
    def test_parse_custom_language(
        self, get_test_directory: Path, language: str | None
    ) -> None:
        """language is stored on the returned EvaluationDataset."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_chinese_empty.csv",
            language=language,
        )
        assert dataset.language == language

    # ------------------------------------------------------------------
    # Empty / one-token basics
    # ------------------------------------------------------------------

    def test_parse_empty(self, get_test_directory: Path) -> None:
        """An empty file (header only) produces an empty dataset."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_chinese_empty.csv"
        )
        assert isinstance(dataset, EvaluationDataset)
        assert dataset.name == "NAACL 2015"
        assert dataset.language == "Chinese"
        assert dataset.text_level == TextLevel.sentence
        assert len(dataset.texts) == 0
        assert dataset.labels_removed is None

    def test_parse_one_token(self, get_test_directory: Path) -> None:
        """Single sentence, single token — uses corrected USAS directly."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_chinese_one_token.csv"
        )
        assert len(dataset.texts) == 1
        text = dataset.texts[0]
        assert text.text == "记者"
        assert text.tokens == ["记者"]
        assert text.semantic_tags == [["Q4.2"]]
        assert text.lemmas is None
        assert text.pos_tags is None
        assert text.mwe_indexes is None

    # ------------------------------------------------------------------
    # No lemmas, pos_tags, or mwe_indexes
    # ------------------------------------------------------------------

    def test_parse_no_lemmas(self, get_test_directory: Path) -> None:
        """The corpus does not supply lemmas; lemmas is always None."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_chinese_small_example.csv"
        )
        for text in dataset.texts:
            assert text.lemmas is None

    def test_parse_no_pos_tags(self, get_test_directory: Path) -> None:
        """The corpus does not supply POS tags; pos_tags is always None."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_chinese_small_example.csv"
        )
        for text in dataset.texts:
            assert text.pos_tags is None

    def test_parse_no_mwe_indexes(self, get_test_directory: Path) -> None:
        """The corpus does not supply MWE annotations; mwe_indexes is always None."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_chinese_small_example.csv"
        )
        for text in dataset.texts:
            assert text.mwe_indexes is None

    # ------------------------------------------------------------------
    # Small example — expected output fixture
    # ------------------------------------------------------------------

    @pytest.fixture(params=[None, {"Z5"}])
    def small_example_expected(
        self, request: pytest.FixtureRequest
    ) -> tuple[EvaluationDataset, set[str] | None]:
        """
        Expected EvaluationDataset for the small example, parameterised by
        label_filter (None or {"Z5"}).

        Sentence 0 — "来自 的 记者"
          - 来自 : empty corrected_usas → ''
          - 的   : Z5
          - 记者 : Q4.2/I3.2/S2mf → Q4.2/I3.2/S2  (gender markers stripped)

        Sentence 1 — "这 是"
          - 这   : Z8mf → Z8  (gender markers stripped)
          - 是   : A3
        """
        label_filter: set[str] | None = request.param

        s0_tags = [[""], ["Z5"], ["Q4.2/I3.2/S2"]]
        s1_tags = [["Z8"], ["A3"]]

        if label_filter is not None:
            s0_tags = [[""], [""], ["Q4.2/I3.2/S2"]]

        texts = [
            EvaluationTexts(
                text="来自 的 记者",
                tokens=["来自", "的", "记者"],
                lemmas=None,
                pos_tags=None,
                semantic_tags=s0_tags,
                mwe_indexes=None,
            ),
            EvaluationTexts(
                text="这 是",
                tokens=["这", "是"],
                lemmas=None,
                pos_tags=None,
                semantic_tags=s1_tags,
                mwe_indexes=None,
            ),
        ]
        dataset = EvaluationDataset(
            name="NAACL 2015",
            text_level=TextLevel.sentence,
            labels_removed=label_filter,
            language="Chinese",
            texts=texts,
        )
        return dataset, label_filter

    def test_parse_small_example(
        self,
        get_test_directory: Path,
        small_example_expected: tuple[EvaluationDataset, set[str] | None],
    ) -> None:
        expected, label_filter = small_example_expected
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_chinese_small_example.csv",
            label_filter=label_filter,
        )

        assert len(dataset.texts) == 2
        for i in range(2):
            assert dataset.texts[i].text == expected.texts[i].text, f"text mismatch at index {i}"
            assert dataset.texts[i].tokens == expected.texts[i].tokens, f"tokens mismatch at index {i}"
            assert dataset.texts[i].semantic_tags == expected.texts[i].semantic_tags, f"semantic_tags mismatch at index {i}"
            assert dataset.texts[i].lemmas is None
            assert dataset.texts[i].pos_tags is None
            assert dataset.texts[i].mwe_indexes is None
        assert dataset.labels_removed == label_filter
        assert dataset.language == "Chinese"

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_parse_invalid_id_format(self, get_test_directory: Path) -> None:
        """A token ID that does not match <article>|<sentence_id>|<token_id> raises ValueError."""
        with pytest.raises(ValueError, match="does not match expected format"):
            NAACL2015ChineseUSAS.parse(
                get_test_directory / "naacl_2015_chinese_invalid_id_format.csv"
            )

    def test_parse_wrong_usas_format(self, get_test_directory: Path) -> None:
        """A corrected USAS tag that cannot be parsed raises ValueError."""
        with pytest.raises(ValueError):
            NAACL2015ChineseUSAS.parse(
                get_test_directory / "naacl_2015_chinese_wrong_format.csv"
            )

    # ------------------------------------------------------------------
    # Label validation
    # ------------------------------------------------------------------

    @pytest.fixture(params=[False, True])
    def label_validation_and_error(
        self, request: pytest.FixtureRequest
    ) -> tuple[set[str], bool]:
        """All labels present → no error; remove A3 → error (sentence 1)."""
        all_labels = {"Z5", "Q4.2", "I3.2", "S2", "Z8", "A3"}
        if request.param:
            return all_labels, False
        else:
            missing_one = copy.deepcopy(all_labels)
            missing_one.remove("A3")
            return missing_one, True

    @pytest.mark.parametrize("label_filter", [None, {"Z5"}])
    def test_parse_label_validation(
        self,
        get_test_directory: Path,
        label_filter: set[str] | None,
        label_validation_and_error: tuple[set[str], bool],
    ) -> None:
        data_file = get_test_directory / "naacl_2015_chinese_small_example.csv"
        validation_labels, to_error = label_validation_and_error
        if to_error:
            with pytest.raises(ValueError):
                NAACL2015ChineseUSAS.parse(
                    data_file,
                    label_validation=validation_labels,
                    label_filter=label_filter,
                )
        else:
            dataset = NAACL2015ChineseUSAS.parse(
                data_file,
                label_validation=validation_labels,
                label_filter=label_filter,
            )
            assert len(dataset.texts) == 2
            assert dataset.labels_removed == label_filter


    # ------------------------------------------------------------------
    # Full corpus smoke tests
    # ------------------------------------------------------------------


    def test_parse_full_dataset(
        self,
        get_test_directory: Path,  # noqa: F811
    ) -> None:
        data_file = get_test_directory / "naacl_2015_chinese_corpus.csv"
        usas_mapper = load_usas_mapper(None, None)
        valid_usas_tags = set(usas_mapper.keys())
        tags_to_filter: set[str] = set()

        dataset = NAACL2015ChineseUSAS.parse(data_file, valid_usas_tags, tags_to_filter)

        assert isinstance(dataset, EvaluationDataset)
        assert dataset.name == "NAACL 2015"
        assert dataset.language == "Chinese"
        assert dataset.text_level == TextLevel.sentence
        assert len(dataset.texts) == 35
        assert dataset.labels_removed is not None
        assert len(dataset.labels_removed) == 0

        token_count = 0
        semantic_tag_count = 0
        multi_tag_count = 0
        for text in dataset.texts:
            assert text.tokens is not None
            assert text.lemmas is None
            assert text.pos_tags is None
            assert text.semantic_tags is not None
            assert text.mwe_indexes is None
            token_count += len(text.tokens)
            for tag_list in text.semantic_tags:
                assert len(tag_list) == 1
                if tag_list[0] and tag_list[0] != "Z9":
                    semantic_tag_count += 1
                if "/" in tag_list[0]:
                    multi_tag_count += 1

        assert token_count == 1056
        assert semantic_tag_count == 512
        assert multi_tag_count == 56