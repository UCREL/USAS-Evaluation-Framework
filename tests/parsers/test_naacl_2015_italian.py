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


class TestNAACL2015ItalianUSAS:

    @pytest.fixture
    def get_test_directory(self, get_test_data_directory: Path) -> Path:  # noqa: F811
        return get_test_data_directory / "parsers" / "naacl_2015_italian"

    # ------------------------------------------------------------------
    # dataset_name parameter
    # ------------------------------------------------------------------

    def test_parse_default_dataset_name(self, get_test_directory: Path) -> None:
        """Without dataset_name the returned dataset is named 'NAACL 2015'."""
        dataset = NAACL2015ChineseUSAS.parse(get_test_directory / "naacl_2015_italian_empty.csv")
        assert dataset.name == "NAACL 2015"

    def test_parse_none_dataset_name_uses_default(self, get_test_directory: Path) -> None:
        """Passing dataset_name=None falls back to 'NAACL 2015'."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_italian_empty.csv",
            dataset_name=None,
        )
        assert dataset.name == "NAACL 2015"

    @pytest.mark.parametrize("dataset_name", ["NAACL 2015 Italian", "Italian Blog Corpus"])
    def test_parse_custom_dataset_name(
        self, get_test_directory: Path, dataset_name: str
    ) -> None:
        """dataset_name is stored on the returned EvaluationDataset."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_italian_empty.csv",
            dataset_name=dataset_name,
        )
        assert dataset.name == dataset_name

    # ------------------------------------------------------------------
    # language parameter
    # ------------------------------------------------------------------

    def test_parse_italian_language(self, get_test_directory: Path) -> None:
        """Caller can set language='Italian' for the returned dataset."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_italian_empty.csv",
            language="Italian",
        )
        assert dataset.language == "Italian"

    @pytest.mark.parametrize("language", ["Italian", "it", None])
    def test_parse_custom_language(
        self, get_test_directory: Path, language: str | None
    ) -> None:
        """language is stored on the returned EvaluationDataset."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_italian_empty.csv",
            language=language,
        )
        assert dataset.language == language

    # ------------------------------------------------------------------
    # Empty / one-token basics
    # ------------------------------------------------------------------

    def test_parse_empty(self, get_test_directory: Path) -> None:
        """An empty file (header only) produces an empty dataset."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_italian_empty.csv",
            language="Italian",
        )
        assert isinstance(dataset, EvaluationDataset)
        assert dataset.text_level == TextLevel.sentence
        assert len(dataset.texts) == 0
        assert dataset.labels_removed is None

    def test_parse_one_token(self, get_test_directory: Path) -> None:
        """Single sentence, single token — mwe_indexes contains one empty frozenset."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_italian_one_token.csv",
            language="Italian",
        )
        assert len(dataset.texts) == 1
        text = dataset.texts[0]
        assert text.tokens == ["ciao"]
        assert text.semantic_tags == [[""]]
        assert text.lemmas is None
        assert text.pos_tags is None
        assert text.mwe_indexes == [frozenset()]

    # ------------------------------------------------------------------
    # MWE presence / absence
    # ------------------------------------------------------------------

    def test_parse_token_without_mwe_gets_empty_frozenset(
        self, get_test_directory: Path
    ) -> None:
        """Tokens with a blank mwe cell produce frozenset() entries."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_italian_small_example.csv",
            language="Italian",
        )
        sentence_1 = dataset.texts[0]
        assert sentence_1.mwe_indexes is not None
        assert all(idx == frozenset() for idx in sentence_1.mwe_indexes)

    def test_parse_token_with_mwe_gets_populated_frozenset(
        self, get_test_directory: Path
    ) -> None:
        """Tokens with a numeric mwe cell produce frozenset({n}) entries."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_italian_small_example.csv",
            language="Italian",
        )
        sentence_2 = dataset.texts[1]
        assert sentence_2.mwe_indexes is not None
        assert sentence_2.mwe_indexes[0] == frozenset({1})
        assert sentence_2.mwe_indexes[1] == frozenset({1})

    def test_parse_tokens_after_mwe_get_empty_frozenset(
        self, get_test_directory: Path
    ) -> None:
        """Tokens after the MWE in a sentence still produce frozenset()."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_italian_small_example.csv",
            language="Italian",
        )
        sentence_2 = dataset.texts[1]
        assert sentence_2.mwe_indexes is not None
        assert sentence_2.mwe_indexes[2] == frozenset()
        assert sentence_2.mwe_indexes[3] == frozenset()
        assert sentence_2.mwe_indexes[4] == frozenset()

    def test_parse_mwe_indexes_length_matches_tokens(
        self, get_test_directory: Path
    ) -> None:
        """mwe_indexes has the same length as tokens for every sentence."""
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_italian_small_example.csv",
            language="Italian",
        )
        for text in dataset.texts:
            assert text.mwe_indexes is not None
            assert len(text.mwe_indexes) == len(text.tokens)

    def test_parse_mwe_indexes_renumbered_per_sentence(
        self, get_test_directory: Path
    ) -> None:
        """
        MWE group integers are re-indexed to start at 1 within each sentence,
        even when the annotator used a global counter across the whole file.

        mwe_remapping.csv contains three sentences:
          - Sentence 1: groups [1, 1, -, -]
          - Sentence 2: groups [2, 2, -, -]  → renumbered to [1, 1, -, -]
          - Sentence 3: groups [3, 3, -, 4, 4, -] → renumbered to [1, 1, -, 2, 2, -]
        """
        dataset = NAACL2015ChineseUSAS.parse(
            get_test_directory / "naacl_2015_italian_mwe_remapping.csv",
            language="Italian",
        )
        assert len(dataset.texts) == 3

        s0 = dataset.texts[0]
        assert s0.mwe_indexes == [frozenset({1}), frozenset({1}), frozenset(), frozenset()]

        s1 = dataset.texts[1]
        assert s1.mwe_indexes == [frozenset({1}), frozenset({1}), frozenset(), frozenset()]

        s2 = dataset.texts[2]
        assert s2.mwe_indexes == [
            frozenset({1}), frozenset({1}), frozenset(),
            frozenset({2}), frozenset({2}), frozenset(),
        ]

    # ------------------------------------------------------------------
    # Small example — full expected output
    # ------------------------------------------------------------------

    @pytest.fixture(params=[None, {"Z5"}])
    def small_example_expected(
        self, request: pytest.FixtureRequest
    ) -> tuple[EvaluationDataset, set[str] | None]:
        """
        Expected EvaluationDataset for the small example, parameterised by
        label_filter (None or {"Z5"}).

        Sentence 0 — "Firenze e bella ."
          - Firenze : empty corrected_usas → ''
          - e       : Z5
          - bella   : empty → ''
          - .       : Z9
          No MWEs.

        Sentence 1 — "New York e bella ."
          - New     : empty → '', mwe_indexes={1}
          - York    : empty → '', mwe_indexes={1}
          - e       : Z5,   mwe_indexes={}
          - bella   : empty → '', mwe_indexes={}
          - .       : Z9,   mwe_indexes={}
        """
        label_filter: set[str] | None = request.param

        s0_tags = [[""], ["Z5"], [""], ["Z9"]]
        s1_tags = [[""], [""], ["Z5"], [""], ["Z9"]]

        if label_filter is not None:
            s0_tags = [[""], [""], [""], ["Z9"]]
            s1_tags = [[""], [""], [""], [""], ["Z9"]]

        texts = [
            EvaluationTexts(
                text="Firenze e bella .",
                tokens=["Firenze", "e", "bella", "."],
                lemmas=None,
                pos_tags=None,
                semantic_tags=s0_tags,
                mwe_indexes=[frozenset(), frozenset(), frozenset(), frozenset()],
            ),
            EvaluationTexts(
                text="New York e bella .",
                tokens=["New", "York", "e", "bella", "."],
                lemmas=None,
                pos_tags=None,
                semantic_tags=s1_tags,
                mwe_indexes=[frozenset({1}), frozenset({1}), frozenset(), frozenset(), frozenset()],
            ),
        ]
        dataset = EvaluationDataset(
            name="NAACL 2015",
            text_level=TextLevel.sentence,
            labels_removed=label_filter,
            language="Italian",
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
            get_test_directory / "naacl_2015_italian_small_example.csv",
            label_filter=label_filter,
            language="Italian",
        )

        assert len(dataset.texts) == 2
        for i in range(2):
            assert dataset.texts[i].text == expected.texts[i].text, f"text mismatch at {i}"
            assert dataset.texts[i].tokens == expected.texts[i].tokens, f"tokens mismatch at {i}"
            assert dataset.texts[i].semantic_tags == expected.texts[i].semantic_tags, f"tags mismatch at {i}"
            assert dataset.texts[i].mwe_indexes == expected.texts[i].mwe_indexes, f"mwe_indexes mismatch at {i}"
            assert dataset.texts[i].lemmas is None
            assert dataset.texts[i].pos_tags is None
        assert dataset.labels_removed == label_filter
        assert dataset.language == "Italian"

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_parse_invalid_id_format(self, get_test_directory: Path) -> None:
        """A token ID that does not match <article>|<sentence_id>|<token_id> raises ValueError."""
        with pytest.raises(ValueError, match="does not match expected format"):
            NAACL2015ChineseUSAS.parse(
                get_test_directory / "naacl_2015_italian_invalid_id_format.csv"
            )

    def test_parse_wrong_usas_format(self, get_test_directory: Path) -> None:
        """A corrected_usas value that cannot be parsed as a USAS tag raises ValueError."""
        with pytest.raises(ValueError):
            NAACL2015ChineseUSAS.parse(
                get_test_directory / "naacl_2015_italian_wrong_format.csv"
            )

    def test_parse_invalid_mwe_format(self, get_test_directory: Path) -> None:
        """A non-numeric mwe cell value raises ValueError."""
        with pytest.raises(ValueError, match="[Ii]nvalid MWE"):
            NAACL2015ChineseUSAS.parse(
                get_test_directory / "naacl_2015_italian_invalid_mwe_format.csv"
            )

    # ------------------------------------------------------------------
    # Label validation
    # ------------------------------------------------------------------

    @pytest.fixture(params=[False, True])
    def label_validation_and_error(
        self, request: pytest.FixtureRequest
    ) -> tuple[set[str], bool]:
        """All labels present → no error; remove Z9 → error (full-stop tokens)."""
        all_labels = {"Z5", "Z9"}
        if request.param:
            return all_labels, False
        else:
            missing_one = copy.deepcopy(all_labels)
            missing_one.remove("Z9")
            return missing_one, True

    @pytest.mark.parametrize("label_filter", [None, {"Z5"}])
    def test_parse_label_validation(
        self,
        get_test_directory: Path,
        label_filter: set[str] | None,
        label_validation_and_error: tuple[set[str], bool],
    ) -> None:
        data_file = get_test_directory / "naacl_2015_italian_small_example.csv"
        validation_labels, to_error = label_validation_and_error
        if to_error:
            with pytest.raises(ValueError):
                NAACL2015ChineseUSAS.parse(
                    data_file,
                    label_validation=validation_labels,
                    label_filter=label_filter,
                    language="Italian",
                )
        else:
            dataset = NAACL2015ChineseUSAS.parse(
                data_file,
                label_validation=validation_labels,
                label_filter=label_filter,
                language="Italian",
            )
            assert len(dataset.texts) == 2
            assert dataset.labels_removed == label_filter

    # ------------------------------------------------------------------
    # Full corpus smoke tests
    # ------------------------------------------------------------------

    def test_parse_full_dataset(self, get_test_directory: Path) -> None:
        data_file = get_test_directory / "naacl_2015_italian_corpus.csv"
        usas_mapper = load_usas_mapper(None, None)
        valid_usas_tags = set(usas_mapper.keys())

        dataset = NAACL2015ChineseUSAS.parse(
            data_file,
            label_validation=valid_usas_tags,
            label_filter=set(),
            language="Italian",
        )

        assert isinstance(dataset, EvaluationDataset)
        assert dataset.name == "NAACL 2015"
        assert dataset.language == "Italian"
        assert dataset.text_level == TextLevel.sentence
        assert len(dataset.texts) == 42

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

        assert token_count == 1305
        assert mwe_token_count == 2

        # MWE group 1 is in sentence 21 (index 20), at token positions 19 and 20
        sentence_21 = dataset.texts[20]
        assert sentence_21.mwe_indexes[19] == frozenset({1})
        assert sentence_21.mwe_indexes[20] == frozenset({1})
