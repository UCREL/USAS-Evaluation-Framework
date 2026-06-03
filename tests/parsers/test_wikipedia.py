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
from usas_evaluation_framework.parsers.wikipedia import WikipediaUSAS


class TestWikipediaUSAS:

    @pytest.fixture
    def get_test_directory(self, get_test_data_directory: Path) -> Path:  # noqa: F811
        return get_test_data_directory / "parsers" / "spanish_wikipedia"

    # ------------------------------------------------------------------
    # dataset_name parameter
    # ------------------------------------------------------------------

    def test_parse_default_dataset_name(self, get_test_directory: Path) -> None:
        """Without dataset_name the returned dataset is named 'Wikipedia USAS'."""
        dataset = WikipediaUSAS.parse(get_test_directory / "spanish_wikipedia_empty.csv")
        assert dataset.name == "Wikipedia USAS"

    @pytest.mark.parametrize("dataset_name", [
        "Medical Wikipedia",
        "English Wikipedia USAS",
        "My Custom Corpus",
    ])
    def test_parse_custom_dataset_name(
        self, get_test_directory: Path, dataset_name: str
    ) -> None:
        """dataset_name is stored on the returned EvaluationDataset."""
        dataset = WikipediaUSAS.parse(
            get_test_directory / "spanish_wikipedia_empty.csv",
            dataset_name=dataset_name,
        )
        assert dataset.name == dataset_name

    # ------------------------------------------------------------------
    # language parameter
    # ------------------------------------------------------------------

    def test_parse_default_language_none(self, get_test_directory: Path) -> None:
        """Without language the returned dataset has language=None."""
        dataset = WikipediaUSAS.parse(get_test_directory / "spanish_wikipedia_empty.csv")
        assert dataset.language is None

    @pytest.mark.parametrize("language", ["Spanish", "English", "Welsh", "Irish"])
    def test_parse_custom_language(
        self, get_test_directory: Path, language: str
    ) -> None:
        """language is stored on the returned EvaluationDataset."""
        dataset = WikipediaUSAS.parse(
            get_test_directory / "spanish_wikipedia_empty.csv",
            language=language,
        )
        assert dataset.language == language

    def test_parse_dataset_name_and_language_combined(
        self, get_test_directory: Path
    ) -> None:
        """Both dataset_name and language can be supplied together."""
        dataset = WikipediaUSAS.parse(
            get_test_directory / "spanish_wikipedia_empty.csv",
            dataset_name="Medical Wikipedia",
            language="Spanish",
        )
        assert dataset.name == "Medical Wikipedia"
        assert dataset.language == "Spanish"

    # ------------------------------------------------------------------
    # Empty / one-token basics
    # ------------------------------------------------------------------

    def test_parse_empty(self, get_test_directory: Path) -> None:
        dataset = WikipediaUSAS.parse(
            get_test_directory / "spanish_wikipedia_empty.csv",
            dataset_name="Wikipedia USAS",
            language="Spanish",
        )
        assert isinstance(dataset, EvaluationDataset)
        assert dataset.name == "Wikipedia USAS"
        assert dataset.language == "Spanish"
        assert dataset.text_level == "sentence"
        assert len(dataset.texts) == 0
        assert dataset.labels_removed is None

    def test_parse_one_token(self, get_test_directory: Path) -> None:
        """Single sentence, single token — uses predicted USAS (no correction)."""
        dataset = WikipediaUSAS.parse(
            get_test_directory / "spanish_wikipedia_one_token.csv",
            dataset_name="Medical Wikipedia",
            language="Spanish",
        )
        assert len(dataset.texts) == 1
        text = dataset.texts[0]
        assert text.text == "Melanoma"
        assert text.tokens == ["Melanoma"]
        assert text.lemmas == ["melanoma"]
        assert text.pos_tags == ["PROPN"]
        assert text.semantic_tags == [["B2"]]
        assert text.mwe_indexes == [frozenset()]

    # ------------------------------------------------------------------
    # Corrected USAS with multiple semicolon-separated tags
    # ------------------------------------------------------------------

    def test_parse_multi_corrected_usas_uses_first_tag(
        self, get_test_directory: Path
    ) -> None:
        """When corrected USAS contains multiple ';'-separated tags (e.g. 'B3/A5 ; F1'),
        only the first tag ('B3/A5') is used as the resolved semantic tag."""
        dataset = WikipediaUSAS.parse(
            get_test_directory / "spanish_wikipedia_multi_corrected_usas.csv"
        )
        assert len(dataset.texts) == 1
        assert dataset.texts[0].semantic_tags == [["B3/A5"]]

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_parse_wrong_format(self, get_test_directory: Path) -> None:
        """A corrected USAS tag that cannot be parsed raises ValueError."""
        with pytest.raises(ValueError):
            WikipediaUSAS.parse(
                get_test_directory / "spanish_wikipedia_wrong_format.csv"
            )

    def test_parse_text_is_usas_label(self, get_test_directory: Path) -> None:
        """A token whose text is itself a valid USAS tag raises ValueError."""
        with pytest.raises(ValueError):
            WikipediaUSAS.parse(
                get_test_directory / "spanish_wikipedia_text_as_label.csv"
            )

    def test_parse_invalid_id_format(self, get_test_directory: Path) -> None:
        """A token ID that does not match <language>|<article>|<sentence_id>|<token_id> raises ValueError."""
        with pytest.raises(ValueError, match="does not match expected format"):
            WikipediaUSAS.parse(
                get_test_directory / "spanish_wikipedia_invalid_id_format.csv"
            )

    # ------------------------------------------------------------------
    # Small example — expected output fixture
    # ------------------------------------------------------------------

    @pytest.fixture(params=[None, {"B3"}])
    def small_example_expected(
        self, request: pytest.FixtureRequest
    ) -> tuple[EvaluationDataset, set[str] | None]:
        """
        Expected EvaluationDataset for the small example, parameterised by
        label_filter (None or {"B3"}).

        Sentence 0 — "La terapia es una tecnica ."
          - La      : predicted Z5 (no correction)
          - terapia : corrected B3 (overrides B3; Y1)
          - es      : corrected A3
          - una     : predicted Z5
          - tecnica : predicted X4.2
          - .       : PUNCT (via POS)

        Sentence 1 — "En general , conduce a la muerte ."
          - En/general: corrected A4.2, no MWE (predicted_mwe ignored; corrected_mwe empty)
          - ,          : PUNCT
          - conduce    : corrected A2.2
          - a/la       : predicted Z5
          - muerte     : corrected L1
          - .          : PUNCT

        Sentence 2 — "vasos sanguineos y mata de hambre ."
          - vasos/sanguineos : corrected B3, MWE from corrected (corrected_mwe=1)
          - y                : corrected Z5
          - mata/de/hambre   : corrected L1/F1 multi-tag, MWE from corrected (corrected_mwe=2)
          - .                : PUNCT
        """
        label_filter: set[str] | None = request.param

        s0_tags = [["Z5"], ["B3"], ["A3"], ["Z5"], ["X4.2"], ["PUNCT"]]
        s2_tags = [["B3"], ["B3"], ["Z5"], ["L1/F1"], ["L1/F1"], ["L1/F1"], ["PUNCT"]]

        if label_filter is not None:
            s0_tags = [["Z5"], [""], ["A3"], ["Z5"], ["X4.2"], ["PUNCT"]]
            s2_tags = [[""], [""], ["Z5"], ["L1/F1"], ["L1/F1"], ["L1/F1"], ["PUNCT"]]

        texts = [
            EvaluationTexts(
                text="La terapia es una tecnica .",
                tokens=["La", "terapia", "es", "una", "tecnica", "."],
                lemmas=["el", "terapia", "ser", "uno", "tecnica", "."],
                pos_tags=["DET", "NOUN", "AUX", "DET", "NOUN", "PUNCT"],
                semantic_tags=s0_tags,
                mwe_indexes=[frozenset()] * 6,
            ),
            EvaluationTexts(
                text="En general , conduce a la muerte .",
                tokens=["En", "general", ",", "conduce", "a", "la", "muerte", "."],
                lemmas=["en", "general", ",", "conducir", "a", "el", "muerte", "."],
                pos_tags=["ADP", "NOUN", "PUNCT", "VERB", "ADP", "DET", "NOUN", "PUNCT"],
                semantic_tags=[
                    ["A4.2"], ["A4.2"], ["PUNCT"],
                    ["A2.2"], ["Z5"], ["Z5"], ["L1"], ["PUNCT"],
                ],
                mwe_indexes=[frozenset()] * 8,
            ),
            EvaluationTexts(
                text="vasos sanguineos y mata de hambre .",
                tokens=["vasos", "sanguineos", "y", "mata", "de", "hambre", "."],
                lemmas=["vaso", "sanguineo", "y", "matar", "de", "hambre", "."],
                pos_tags=["NOUN", "ADJ", "CCONJ", "VERB", "ADP", "NOUN", "PUNCT"],
                semantic_tags=s2_tags,
                mwe_indexes=[
                    frozenset({1}), frozenset({1}), frozenset(),
                    frozenset({2}), frozenset({2}), frozenset({2}),
                    frozenset(),
                ],
            ),
        ]
        dataset = EvaluationDataset(
            name="Wikipedia USAS",
            text_level=TextLevel.sentence,
            labels_removed=label_filter,
            language="Spanish",
            texts=texts,
        )
        return dataset, label_filter

    @pytest.mark.parametrize(
        "data_file_name",
        [
            "spanish_wikipedia_small_example.csv",
            "spanish_wikipedia_small_with_extra_empty_lines.csv",
        ],
    )
    def test_parse_small_example(
        self,
        get_test_directory: Path,
        data_file_name: str,
        small_example_expected: tuple[EvaluationDataset, set[str] | None],
    ) -> None:
        expected, label_filter = small_example_expected
        dataset = WikipediaUSAS.parse(
            get_test_directory / data_file_name,
            label_filter=label_filter,
            language="Spanish",
        )

        assert len(dataset.texts) == 3
        for i in range(3):
            assert dataset.texts[i].text == expected.texts[i].text, f"text mismatch at index {i}"
            assert dataset.texts[i].tokens == expected.texts[i].tokens, f"tokens mismatch at index {i}"
            assert dataset.texts[i].lemmas == expected.texts[i].lemmas, f"lemmas mismatch at index {i}"
            assert dataset.texts[i].pos_tags == expected.texts[i].pos_tags, f"pos_tags mismatch at index {i}"
            assert dataset.texts[i].semantic_tags == expected.texts[i].semantic_tags, f"semantic_tags mismatch at index {i}"
            assert dataset.texts[i].mwe_indexes == expected.texts[i].mwe_indexes, f"mwe_indexes mismatch at index {i}"
        assert dataset.labels_removed == label_filter
        assert dataset.language == "Spanish"

    # ------------------------------------------------------------------
    # MWE index remapping
    # ------------------------------------------------------------------

    def test_parse_mwe_remapping(self, get_test_directory: Path) -> None:
        """corrected_mwe IDs that do not start at 1 are remapped to sequential integers starting at 1."""
        dataset = WikipediaUSAS.parse(
            get_test_directory / "spanish_wikipedia_mwe_remapping.csv"
        )
        assert len(dataset.texts) == 1
        mwe_indexes = dataset.texts[0].mwe_indexes
        # corrected_mwe values 3 and 5 must be remapped to 1 and 2 respectively
        assert mwe_indexes == [
            frozenset({1}), frozenset({1}),  # vasos/sanguineos — group 3 → 1
            frozenset(),                      # y — no MWE
            frozenset({2}), frozenset({2}), frozenset({2}),  # mata/de/hambre — group 5 → 2
            frozenset(),                      # .
        ]

    # ------------------------------------------------------------------
    # Label validation
    # ------------------------------------------------------------------

    @pytest.fixture(params=[False, True])
    def label_validation_and_error(
        self, request: pytest.FixtureRequest
    ) -> tuple[set[str], bool]:
        """All labels present → no error; remove A4.2 → error (sentence 1)."""
        all_labels = {"Z5", "B3", "A3", "X4.2", "A4.2", "A2.2", "L1", "F1"}
        if request.param:
            return all_labels, False
        else:
            one_less = copy.deepcopy(all_labels)
            one_less.remove("A4.2")
            return one_less, True

    @pytest.mark.parametrize("label_filter", [None, {"B3"}])
    def test_parse_label_validation(
        self,
        get_test_directory: Path,
        label_filter: set[str] | None,
        label_validation_and_error: tuple[set[str], bool],
    ) -> None:
        data_file = get_test_directory / "spanish_wikipedia_small_example.csv"
        validation_labels, to_error = label_validation_and_error
        if to_error:
            with pytest.raises(ValueError):
                WikipediaUSAS.parse(
                    data_file,
                    label_validation=validation_labels,
                    label_filter=label_filter,
                )
        else:
            dataset = WikipediaUSAS.parse(
                data_file,
                label_validation=validation_labels,
                label_filter=label_filter,
            )
            assert len(dataset.texts) == 3
            assert dataset.labels_removed == label_filter

    # ------------------------------------------------------------------
    # Full corpus smoke test
    # ------------------------------------------------------------------

    def test_parse_full_dataset(self) -> None:
        """Parse the full spanish.csv corpus and verify sentence/token counts."""
        data_file = Path(__file__).parent.parent.parent / "Data" / "spanish.csv"
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
                if "/" in tag_list[0]:
                    multi_tag_count += 1

        assert token_count > 0
        assert multi_tag_count > 0
