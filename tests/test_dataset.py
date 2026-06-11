"""Tests for the dataset module."""
import copy
from typing import Literal, TypedDict

import pytest

from usas_evaluation_framework.dataset import (
    DatasetStats,
    EvaluationDataset,
    EvaluationTexts,
    TextLevel,
)


class EvaluationTextsData(TypedDict):
    text: str
    tokens: list[str]
    lemmas: list[str]
    pos_tags: list[str]
    semantic_tags: list[list[str]]
    mwe_indexes: list[frozenset[int]]

def test_text_level_enum_values() -> None:
    """Test that TextLevel enum has the correct values."""
    assert TextLevel.sentence.value == "sentence"
    assert TextLevel.paragraph.value == "paragraph"
    assert TextLevel.document.value == "document"


def test_text_level_enum_members() -> None:
    """Test that TextLevel enum has the correct members."""
    assert TextLevel.sentence in TextLevel
    assert TextLevel.paragraph in TextLevel
    assert TextLevel.document in TextLevel


def test_text_level_enum_iteration() -> None:
    """Test that TextLevel enum can be iterated over."""
    members = list(TextLevel)
    assert len(members) == 3
    assert TextLevel.sentence in members
    assert TextLevel.paragraph in members
    assert TextLevel.document in members


def test_text_level_enum_from_string() -> None:
    """Test that TextLevel enum can be created from strings."""
    assert TextLevel("sentence") == TextLevel.sentence
    assert TextLevel("paragraph") == TextLevel.paragraph
    assert TextLevel("document") == TextLevel.document


def test_text_level_enum_invalid_value() -> None:
    """Test that TextLevel enum raises ValueError for invalid values."""
    try:
        TextLevel("invalid")
        assert False, "Expected ValueError for invalid enum value"
    except ValueError:
        pass


@pytest.fixture
def evaluation_texts_data() -> EvaluationTextsData:
    return {
        "text": "This is a test sentence.",
        "tokens": ["This", "is", "a", "test", "sentence", "."],
        "lemmas": ["This", "be", "a", "test", "sentence", "."],
        "pos_tags": ["DT", "VBZ", "DT", "NN", "NN", "."],
        "semantic_tags": [["Z1"], ["Z2"], ["Z3"], ["Z4"], ["Z5"], ["Z6"]],
        "mwe_indexes": [frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5}), frozenset({6})]
    } 


def test_evaluation_texts_valid_initialization(evaluation_texts_data: EvaluationTextsData) -> None:
    texts = EvaluationTexts(
        text=evaluation_texts_data["text"],
        tokens=evaluation_texts_data["tokens"],
        lemmas=evaluation_texts_data["lemmas"],
        pos_tags=evaluation_texts_data["pos_tags"],
        semantic_tags=evaluation_texts_data["semantic_tags"],
        mwe_indexes=evaluation_texts_data["mwe_indexes"]
    )
    assert texts.text == evaluation_texts_data["text"]
    assert texts.tokens == evaluation_texts_data["tokens"]
    assert texts.lemmas == evaluation_texts_data["lemmas"]
    assert texts.pos_tags == evaluation_texts_data["pos_tags"]
    assert texts.semantic_tags == evaluation_texts_data["semantic_tags"]
    assert texts.mwe_indexes == evaluation_texts_data["mwe_indexes"]


def test_evaluation_texts__eq__(evaluation_texts_data: EvaluationTextsData) -> None:
    expected_evaluation_texts = EvaluationTexts(**evaluation_texts_data)
    assert expected_evaluation_texts == expected_evaluation_texts
    incorrect_key_values = [
        ("text", "Different text"),
        ("tokens", ["Different tokens"] * len(evaluation_texts_data["tokens"])),
        ("lemmas", ["Different lemmas"]* len(evaluation_texts_data["tokens"])),
        ("pos_tags", ["Different pos tags"]* len(evaluation_texts_data["tokens"])),
        ("semantic_tags", [["Different semantic tags"]]* len(evaluation_texts_data["tokens"])),
        ("mwe_indexes", [frozenset({})]* len(evaluation_texts_data["tokens"]))
    ]
    for incorrect_key, incorrect_value in incorrect_key_values:
        temp_evaluation_texts_data = copy.deepcopy(evaluation_texts_data)
        temp_evaluation_texts_data[incorrect_key] = incorrect_value
        assert expected_evaluation_texts != EvaluationTexts(**temp_evaluation_texts_data)

    alt_evaluation_texts_data = dict(copy.deepcopy(evaluation_texts_data))
    alt_evaluation_texts_data["lemmas"] = None
    alt_evaluation_texts_data["pos_tags"] = None
    alt_evaluation_texts = EvaluationTexts(**alt_evaluation_texts_data)
    assert alt_evaluation_texts == alt_evaluation_texts

    assert alt_evaluation_texts != expected_evaluation_texts


@pytest.mark.parametrize("list_attribute_testing", ["lemmas", "pos_tags", "semantic_tags", "mwe_indexes"])
def test_evaluation_texts_mismatched_lengths(list_attribute_testing: Literal["lemmas", "pos_tags", "semantic_tags", "mwe_indexes"],
                                             evaluation_texts_data: EvaluationTextsData) -> None:
    lemmas = evaluation_texts_data["lemmas"]
    pos_tags = evaluation_texts_data["pos_tags"]
    semantic_tags = evaluation_texts_data["semantic_tags"]
    mwe_indexes = evaluation_texts_data["mwe_indexes"]
    match list_attribute_testing:
        case "lemmas":
            lemmas.pop()
        case "pos_tags":
            pos_tags.pop()
        case "semantic_tags":
            semantic_tags.pop()
        case "mwe_indexes":
            mwe_indexes.pop()
    with pytest.raises(ValueError):
        EvaluationTexts(
            text=evaluation_texts_data["text"],
            tokens=evaluation_texts_data["tokens"],
            lemmas=lemmas,
            pos_tags=pos_tags,
            semantic_tags=semantic_tags,
            mwe_indexes=mwe_indexes
        )


def test_evaluation_texts_none_values(evaluation_texts_data: EvaluationTextsData) -> None:
    texts = EvaluationTexts(
        text=evaluation_texts_data['text'],
        tokens=evaluation_texts_data['tokens'],
        lemmas=None,
        pos_tags=None,
        semantic_tags=None,
        mwe_indexes=None
    )
    assert texts.text == evaluation_texts_data['text']
    assert texts.tokens == evaluation_texts_data['tokens']
    assert texts.lemmas is None
    assert texts.pos_tags is None
    assert texts.semantic_tags is None
    assert texts.mwe_indexes is None


@pytest.mark.parametrize("language", [None, "Spanish", "English", "Welsh", "Irish"])
def test_evaluation_dataset_language(language: str | None) -> None:
    """EvaluationDataset.language round-trips the supplied value (including None)."""
    dataset = EvaluationDataset(
        name="Test Dataset",
        text_level=TextLevel.sentence,
        texts=[],
        language=language,
    )
    assert dataset.language == language


@pytest.mark.parametrize("language_a,language_b", [
    ("Spanish", "English"),
    ("Spanish", None),
    (None, "English"),
])
def test_evaluation_dataset_language_in_equality(
    language_a: str | None,
    language_b: str | None,
    evaluation_texts_data: EvaluationTextsData,
) -> None:
    """Two EvaluationDatasets with different language values are not equal."""
    texts = [EvaluationTexts(**evaluation_texts_data)]
    dataset_a = EvaluationDataset(
        name="Test Dataset", text_level=TextLevel.sentence, texts=texts, language=language_a
    )
    dataset_b = EvaluationDataset(
        name="Test Dataset", text_level=TextLevel.sentence, texts=texts, language=language_b
    )
    assert dataset_a != dataset_b


def test_evaluation_dataset_valid_initialization(evaluation_texts_data: EvaluationTextsData) -> None:
    texts = [
        EvaluationTexts(
            **evaluation_texts_data
        )
    ]
    dataset = EvaluationDataset(
        name="Test Dataset",
        text_level=TextLevel.sentence,
        labels_removed={"Z1", "Z2"},
        texts=texts
    )
    assert dataset.name == "Test Dataset"
    assert dataset.text_level == TextLevel.sentence
    assert dataset.labels_removed == {"Z1", "Z2"}
    assert dataset.texts == texts


def test_evaluation_dataset_empty_texts() -> None:
    dataset = EvaluationDataset(
        name="Test Dataset",
        text_level=TextLevel.sentence,
        labels_removed=None,
        texts=[]
    )
    assert dataset.name == "Test Dataset"
    assert dataset.text_level == TextLevel.sentence
    assert dataset.labels_removed is None
    assert dataset.texts == []

def test_evaluation_dataset_len_empty() -> None:
    """Test that __len__ returns 0 for an empty dataset."""
    dataset = EvaluationDataset(
        name="Test Dataset",
        text_level=TextLevel.sentence,
        texts=[]
    )
    assert len(dataset) == 0


def test_evaluation_dataset_len_single(evaluation_texts_data: EvaluationTextsData) -> None:
    """Test that __len__ returns 1 for a dataset with one text."""
    texts = [EvaluationTexts(**evaluation_texts_data)]
    dataset = EvaluationDataset(
        name="Test Dataset",
        text_level=TextLevel.sentence,
        texts=texts
    )
    assert len(dataset) == 1


def test_evaluation_dataset_len_multiple(evaluation_texts_data: EvaluationTextsData) -> None:
    """Test that __len__ returns the correct count for a dataset with multiple texts."""
    texts = [
        EvaluationTexts(**evaluation_texts_data),
        EvaluationTexts(**evaluation_texts_data),
        EvaluationTexts(**evaluation_texts_data)
    ]
    dataset = EvaluationDataset(
        name="Test Dataset",
        text_level=TextLevel.sentence,
        texts=texts
    )
    assert len(dataset) == 3


def test_text_tokens_equal_identical_datasets(evaluation_texts_data: EvaluationTextsData) -> None:
    """Test that text_tokens_equal returns True for identical datasets."""
    texts = [EvaluationTexts(**evaluation_texts_data)]
    dataset1 = EvaluationDataset(
        name="Dataset 1",
        text_level=TextLevel.sentence,
        texts=texts
    )
    dataset2 = EvaluationDataset(
        name="Dataset 2",
        text_level=TextLevel.sentence,
        texts=texts
    )
    assert dataset1.text_tokens_equal(dataset2) is True


def test_text_tokens_equal_different_tokens(evaluation_texts_data: EvaluationTextsData) -> None:
    """Test that text_tokens_equal returns False for datasets with different tokens."""
    texts1 = [EvaluationTexts(**evaluation_texts_data)]
    dataset1 = EvaluationDataset(
        name="Dataset 1",
        text_level=TextLevel.sentence,
        texts=texts1
    )
    
    # Create a copy with different tokens (very similar but not the same)
    different_texts_data = copy.deepcopy(evaluation_texts_data)
    different_texts_data["tokens"] = ["This", "is", "a", "test", "sentences", "."]
    texts2 = [EvaluationTexts(**different_texts_data)]
    dataset2 = EvaluationDataset(
        name="Dataset 2",
        text_level=TextLevel.sentence,
        texts=texts2
    )
    assert dataset1.text_tokens_equal(dataset2) is False


def test_text_tokens_equal_different_lengths(evaluation_texts_data: EvaluationTextsData) -> None:
    """Test that text_tokens_equal returns False for datasets with different lengths."""
    texts1 = [EvaluationTexts(**evaluation_texts_data)]
    dataset1 = EvaluationDataset(
        name="Dataset 1",
        text_level=TextLevel.sentence,
        texts=texts1
    )
    
    texts2 = [
        EvaluationTexts(**evaluation_texts_data),
        EvaluationTexts(**evaluation_texts_data)
    ]
    dataset2 = EvaluationDataset(
        name="Dataset 2",
        text_level=TextLevel.sentence,
        texts=texts2
    )
    assert dataset1.text_tokens_equal(dataset2) is False


def test_text_tokens_equal_empty_datasets() -> None:
    """Test that text_tokens_equal returns True for empty datasets."""
    dataset1 = EvaluationDataset(
        name="Dataset 1",
        text_level=TextLevel.sentence,
        texts=[]
    )
    dataset2 = EvaluationDataset(
        name="Dataset 2",
        text_level=TextLevel.sentence,
        texts=[]
    )
    assert dataset1.text_tokens_equal(dataset2) is True


def test_text_tokens_equal_mixed_matching(evaluation_texts_data: EvaluationTextsData) -> None:
    """Test that text_tokens_equal returns False when some texts match and others don't."""
    texts1 = [
        EvaluationTexts(**evaluation_texts_data),
        EvaluationTexts(**evaluation_texts_data)
    ]
    dataset1 = EvaluationDataset(
        name="Dataset 1",
        text_level=TextLevel.sentence,
        texts=texts1
    )

    # Create a copy with different tokens for the second text (very similar but not the same)
    different_texts_data = copy.deepcopy(evaluation_texts_data)
    different_texts_data["tokens"] = ["This", "is", "a", "test", "sentences", "."]
    texts2 = [
        EvaluationTexts(**evaluation_texts_data),
        EvaluationTexts(**different_texts_data)
    ]
    dataset2 = EvaluationDataset(
        name="Dataset 2",
        text_level=TextLevel.sentence,
        texts=texts2
    )
    assert dataset1.text_tokens_equal(dataset2) is False


# --- stats() tests ---

def _make_dataset(*texts: EvaluationTexts) -> EvaluationDataset:
    return EvaluationDataset(name="Test", text_level=TextLevel.sentence, texts=list(texts))


def _make_text(
    tokens: list[str],
    semantic_tags: list[list[str]] | None = None,
    mwe_indexes: list[frozenset[int]] | None = None,
) -> EvaluationTexts:
    return EvaluationTexts(
        text=" ".join(tokens),
        tokens=tokens,
        lemmas=None,
        pos_tags=None,
        semantic_tags=semantic_tags,
        mwe_indexes=mwe_indexes,
    )


def test_stats_empty_dataset() -> None:
    """Empty dataset returns zero texts/tokens and None for unannotated fields."""
    stats = _make_dataset().stats()
    assert stats == DatasetStats(
        num_texts=0,
        num_tokens=0,
        num_semantic_tags=None,
        num_labelled_tokens=None,
        num_compound_semantic_tags=None,
        unique_semantic_tags=None,
        num_mwes=None,
    )


def test_stats_no_annotations() -> None:
    """Dataset with no semantic_tags or mwe_indexes returns None for those fields."""
    dataset = _make_dataset(
        _make_text(["Hello", "world"]),
        _make_text(["Foo", "bar", "baz"]),
    )
    stats = dataset.stats()
    assert stats.num_texts == 2
    assert stats.num_tokens == 5
    assert stats.num_semantic_tags is None
    assert stats.num_labelled_tokens is None
    assert stats.num_compound_semantic_tags is None
    assert stats.unique_semantic_tags is None
    assert stats.num_mwes is None


def test_stats_token_count_multiple_texts(evaluation_texts_data: EvaluationTextsData) -> None:
    """num_tokens sums tokens across all texts."""
    dataset = _make_dataset(
        EvaluationTexts(**evaluation_texts_data),
        EvaluationTexts(**evaluation_texts_data),
    )
    assert dataset.stats().num_tokens == 12  # 6 tokens × 2 texts


def test_stats_semantic_tags_simple() -> None:
    """num_semantic_tags counts individual tag strings, not tokens."""
    dataset = _make_dataset(
        _make_text(["a", "b", "c"], semantic_tags=[["A1"], ["B2"], ["C3"]]),
    )
    stats = dataset.stats()
    assert stats.num_semantic_tags == 3
    assert stats.num_labelled_tokens == 3
    assert stats.num_compound_semantic_tags == 0
    assert stats.unique_semantic_tags == frozenset({"A1", "B2", "C3"})


def test_stats_semantic_tags_multiple_per_token() -> None:
    """Tokens with multiple candidate tags are each counted individually."""
    dataset = _make_dataset(
        _make_text(["a", "b"], semantic_tags=[["A1", "B2"], ["C3"]]),
    )
    stats = dataset.stats()
    assert stats.num_semantic_tags == 3
    assert stats.num_labelled_tokens == 2
    assert stats.unique_semantic_tags == frozenset({"A1", "B2", "C3"})


def test_stats_compound_semantic_tags() -> None:
    """Tags containing '/' are counted as compound tags."""
    dataset = _make_dataset(
        _make_text(["a", "b", "c"], semantic_tags=[["A1/B2"], ["C3"], ["D4/E5/F6"]]),
    )
    stats = dataset.stats()
    assert stats.num_semantic_tags == 3
    assert stats.num_labelled_tokens == 3
    assert stats.num_compound_semantic_tags == 2


def test_stats_unique_semantic_tags_deduplication() -> None:
    """unique_semantic_tags deduplicates the same tag appearing across multiple texts."""
    dataset = _make_dataset(
        _make_text(["a", "b"], semantic_tags=[["A1"], ["B2"]]),
        _make_text(["c", "d"], semantic_tags=[["A1"], ["C3"]]),
    )
    stats = dataset.stats()
    assert stats.num_semantic_tags == 4  # total tag strings, including duplicates
    assert stats.num_labelled_tokens == 4
    assert stats.unique_semantic_tags == frozenset({"A1", "B2", "C3"})


def test_stats_labelled_tokens_partial() -> None:
    """num_labelled_tokens counts only tokens with at least one tag, not those with empty lists."""
    dataset = _make_dataset(
        _make_text(["a", "b", "c"], semantic_tags=[["A1"], [], ["C3"]]),
    )
    stats = dataset.stats()
    assert stats.num_labelled_tokens == 2
    assert stats.num_semantic_tags == 2


def test_stats_mwe_count_distinct_per_text() -> None:
    """num_mwes counts distinct MWE IDs per text and sums across texts."""
    dataset = _make_dataset(
        _make_text(
            ["a", "b", "c", "d"],
            mwe_indexes=[frozenset(), frozenset({1}), frozenset({1}), frozenset({2})],
        ),
        _make_text(
            ["e", "f", "g"],
            mwe_indexes=[frozenset({1}), frozenset({1}), frozenset({1})],
        ),
    )
    # text 1 has MWE IDs {1, 2} → 2; text 2 has MWE IDs {1} → 1; total = 3
    assert dataset.stats().num_mwes == 3


def test_stats_mwe_all_single_tokens() -> None:
    """Tokens not part of any MWE (empty frozensets) yield num_mwes=0."""
    dataset = _make_dataset(
        _make_text(["a", "b", "c"], mwe_indexes=[frozenset(), frozenset(), frozenset()]),
    )
    assert dataset.stats().num_mwes == 0


def test_stats_uses_fixture_data(evaluation_texts_data: EvaluationTextsData) -> None:
    """Sanity-check stats against the shared fixture with known values."""
    dataset = _make_dataset(EvaluationTexts(**evaluation_texts_data))
    stats = dataset.stats()
    assert stats.num_texts == 1
    assert stats.num_tokens == 6
    assert stats.num_semantic_tags == 6   # one tag per token
    assert stats.num_labelled_tokens == 6
    assert stats.num_compound_semantic_tags == 0
    assert stats.unique_semantic_tags == frozenset({"Z1", "Z2", "Z3", "Z4", "Z5", "Z6"})
    assert stats.num_mwes == 6  # each token is its own MWE (IDs 1–6)