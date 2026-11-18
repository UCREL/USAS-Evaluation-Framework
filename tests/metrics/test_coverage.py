"""Tests for the coverage metric."""
from collections.abc import Callable, Iterable
from typing import Literal

import pytest

from usas_evaluation_framework.dataset import (
    EvaluationDataset,
    EvaluationTexts,
    TextLevel,
)
from usas_evaluation_framework.metrics.coverage import (
    coverage,
    is_excluded,
    iter_token_labels,
    type_coverage,
)


def make_text(tokens: list[str], semantic_tags: list[list[str]] | None) -> EvaluationTexts:
    """
    Create a text with only tokens and semantic tags.

    Args:
        tokens: The tokens of the text.
        semantic_tags: The semantic tags of the text.

    Returns:
        The text.
    """
    return EvaluationTexts(
        text=" ".join(tokens),
        tokens=tokens,
        lemmas=None,
        pos_tags=None,
        semantic_tags=semantic_tags,
        mwe_indexes=None,
    )


def make_dataset(texts: list[EvaluationTexts]) -> EvaluationDataset:
    """
    Create a sentence level dataset from the given texts.

    Args:
        texts: The texts of the dataset.

    Returns:
        The dataset.
    """
    return EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=texts)


@pytest.mark.parametrize(
    ("label_filter", "expected"),
    [
        (None, 6 / 7),
        (set(), 6 / 7),
        ({"Z99"}, 4 / 7),
        ({"Z99", "Z9"}, 3 / 7),
        # Whole tag matching, `F2` does not remove `F2/O2`.
        ({"F2"}, 6 / 7),
        ({"F2/O2"}, 5 / 7),
        ({"Z99", "Z9", "Z5", "A1", "F2/O2"}, 0.0),
    ],
)
def test_coverage(label_filter: set[str] | None, expected: float):
    """Test coverage across multiple texts with and without a label filter."""
    y_pred = make_dataset([
        make_text(["The", "cat", "sat", "."], [["Z5"], ["Z99"], ["A1", "Z99"], ["Z9"]]),
        make_text(["Hello", "drinks", "x"], [["Z99"], ["F2/O2"], [""]]),
    ])
    assert coverage(y_pred, label_filter) == pytest.approx(expected)


@pytest.mark.parametrize("label_filter", [None, {"Z99"}, {"Z99", "Z9"}])
def test_coverage_iterable_matches_dataset(label_filter: set[str] | None):
    """Test that a list or generator of texts gives the same result as the dataset."""
    texts = [
        make_text(["The", "cat", "sat", "."], [["Z5"], ["Z99"], ["A1", "Z99"], ["Z9"]]),
        make_text(["Hello", "x"], [["Z99"], [""]]),
    ]
    expected = coverage(make_dataset(texts), label_filter)
    assert coverage(texts, label_filter) == expected
    assert coverage((text for text in texts), label_filter) == expected


def test_coverage_does_not_modify_input():
    """Test that filtering does not change the semantic tags of the given dataset."""
    y_pred = make_dataset([make_text(["a", "b"], [["Z99"], ["A1", "Z99"]])])
    coverage(y_pred, {"Z99"})
    assert y_pred.texts[0].semantic_tags == [["Z99"], ["A1", "Z99"]]


@pytest.mark.parametrize("semantic_tags", [[[]], [[""]], [[" "]]])
def test_coverage_no_labels(semantic_tags: list[list[str]]):
    """Test that empty tag lists, empty strings and whitespace tags are not labels."""
    y_pred = make_dataset([make_text(["a"], semantic_tags)])
    assert coverage(y_pred) == 0.0


def test_coverage_missing_semantic_tags():
    """Test that a text without semantic tags raises."""
    y_pred = make_dataset([make_text(["a"], [["A1"]]), make_text(["b"], None)])
    with pytest.raises(ValueError, match="y_pred text 1 does not have semantic_tags"):
        coverage(y_pred)


def test_coverage_missing_semantic_tags_iterable():
    """Test that a streamed text without semantic tags raises."""
    texts = (text for text in [make_text(["a"], [["A1"]]), make_text(["b"], None)])
    with pytest.raises(ValueError, match="y_pred text 1 does not have semantic_tags"):
        coverage(texts)


@pytest.mark.parametrize("texts", [[], [make_text([], [])]])
def test_coverage_no_tokens(texts: list[EvaluationTexts]):
    """Test that a dataset, list or generator with no tokens raises."""
    for y_pred in (make_dataset(texts), texts, (text for text in texts)):
        with pytest.raises(ValueError, match="y_pred does not contain any tokens"):
            coverage(y_pred)


TYPE_TEXTS = [
    make_text(["The", "cat", "sat", "."], [["Z5"], ["L2"], ["Z99"], ["Z9"]]),
    # `cat` is unlabelled here but covered by its occurrence above.
    make_text(["the", "cat", "ran", "."], [["Z99"], ["Z99"], ["M1", "Z99"], ["Z9"]]),
]


@pytest.mark.parametrize(
    ("label_filter", "lowercase", "average", "expected"),
    [
        # Types: The, cat, sat, ., the, ran
        (None, False, "binary", 6 / 6),
        ({"Z99"}, False, "binary", 4 / 6),
        ({"Z99", "Z9"}, False, "binary", 3 / 6),
        # Types: the, cat, sat, ., ran. `the` is covered by `The`.
        ({"Z99"}, True, "binary", 4 / 5),
        ({"Z99", "Z9"}, True, "binary", 3 / 5),
        (None, False, "macro", 6 / 6),
        # The=1, cat=1/2, sat=0, .=1, the=0, ran=1
        ({"Z99"}, False, "macro", 3.5 / 6),
        # The=1, cat=1/2, sat=0, .=0, the=0, ran=1
        ({"Z99", "Z9"}, False, "macro", 2.5 / 6),
        # the=1/2, cat=1/2, sat=0, .=1, ran=1
        ({"Z99"}, True, "macro", 3 / 5),
        # the=1/2, cat=1/2, sat=0, .=0, ran=1
        ({"Z99", "Z9"}, True, "macro", 2 / 5),
    ],
)
def test_type_coverage(label_filter: set[str] | None,
                       lowercase: bool,
                       average: Literal["binary", "macro"],
                       expected: float):
    """Test type coverage with a dataset, list and generator of texts."""
    for y_pred in (make_dataset(TYPE_TEXTS), TYPE_TEXTS, (text for text in TYPE_TEXTS)):
        result = type_coverage(y_pred, label_filter=label_filter, lowercase=lowercase, average=average)
        assert result == pytest.approx(expected)


def test_type_coverage_macro_one_type_per_token_equals_coverage():
    """Test that when every token is a different word type macro equals token coverage."""
    y_pred = make_dataset([make_text(["a", "b", "c", "d"], [["A1"], ["Z99"], ["B2"], [""]])])
    assert type_coverage(y_pred, {"Z99"}, average="macro") == coverage(y_pred, {"Z99"})


def test_type_coverage_invalid_average():
    """Test that an invalid average raises."""
    y_pred = make_dataset([make_text(["a"], [["A1"]])])
    with pytest.raises(ValueError, match="average must be either 'binary' or 'macro'"):
        type_coverage(y_pred, average="micro")  # ty: ignore[invalid-argument-type]


def test_type_coverage_missing_semantic_tags():
    """Test that a text without semantic tags raises."""
    y_pred = make_dataset([make_text(["a"], [["A1"]]), make_text(["b"], None)])
    with pytest.raises(ValueError, match="y_pred text 1 does not have semantic_tags"):
        type_coverage(y_pred)


@pytest.mark.parametrize("texts", [[], [make_text([], [])]])
def test_type_coverage_no_tokens(texts: list[EvaluationTexts]):
    """Test that input with no tokens raises."""
    with pytest.raises(ValueError, match="y_pred does not contain any tokens"):
        type_coverage(make_dataset(texts))


@pytest.mark.parametrize(
    "to_input",
    [make_dataset, list, iter],
    ids=["dataset", "list", "iterator"],
)
@pytest.mark.parametrize(
    ("label_filter", "expected_labelled"),
    [
        (None, [True] * 8),
        ({"Z99", "Z9"}, [True, True, False, False, False, False, True, False]),
    ],
)
def test_iter_token_labels(to_input: Callable[[list[EvaluationTexts]], EvaluationDataset | Iterable[EvaluationTexts]],
                           label_filter: set[str] | None,
                           expected_labelled: list[bool]):
    """Test that tokens are yielded in order across texts with their labelled state, for each input type."""
    expected_tokens = [token for text in TYPE_TEXTS for token in text.tokens]
    expected = list(zip(expected_tokens, expected_labelled, strict=True))
    assert list(iter_token_labels(to_input(TYPE_TEXTS), label_filter)) == expected


COVERAGE_TEXTS = [
    make_text(["The", "cat", "sat", "."], [["Z5"], ["Z99"], ["A1", "Z99"], ["Z9"]]),
    make_text(["Hello", "drinks", "x"], [["Z99"], ["F2/O2"], [""]]),
]


@pytest.mark.parametrize(
    ("label_filter", "exclude_labels", "expected"),
    [
        # `.` is excluded: 6 tokens, all but `x` labelled. Filtering `Z9` instead gives 6 / 7.
        (None, {"Z9"}, 5 / 6),
        # `.` is excluded: 6 tokens, The, sat, drinks labelled.
        ({"Z99"}, {"Z9"}, 3 / 6),
        # cat and Hello are excluded, sat is not as it also has `A1`: The, sat, ., drinks, x.
        (None, {"Z99"}, 4 / 5),
        # The, sat, drinks, x. `x` has no tags so it is never excluded.
        (None, {"Z9", "Z99"}, 3 / 4),
        # Whole tag matching, `F2` does not exclude `F2/O2`.
        (None, {"F2"}, 6 / 7),
        (None, set(), 6 / 7),
    ],
)
def test_coverage_exclude_labels(label_filter: set[str] | None, exclude_labels: set[str], expected: float):
    """Test that excluded tokens are left out of both the numerator and denominator."""
    for y_pred in (make_dataset(COVERAGE_TEXTS), COVERAGE_TEXTS, (text for text in COVERAGE_TEXTS)):
        result = coverage(y_pred, label_filter=label_filter, exclude_labels=exclude_labels)
        assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("lowercase", "average", "expected"),
    [
        # `.` disappears. Types: The, cat, sat, the, ran
        (False, "binary", 3 / 5),
        # The=1, cat=1/2, sat=0, the=0, ran=1
        (False, "macro", 2.5 / 5),
        # Types: the, cat, sat, ran
        (True, "binary", 3 / 4),
        # the=1/2, cat=1/2, sat=0, ran=1
        (True, "macro", 2 / 4),
    ],
)
def test_type_coverage_exclude_labels(lowercase: bool, average: Literal["binary", "macro"], expected: float):
    """Test that a word type whose every occurrence is excluded is not a word type."""
    for y_pred in (make_dataset(TYPE_TEXTS), TYPE_TEXTS, (text for text in TYPE_TEXTS)):
        result = type_coverage(y_pred, label_filter={"Z99"}, exclude_labels={"Z9"},
                                            lowercase=lowercase, average=average)
        assert result == pytest.approx(expected)


def test_type_coverage_exclude_labels_partial_word_type():
    """Test that excluded occurrences do not count towards a word type's macro score."""
    y_pred = make_dataset([make_text(["x", "x", "x"], [["Z9"], ["A1"], ["Z99"]])])
    # Filtering `Z9` counts that occurrence against the word: one of three labelled.
    assert type_coverage(y_pred, label_filter={"Z99", "Z9"}, average="macro") == pytest.approx(1 / 3)
    # Excluding the `Z9` occurrence instead leaves one of two.
    result = type_coverage(y_pred, label_filter={"Z99"}, exclude_labels={"Z9"}, average="macro")
    assert result == pytest.approx(1 / 2)


@pytest.mark.parametrize("metric", [coverage, type_coverage])
def test_coverage_all_tokens_excluded(metric: Callable[..., float]):
    """Test that excluding every token raises."""
    y_pred = make_dataset([make_text([".", "!"], [["Z9"], ["Z9", ""]])])
    with pytest.raises(ValueError, match="y_pred does not contain any tokens that are not excluded"):
        metric(y_pred, exclude_labels={"Z9"})


@pytest.mark.parametrize(
    "metric",
    [coverage, type_coverage, lambda *args, **kwargs: list(iter_token_labels(*args, **kwargs))],
    ids=["coverage", "type_coverage", "iter_token_labels"],
)
def test_label_filter_and_exclude_labels_overlap(metric: Callable[..., object]):
    """Test that a tag in both label_filter and exclude_labels raises, naming the shared tags."""
    y_pred = make_dataset([make_text(["a"], [["A1"]])])
    with pytest.raises(ValueError, match=r"must not share any tags, but both contain: \['Z9', 'Z99'\]"):
        metric(y_pred, label_filter={"Z99", "Z9", "F2"}, exclude_labels={"Z9", "Z99"})


def test_iter_token_labels_exclude_labels():
    """Test that excluded tokens are not yielded."""
    assert list(iter_token_labels(TYPE_TEXTS, label_filter={"Z99"}, exclude_labels={"Z9"})) == [
        ("The", True), ("cat", True), ("sat", False),
        ("the", False), ("cat", False), ("ran", True),
    ]


@pytest.mark.parametrize(
    ("token_tags", "exclude_labels", "expected"),
    [
        (["Z9"], {"Z9"}, True),
        (["Z9", "Z99"], {"Z9", "Z99"}, True),
        # Empty and whitespace tags are ignored when checking every tag.
        (["Z9", "", " "], {"Z9"}, True),
        (["Z9", "A1"], {"Z9"}, False),
        (["A1"], {"Z9"}, False),
        # Whole tag matching, `Z9` does not exclude `Z9/Z99`.
        (["Z9/Z99"], {"Z9"}, False),
        # Tokens with no tags are never excluded.
        ([], {"Z9"}, False),
        ([""], {"Z9"}, False),
        (["Z9"], set(), False),
        (["Z9"], frozenset({"Z9"}), True),
    ],
)
def test_is_excluded(token_tags: list[str], exclude_labels: set[str] | frozenset[str], expected: bool):
    """Test which tokens are excluded."""
    assert is_excluded(token_tags, exclude_labels) is expected
