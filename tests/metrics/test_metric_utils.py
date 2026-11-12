"""Tests for the shared metric utility functions."""
import pytest

from usas_evaluation_framework.dataset import (
    EvaluationDataset,
    EvaluationTexts,
    TextLevel,
)
from usas_evaluation_framework.metrics.metric_utils import (
    collect_and_filter_tags,
    validate_datasets,
)


def make_dataset(texts: list[tuple[list[str], list[list[str]] | None]]) -> EvaluationDataset:
    """
    Create a sentence level dataset from (tokens, semantic tags) pairs.

    Args:
        texts: One (tokens, semantic tags) pair per text.

    Returns:
        The dataset containing the given texts.
    """
    evaluation_texts = [
        EvaluationTexts(
            text=" ".join(tokens),
            tokens=tokens,
            lemmas=None,
            pos_tags=None,
            mwe_indexes=None,
            semantic_tags=semantic_tags
        )
        for tokens, semantic_tags in texts
    ]
    return EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=evaluation_texts)


def test_validate_datasets_valid():
    """Test that comparable datasets do not raise."""
    y_true = make_dataset([(["token1"], [["tag1"]]), (["token2"], [["tag2"]])])
    y_pred = make_dataset([(["token1"], [["tag3"]]), (["token2"], [[]])])
    validate_datasets(y_true, y_pred)


def test_validate_datasets_mismatched_number_of_texts():
    """Test with datasets that have a different number of texts."""
    y_true = make_dataset([(["token1"], [["tag1"]]), (["token2"], [["tag2"]])])
    y_pred = make_dataset([(["token1"], [["tag1"]])])

    with pytest.raises(ValueError, match="y_true and y_pred must have the same number of texts"):
        validate_datasets(y_true, y_pred)


def test_validate_datasets_different_tokens():
    """Test with datasets that have the same number of texts but different tokens."""
    y_true = make_dataset([(["token1"], [["tag1"]])])
    y_pred = make_dataset([(["token2"], [["tag1"]])])

    with pytest.raises(ValueError, match="y_true and y_pred must have the same tokens in the same order"):
        validate_datasets(y_true, y_pred)


@pytest.mark.parametrize("true_missing_semantic_tags", [True, False])
def test_validate_datasets_missing_semantic_tags(true_missing_semantic_tags: bool):
    """Test with datasets where the semantic tags of a text are None."""
    match true_missing_semantic_tags:
        case True:
            y_true = make_dataset([(["token1"], None)])
            y_pred = make_dataset([(["token1"], [["tag1"]])])
            expected_error = "y_true text 0 does not have semantic_tags"
        case False:
            y_true = make_dataset([(["token1"], [["tag1"]])])
            y_pred = make_dataset([(["token1"], None)])
            expected_error = "y_pred text 0 does not have semantic_tags"

    with pytest.raises(ValueError, match=expected_error):
        validate_datasets(y_true, y_pred)


def test_collect_and_filter_tags():
    """
    Test that empty strings are removed and tokens without true tags are skipped.

    The predicted tags for a skipped token are also skipped, and the order of
    the predicted tags is preserved.
    """
    y_true = make_dataset([
        (["token1", "token2", "token3"], [["tag1", ""], [""], ["tag2"]]),
        (["token4"], [["tag3", "tag4"]]),
    ])
    y_pred = make_dataset([
        (["token1", "token2", "token3"], [["", "tag5", "tag1"], ["tag6"], []]),
        (["token4"], [["tag4", "tag3"]]),
    ])

    true_tags_list, pred_tags_list = collect_and_filter_tags(y_true, y_pred)

    assert true_tags_list == [["tag1"], ["tag2"], ["tag3", "tag4"]]
    assert pred_tags_list == [["tag5", "tag1"], [], ["tag4", "tag3"]]


@pytest.mark.parametrize("empty_true_tags", [[[]], [[""]], [[], [""]]])
def test_collect_and_filter_tags_empty_true_labels(empty_true_tags: list[list[str]]):
    """Test with datasets where all of the true tags are empty lists or empty strings."""
    tokens = [f"token{i}" for i in range(len(empty_true_tags))]
    y_true = make_dataset([(tokens, empty_true_tags)])
    y_pred = make_dataset([(tokens, [["tag1"] for _ in tokens])])

    with pytest.raises(ValueError, match="All of the true semantic tags are either empty strings or are empty lists"):
        collect_and_filter_tags(y_true, y_pred)
