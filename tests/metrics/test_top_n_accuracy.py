"""Tests for the top_n_accuracy function."""
from typing import Literal

import pytest

from usas_evaluation_framework.dataset import (
    EvaluationDataset,
    EvaluationTexts,
    TextLevel,
)
from usas_evaluation_framework.metrics.top_n_accuracy import (
    top_n_accuracy,
    validate_top_n_inputs,
)


@pytest.mark.parametrize("n", [0, -1])
@pytest.mark.parametrize("average", ["micro", "macro"])
def test_validate_top_n_inputs_invalid_n(n: int, average: Literal["micro", "macro"]):
    """Test with n <= 0."""
    with pytest.raises(ValueError, match="n must be a positive integer"):
        validate_top_n_inputs(n, average)


def test_validate_top_n_inputs_invalid_average():
    """Test with average not in ['micro', 'macro']."""
    with pytest.raises(ValueError, match="average must be either 'micro' or 'macro'"):
        validate_top_n_inputs(1, "invalid")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("n", [1, 5])
@pytest.mark.parametrize("average", ["micro", "macro"])
def test_validate_top_n_inputs_valid(n: int, average: Literal["micro", "macro"]):
    """Test that valid inputs do not raise."""
    validate_top_n_inputs(n, average)


@pytest.mark.parametrize(
    ("n", "average", "pred_tokens", "true_semantic_tags", "expected_error"),
    [
        (0, "micro", ["token1"], [["tag1"]], "n must be a positive integer"),
        (1, "invalid", ["token1"], [["tag1"]], "average must be either 'micro' or 'macro'"),
        (1, "micro", ["token2"], [["tag1"]], "y_true and y_pred must have the same tokens in the same order"),
        (1, "macro", ["token1"], None, "y_true text 0 does not have semantic_tags"),
        (1, "macro", ["token1"], [[""]], "All of the true semantic tags are either empty strings or are empty lists"),
    ],
)
def test_top_n_accuracy_raises(n: int,
                               average: Literal["micro", "macro"],
                               pred_tokens: list[str],
                               true_semantic_tags: list[list[str]] | None,
                               expected_error: str):
    """
    Test that top_n_accuracy applies all of its validation.

    The validation functions themselves are tested in detail in their own tests,
    this checks that `top_n_accuracy` calls each of them.
    """
    true_text = EvaluationTexts(
        text="test",
        tokens=["token1"],
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=true_semantic_tags
    )
    pred_text = EvaluationTexts(
        text="test",
        tokens=pred_tokens,
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=[["tag1"]]
    )

    y_true = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[true_text])
    y_pred = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[pred_text])

    with pytest.raises(ValueError, match=expected_error):
        top_n_accuracy(y_true, y_pred, n=n, average=average)


@pytest.mark.parametrize("average", ["micro", "macro"])
def test_basic_functionality(average: Literal["micro", "macro"]):
    """Test with simple datasets where predictions match true labels exactly."""
    # Create true and predicted datasets
    true_text = EvaluationTexts(
        text="test",
        tokens=["token1", "token2"],
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=[["tag1"], ["tag2"]]
    )
    pred_text = EvaluationTexts(
        text="test",
        tokens=["token1", "token2"],
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=[["tag1"], ["tag2"]]
    )
    
    y_true = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[true_text])
    y_pred = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[pred_text])
    
    # Test with n=1
    accuracy = top_n_accuracy(y_true, y_pred, n=1, average=average)
    assert accuracy == 1.0


@pytest.mark.parametrize("average", ["micro", "macro"])
def test_empty_predictions(average: Literal["micro", "macro"]):
    """Test with datasets where predicted tags are empty."""
    true_text = EvaluationTexts(
        text="test",
        tokens=["token1"],
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=[["tag1"]]
    )
    pred_text = EvaluationTexts(
        text="test",
        tokens=["token1"],
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=[[]]
    )
    
    y_true = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[true_text])
    y_pred = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[pred_text])
    
    accuracy = top_n_accuracy(y_true, y_pred, n=1, average=average)
    assert accuracy == 0.0





@pytest.mark.parametrize("average", ["micro", "macro"])
def test_multiple_true_tags_per_token(average: Literal["micro", "macro"]):
    """Test with datasets where a token has multiple true tags."""
    true_text = EvaluationTexts(
        text="test",
        tokens=["token1"],
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=[["tag1", "tag2"]]
    )
    pred_text = EvaluationTexts(
        text="test",
        tokens=["token1"],
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=[["tag1", "tag2", "tag3"]]
    )
    
    y_true = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[true_text])
    y_pred = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[pred_text])
    
    accuracy = top_n_accuracy(y_true, y_pred, n=2, average=average)
    assert accuracy == 1.0

    accuracy = top_n_accuracy(y_true, y_pred, n=1, average=average)
    assert accuracy == 0.5


@pytest.mark.parametrize("average", ["micro", "macro"])
def test_n_larger_than_predicted_tags(average: Literal["micro", "macro"]):
    """Test with n larger than the number of predicted tags."""
    true_text = EvaluationTexts(
        text="test",
        tokens=["token1"],
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=[["tag1", "tag2"]]
    )
    pred_text = EvaluationTexts(
        text="test",
        tokens=["token1"],
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=[["tag1"]]
    )
    
    y_true = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[true_text])
    y_pred = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[pred_text])
    
    accuracy = top_n_accuracy(y_true, y_pred, n=5, average=average)
    assert accuracy == 0.5


@pytest.mark.parametrize("average", ["micro", "macro"])
def test_macro_micro_different_results(average: Literal["micro", "macro"]):
    """Test that macro and micro averaging produce different results."""
    # Create datasets where micro and macro should differ
    # 2 tokens with label "A", 1 token with label "B"
    true_text = EvaluationTexts(
        text="test",
        tokens=["token1", "token2", "token3"],
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=[["A"], ["A"], ["B"]]
    )
    # Predictions: all correct but with different ranks
    # token1: "A" is first (correct)
    # token2: "A" is second (correct but not in top-1)
    # token3: "B" is first (correct)
    pred_text = EvaluationTexts(
        text="test",
        tokens=["token1", "token2", "token3"],
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=[
            ["A", "C"],  # A is correct, C is wrong
            ["D", "A"],  # A is correct but not in top-1
            ["B", "E"]   # B is correct
        ]
    )
    
    y_true = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[true_text])
    y_pred = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[pred_text])
    
    # Test with n=1
    accuracy = top_n_accuracy(y_true, y_pred, n=1, average=average)
    
    match average:
        case "micro":
            assert accuracy == 2 / 3, f"Expected accuracy to be 2/3, got {accuracy}"
        case "macro":
            accuracy_a = 1 / 2
            accuracy_b = 1
            expected_accuracy = (accuracy_a + accuracy_b) / 2  # 0.75
            assert accuracy == expected_accuracy, f"Expected accuracy to be ((1/2 + 1) / 2 = 0.75), got {accuracy}"
    
    # Test with n=2 should be 1.0 no matter the average as all tokens are correct
    accuracy = top_n_accuracy(y_true, y_pred, n=2, average=average)
    assert accuracy == 1.0
