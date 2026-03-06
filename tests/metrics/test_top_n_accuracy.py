"""Tests for the top_n_accuracy function."""
from typing import Literal

import pytest

from usas_evaluation_framework.dataset import (
    EvaluationDataset,
    EvaluationTexts,
    TextLevel,
)
from usas_evaluation_framework.metrics.top_n_accuracy import top_n_accuracy


@pytest.mark.parametrize("average", ["micro", "macro"])
def test_validate_inputs_invalid_n(average: Literal["micro", "macro"]):
    """Test with n <= 0."""
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
        semantic_tags=[["tag1"]]
    )
    
    y_true = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[true_text])
    y_pred = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[pred_text])
    
    with pytest.raises(ValueError, match="n must be a positive integer"):
        top_n_accuracy(y_true, y_pred, n=0, average=average)


def test_validate_inputs_invalid_average():
    """Test with average not in ['micro', 'macro']."""
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
        semantic_tags=[["tag1"]]
    )
    
    y_true = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[true_text])
    y_pred = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[pred_text])
    
    with pytest.raises(ValueError, match="average must be either 'micro' or 'macro'"):
        top_n_accuracy(y_true, y_pred, n=1, average="invalid")  # ty: ignore[invalid-argument-type]

@pytest.mark.parametrize("average", ["micro", "macro"])
def test_validate_inputs_mismatched_datasets(average: Literal["micro", "macro"]):
    """Test with datasets that have different numbers of texts or tokens."""
    true_text1 = EvaluationTexts(
        text="test1",
        tokens=["token1"],
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=[["tag1"]]
    )
    true_text2 = EvaluationTexts(
        text="test2",
        tokens=["token2"],
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=[["tag2"]]
    )
    pred_text = EvaluationTexts(
        text="test",
        tokens=["token1"],
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=[["tag1"]]
    )
    
    y_true = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[true_text1, true_text2])
    y_pred = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[pred_text])
    
    with pytest.raises(ValueError, match="y_true and y_pred must have the same number of texts"):
        top_n_accuracy(y_true, y_pred, n=1, average=average)

@pytest.mark.parametrize("average", ["micro", "macro"])
def test_validate_inputs_different_tokens(average: Literal["micro", "macro"]):
    """Test with average not in ['micro', 'macro']."""
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
        tokens=["token2"],
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=[["tag1"]]
    )
    
    y_true = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[true_text])
    y_pred = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[pred_text])
    
    with pytest.raises(ValueError, match="y_true and y_pred must have the same tokens in the same order"):
        top_n_accuracy(y_true, y_pred, n=1, average=average)


@pytest.mark.parametrize("true_missing_semantic_tags", [True, False])
@pytest.mark.parametrize("average", ["micro", "macro"])
def test_validate_inputs_missing_semantic_tags(average: Literal["micro", "macro"],
                                               true_missing_semantic_tags: bool):
    """Test with datasets where semantic tags are None."""
    true_semantic_tags = None
    pred_semantic_tags = None
    if true_missing_semantic_tags:
        pred_semantic_tags = [["tag1"]]
    else:
        true_semantic_tags = [["tag1"]]
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
        tokens=["token1"],
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=pred_semantic_tags
    )
    
    y_true = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[true_text])
    y_pred = EvaluationDataset(name="test", text_level=TextLevel.sentence, texts=[pred_text])
    
    if true_missing_semantic_tags:
        with pytest.raises(ValueError, match="y_true text 0 does not have semantic_tags"):
            top_n_accuracy(y_true, y_pred, n=1, average=average)
    else:
        with pytest.raises(ValueError, match="y_pred text 0 does not have semantic_tags"):
            top_n_accuracy(y_true, y_pred, n=1, average=average)

@pytest.mark.parametrize("average", ["micro", "macro"])
@pytest.mark.parametrize("is_empty_string", [False, True])
def test_collect_and_filter_tags_empty_true_labels(average: Literal["micro", "macro"],
                                                   is_empty_string: bool):
    """Test with datasets where true tags are empty."""
    semantic_tags = [[]]
    if is_empty_string:
        semantic_tags = [[""]]
    true_text = EvaluationTexts(
        text="test",
        tokens=["token1"],
        lemmas=None,
        pos_tags=None,
        mwe_indexes=None,
        semantic_tags=semantic_tags
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
    
    with pytest.raises(ValueError, match="All of the true semantic tags are either empty strings or are empty lists"):
        top_n_accuracy(y_true, y_pred, n=1, average=average)
        


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
