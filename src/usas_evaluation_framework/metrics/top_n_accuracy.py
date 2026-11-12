from collections import defaultdict
from typing import Literal

from usas_evaluation_framework.dataset import EvaluationDataset
from usas_evaluation_framework.metrics.metric_utils import (
    collect_and_filter_tags,
    validate_datasets,
)


def top_n_accuracy(y_true: EvaluationDataset,
                   y_pred: EvaluationDataset,
                   n: int,
                   average: Literal["micro", "macro"]) -> float:
    r"""
    Calculate the top-n accuracy for the semantic tags in the dataset. If the
    `true` dataset contains more than one semantic tag per token, i.e. more than
    one true semantic tag for the token, then the accuracy score will be calculated
    for each `true` semantic tag. True semantic tags that are an empty string are
    ignored.

    .. math::

        \text{Top-n Accuracy} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(\text{correct answer} \in \text{top-}n(\hat{y}_{j,i}))

    Where:

    - :math:`|Q|` is the total number of true semantic tags for all tokens.
    - :math:`\hat{y}_{j,i}` represents the predicted scores for all semantic tags for the :math:`i`-th token in the :math:`j`-th text.
    - :math:`\text{top-}n(\hat{y}_{j,i})` denotes the set of top-:math:`n` predicted semantic tags for the :math:`i`-th token in the :math:`j`-th text.
    - :math:`\mathbb{I}(\cdot)` is the indicator function, which equals 1 if the correct answer is in the top-:math:`n` predictions and 0 otherwise.

    Args:
        y_true: The true labels for the dataset.
        y_pred: The predicted labels for the dataset.
        n: The number of semantic tags from the predicted labels to consider.
        average: The averaging method applied. "micro" everything is calculated
            globally as shown in the equation above, "macro" everything is
            calculated per label using the equation above and then
            averaged by the number of labels.

    Returns:
        float: The top-n accuracy of the dataset.

    Raises:
        ValueError: If any of the inputs are invalid.
        ValueError: If all of the true semantic tags are either empty strings or are empty lists.
    """
    validate_top_n_inputs(n, average)
    validate_datasets(y_true, y_pred)

    true_tags_list, pred_tags_list = collect_and_filter_tags(y_true, y_pred)
    
    match average:
        case "micro":
            return micro_accuracy(true_tags_list, pred_tags_list, n)
        case "macro":            
            return macro_accuracy(true_tags_list, pred_tags_list, n)


def validate_top_n_inputs(n: int,
                          average: Literal["micro", "macro"]) -> None:
    """
    Validate the top-n specific inputs to the top_n_accuracy function.

    Args:
        n: The number of semantic tags from the predicted labels to consider.
        average: The averaging method applied.

    Raises:
        ValueError: If `n` is not a positive integer.
        ValueError: If `average` is not either 'micro' or 'macro'.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if average not in ("micro", "macro"):
        raise ValueError("average must be either 'micro' or 'macro'")


def micro_accuracy(
    true_tags_list: list[list[str]],
    pred_tags_list: list[list[str]],
    n: int,
    ) -> float:
    """
    Compute micro-averaged top-n accuracy. We assume all empty strings within the
    true tag lists have been filtered out, see `metric_utils.collect_and_filter_tags`.

    Micro-averaging: calculates accuracy globally.

    Args:
        true_tags_list: A list of lists of true semantic tags for each token in each text.
            Each semantic tag in this list is considered a true tag.
        pred_tags_list: A list of lists of predicted semantic tags for each token in each text.
        n: The number of predicted semantic tags to consider.

    Returns:
        float: The micro-averaged top-n accuracy.
    """
    true_positive = 0
    total = 0
    for true_tags, pred_tags in zip(true_tags_list, pred_tags_list, strict=True):
        for true_tag in true_tags:
            total += 1
            if true_tag in pred_tags[:n]:
                true_positive += 1
    
    if total == 0:
        return 0.0
    else:
        return true_positive / total


def macro_accuracy(
    true_tags_list: list[list[str]],
    pred_tags_list: list[list[str]],
    n: int
    ) -> float:
    """
    Compute macro-averaged top-n accuracy. We assume all empty strings within the
    true tag lists have been filtered out, see `metric_utils.collect_and_filter_tags`.

    Macro-averaging: calculates accuracy for each label and then averages the accuracy
    scores over all labels.

    Args:
        true_tags_list: A list of lists of true semantic tags for each token in each text.
            Each semantic tag in this list is considered a true tag.
        pred_tags_list: A list of lists of predicted semantic tags for each token in each text.
        n: The number of predicted semantic tags to consider.

    Returns:
        float: The macro-averaged top-n accuracy.
    """
    correct_tag_count: dict[str, int] = defaultdict(lambda: 0)
    total_tag_count: dict[str, int] = defaultdict(lambda: 0)

    for true_tags, pred_tags in zip(true_tags_list, pred_tags_list, strict=True):
        top_n_pred = pred_tags[:n]
        for tag in true_tags:
            total_tag_count[tag] += 1
            if tag in top_n_pred:
                correct_tag_count[tag] += 1

    per_label_accuracy: list[float] = []
    for label, total in total_tag_count.items():
        accuracy = correct_tag_count.get(label, 0.0) / total
        per_label_accuracy.append(accuracy)

    return sum(per_label_accuracy) / len(per_label_accuracy)
