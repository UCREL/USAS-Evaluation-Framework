from collections import defaultdict
from typing import Literal, cast

from usas_evaluation_framework.dataset import EvaluationDataset


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

    $$
    \text{Top-n Accuracy} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(\text{correct answer} \in \text{top-}n(\hat{y}_{j,i}))
    $$

    Where:
    - \( |Q| \) is the total number of true semantic tags for all tokens.
    - \( \hat{y}_{j,i} \) represents the predicted scores for all semantic tags for the \(i\)-th token in the \(j\)-th text.
    - \( \text{top-}n(\hat{y}_i) \) denotes the set of top-\(n\) predicted semantic tags for the \(i\)-th token in the \(j\)-th text.
    - \( \mathbb{I}(\cdot) \) is the indicator function, which equals 1 if the correct answer is in the top-nnn predictions and 0 otherwise.

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
    validate_inputs(y_true, y_pred, n, average)
        
    true_tags_list, pred_tags_list = collect_and_filter_tags(y_true, y_pred)
    
    match average:
        case "micro":
            return micro_accuracy(true_tags_list, pred_tags_list, n)
        case "macro":            
            return macro_accuracy(true_tags_list, pred_tags_list, n)


def validate_inputs(y_true: EvaluationDataset,
                    y_pred: EvaluationDataset,
                    n: int,
                    average: Literal["micro", "macro"]) -> None:
    """
    Validate the inputs to the top_n_accuracy function.

    Args:
        y_true: The true labels for the dataset.
        y_pred: The predicted labels for the dataset.
        n: The number of semantic tags from the predicted labels to consider.
        average: The averaging method applied.

    Raises:
        ValueError: If any of the inputs are invalid.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if average not in ("micro", "macro"):
        raise ValueError("average must be either 'micro' or 'macro'")
    
    # Check that datasets have the same number of texts and tokens
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same number of texts")
    
    if not y_true.text_tokens_equal(y_pred):
        raise ValueError("y_true and y_pred must have the same tokens in the same order")
    
    # Check that both datasets have semantic_tags
    for i, (true_text, pred_text) in enumerate(zip(y_true.texts, y_pred.texts)):
        if true_text.semantic_tags is None:
            raise ValueError(f"y_true text {i} does not have semantic_tags")
        if pred_text.semantic_tags is None:
            raise ValueError(f"y_pred text {i} does not have semantic_tags")

def collect_and_filter_tags(
    y_true: EvaluationDataset,
    y_pred: EvaluationDataset,
    ) -> tuple[list[list[str]], list[list[str]]]:
    """
    Collect all true and predicted tags for all tokens for all texts and filter out empty strings.
    If the true semantic tags for a token are only empty/empty strings then we filter out that token.

    Args:
        y_true: The true labels for the dataset.
        y_pred: The predicted labels for the dataset.

    Returns:
        tuple[list[list[str]], list[list[str]]]: A tuple containing two lists of lists of semantic tags.
            The first list contains the true semantic tags, and the second list contains the predicted semantic tags.
            Each inner list contains the semantic tags for a single token.
            Empty strings have been filtered out.

    Raises:
        ValueError: If all of the true semantic tags are either empty strings or are empty lists.
    """
    true_tags_list: list[list[str]] = []
    pred_tags_list: list[list[str]] = []
    all_true_labels: set[str] = set()

    for true_text, pred_text in zip(y_true.texts, y_pred.texts):
        
        true_text_semantic_tags: list[list[str]] = cast(list[list[str]], true_text.semantic_tags)
        pred_text_semantic_tags: list[list[str]] = cast(list[list[str]], pred_text.semantic_tags)
        
        # We are looping over the semantic tags for each token
        for true_tags, pred_tags in zip(true_text_semantic_tags, pred_text_semantic_tags):
            # This is just for linting and readability, no intialisation.
            true_tags: list[str] = cast(list[str], true_tags)
            pred_tags: list[str] = cast(list[str], pred_tags)

            true_tags_filtered: list[str] = []
            for tag in true_tags:
                if tag:
                    true_tags_filtered.append(tag)
                    all_true_labels.add(tag)

            pred_tags_filtered: list[str] = [tag for tag in pred_tags if tag]
            
            # If no true tags for the token, skip
            if not true_tags_filtered:
                continue
            
            true_tags_list.append(true_tags_filtered)
            pred_tags_list.append(pred_tags_filtered)
    
    if not all_true_labels:
        raise ValueError("All of the true semantic tags are either empty strings or are empty lists.")

    return true_tags_list, pred_tags_list


def micro_accuracy(
    true_tags_list: list[list[str]],
    pred_tags_list: list[list[str]],
    n: int,
    ) -> float:
    """
    Compute micro-averaged top-n accuracy. We assume all empty strings within the
    true tag lists have been filtered out, see `collect_and_filter_tags`.

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
    for true_tags, pred_tags in zip(true_tags_list, pred_tags_list):
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
    true tag lists have been filtered out, see `collect_and_filter_tags`.

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

    for true_tags, pred_tags in zip(true_tags_list, pred_tags_list):
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
