from typing import cast

from usas_evaluation_framework.dataset import EvaluationDataset


def validate_datasets(y_true: EvaluationDataset,
                      y_pred: EvaluationDataset) -> None:
    """
    Validate that the true and predicted datasets can be compared by a metric.

    The datasets must have the same number of texts, the same tokens in the
    same order, and every text in both datasets must have semantic tags.

    Args:
        y_true: The true labels for the dataset.
        y_pred: The predicted labels for the dataset.

    Raises:
        ValueError: If the datasets have a different number of texts.
        ValueError: If the datasets do not have the same tokens in the same order.
        ValueError: If any text in either dataset does not have semantic tags.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same number of texts")

    if not y_true.text_tokens_equal(y_pred):
        raise ValueError("y_true and y_pred must have the same tokens in the same order")

    for i, (true_text, pred_text) in enumerate(zip(y_true.texts, y_pred.texts, strict=True)):
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

    If the true semantic tags for a token are only empty/empty strings then we
    treat that token as unannotated and exclude it from scoring, the predicted
    tags for that token are excluded as well.

    The datasets are assumed to have been validated, see `validate_datasets`.
    This function relies on the true and predicted tags being aligned by index,
    which is guaranteed by the datasets having the same tokens in the same order
    and each text having one list of semantic tags per token.

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

    for true_text, pred_text in zip(y_true.texts, y_pred.texts, strict=True):

        true_text_semantic_tags: list[list[str]] = cast(list[list[str]], true_text.semantic_tags)
        pred_text_semantic_tags: list[list[str]] = cast(list[list[str]], pred_text.semantic_tags)

        # We are looping over the semantic tags for each token
        for true_tags, pred_tags in zip(true_text_semantic_tags, pred_text_semantic_tags, strict=True):
            true_tags_filtered: list[str] = [tag for tag in true_tags if tag]
            pred_tags_filtered: list[str] = [tag for tag in pred_tags if tag]

            # If no true tags for the token, skip
            if not true_tags_filtered:
                continue

            true_tags_list.append(true_tags_filtered)
            pred_tags_list.append(pred_tags_filtered)

    if not true_tags_list:
        raise ValueError("All of the true semantic tags are either empty strings or are empty lists.")

    return true_tags_list, pred_tags_list
