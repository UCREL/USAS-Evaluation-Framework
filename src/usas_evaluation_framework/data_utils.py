"""
General data utilities.
"""
from typing import TypeVar

T = TypeVar("T")


def create_inner_list(a_list: list[T]) -> list[list[T]]:
    """
    Creates a list of lists from a list of objects. The inner list will only
    contain the one object that was in the original list at the same index.

    Args:
        a_list: The list to create the inner list from.

    Returns:
        list[list[T]]: A list of lists from the original list.

    Examples:
        >>> create_inner_list([1, 2, 3])
        [[1], [2], [3]]
    """
    return [[item] for item in a_list]


def is_labelled(token_tags: list[str],
                label_filter: set[str] | frozenset[str] = frozenset()) -> bool:
    """
    Return True if a token has at least one semantic tag that counts as a label.

    A tag counts as a label if it is not empty, not only whitespace, and not in
    ``label_filter``. The filter matches the whole tag string, so filtering
    ``F2`` does not remove ``F2/O2``.

    This is the single definition of a labelled token, used by both
    :attr:`.DatasetStats.num_labelled_tokens` and the coverage metrics in
    :mod:`usas_evaluation_framework.metrics.coverage`.

    Args:
        token_tags: The semantic tags of a single token.
        label_filter: The semantic tags that do not count as a label. Default
            is an empty set, no tags are filtered.

    Returns:
        bool: True if the token has a label, False otherwise.

    Examples:
        >>> is_labelled(["Q2.2", "Z99"])
        True
        >>> is_labelled(["Q2.2", "Z99"], {"Z99"})
        True
        >>> is_labelled(["Z99"], {"Z99"})
        False
        >>> is_labelled(["F2/O2"], {"F2"})
        True
        >>> is_labelled(["", " "])
        False
    """
    return any(tag.strip() and tag not in label_filter for tag in token_tags)
