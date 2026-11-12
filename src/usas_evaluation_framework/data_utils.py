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
