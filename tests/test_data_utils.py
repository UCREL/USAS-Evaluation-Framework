import pytest
from usas_validator.usas_tag import USASTag

from usas_evaluation_framework import data_utils


def test_create_inner_list() -> None:
    """Test the create_inner_list function with various input types."""

    # Test with integers
    result = data_utils.create_inner_list([1, 2, 3])
    expected = [[1], [2], [3]]
    assert result == expected

    # Test with strings
    result = data_utils.create_inner_list(["a", "b", "c"])
    expected = [["a"], ["b"], ["c"]]
    assert result == expected

    # Test with mixed types
    result = data_utils.create_inner_list([1, "a", 3.14])
    expected = [[1], ["a"], [3.14]]
    assert result == expected

    # Test with empty list
    result = data_utils.create_inner_list([])
    expected = []
    assert result == expected

    # Test with single element
    result = data_utils.create_inner_list([42])
    expected = [[42]]
    assert result == expected

    # Test with complex objects (USASTag instances)
    tags = [
        USASTag(tag="A1.1.1"),
        USASTag(tag="B2"),
        USASTag(tag="C3.1")
    ]
    result = data_utils.create_inner_list(tags)
    expected = [[tags[0]], [tags[1]], [tags[2]]]
    assert result == expected


@pytest.mark.parametrize(
    ("token_tags", "label_filter", "expected"),
    [
        (["A1"], set(), True),
        (["Z99"], {"Z99"}, False),
        (["Z99", "A1"], {"Z99"}, True),
        (["Z99"], frozenset({"Z99"}), False),
        # Whole tag matching, `F2` does not remove `F2/O2`.
        (["F2/O2"], {"F2"}, True),
        (["F2/O2"], {"F2/O2"}, False),
        ([], set(), False),
        (["", " "], set(), False),
        # A filtered tag alongside an empty one leaves no label.
        (["", "Z99"], {"Z99"}, False),
    ],
)
def test_is_labelled(token_tags: list[str], label_filter: set[str] | frozenset[str], expected: bool) -> None:
    """Test which tokens count as labelled."""
    assert data_utils.is_labelled(token_tags, label_filter) is expected


@pytest.mark.parametrize(
    ("token_tags", "expected"),
    [(["A1"], True), (["Z99"], True), ([], False), ([""], False), ([" ", "\t"], False)],
)
def test_is_labelled_default_filter(token_tags: list[str], expected: bool) -> None:
    """Test that without a filter only empty and whitespace tags are not labels."""
    assert data_utils.is_labelled(token_tags) is expected
