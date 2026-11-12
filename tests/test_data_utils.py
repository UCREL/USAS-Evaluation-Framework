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
