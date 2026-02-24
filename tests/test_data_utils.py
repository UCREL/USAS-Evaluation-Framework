from pathlib import Path

import pytest

from tests.utils_test import get_test_data_directory  # noqa: F401
from usas_evaluation_framework import data_utils
from usas_evaluation_framework.data_utils import USASTag, USASTagGroup


@pytest.fixture
def get_test_data_utils_directory(get_test_data_directory: Path) -> Path:  # noqa: F811
    return get_test_data_directory / "data_utils"



def test_get_all_mwe_token_indexes() -> None:
    """Test the get_all_mwe_token_indexes function with various MWE index ranges."""

    # Verify return type is frozenset
    result = data_utils.get_all_mwe_token_indexes([(1, 4)])
    assert isinstance(result, frozenset)
    
    # Test single token range
    result = data_utils.get_all_mwe_token_indexes([(5, 6)])
    expected = frozenset({5})
    assert result == expected

    # Test single MWE range
    result = data_utils.get_all_mwe_token_indexes([(0, 3)])
    expected = frozenset({0, 1, 2})
    assert result == expected

    # Test multiple non-overlapping MWE ranges
    result = data_utils.get_all_mwe_token_indexes([(0, 2), (5, 7)])
    expected = frozenset({0, 1, 5, 6})
    assert result == expected

    # Test multiple overlapping MWE ranges
    result = data_utils.get_all_mwe_token_indexes([(0, 4), (2, 6)])
    expected = frozenset({0, 1, 2, 3, 4, 5})
    assert result == expected

    # Test multiple ranges with some single tokens
    result = data_utils.get_all_mwe_token_indexes([(0, 1), (3, 4), (6, 8)])
    expected = frozenset({0, 3, 6, 7})
    assert result == expected

    # Test with completely overlapping ranges
    result = data_utils.get_all_mwe_token_indexes([(2, 5), (2, 5)])
    expected = frozenset({2, 3, 4})
    assert result == expected

    # Test empty input
    result = data_utils.get_all_mwe_token_indexes([])
    expected = frozenset()
    assert result == expected


def test_parse_usas_token_group() -> None:
    """Test the parse_usas_token_group function with various USAS tag formats."""
    
    # Test simple single tag
    result = data_utils.parse_usas_token_group("A1.1.1")
    expected = [USASTagGroup(tags=[USASTag(tag="A1.1.1")])]
    assert result == expected
    
    # Test tag with positive markers
    result = data_utils.parse_usas_token_group("X5.2+")
    expected = [USASTagGroup(tags=[USASTag(tag="X5.2", number_positive_markers=1)])]
    assert result == expected
    
    result = data_utils.parse_usas_token_group("X5.2++")
    expected = [USASTagGroup(tags=[USASTag(tag="X5.2", number_positive_markers=2)])]
    assert result == expected
    
    # Test tag with negative markers
    result = data_utils.parse_usas_token_group("E3-")
    expected = [USASTagGroup(tags=[USASTag(tag="E3", number_negative_markers=1)])]
    assert result == expected
    
    result = data_utils.parse_usas_token_group("O4.2--")
    expected = [USASTagGroup(tags=[USASTag(tag="O4.2", number_negative_markers=2)])]
    assert result == expected
    
    # Test tag with gender markers
    result = data_utils.parse_usas_token_group("S2mf")
    expected = [USASTagGroup(tags=[USASTag(tag="S2", male=True, female=True)])]
    assert result == expected
    
    result = data_utils.parse_usas_token_group("S2m")
    expected = [USASTagGroup(tags=[USASTag(tag="S2", male=True)])]
    assert result == expected
    
    result = data_utils.parse_usas_token_group("S2f")
    expected = [USASTagGroup(tags=[USASTag(tag="S2", female=True)])]
    assert result == expected
    
    # Test tag with rarity markers
    result = data_utils.parse_usas_token_group("A1%")
    expected = [USASTagGroup(tags=[USASTag(tag="A1", rarity_marker_1=True)])]
    assert result == expected
    
    result = data_utils.parse_usas_token_group("B2@")
    expected = [USASTagGroup(tags=[USASTag(tag="B2", rarity_marker_2=True)])]
    assert result == expected
    
    # Test tag with antecedent marker
    result = data_utils.parse_usas_token_group("A1c")
    expected = [USASTagGroup(tags=[USASTag(tag="A1", antecedents=True)])]
    assert result == expected
    
    # Test tag with neuter marker
    result = data_utils.parse_usas_token_group("A1n")
    expected = [USASTagGroup(tags=[USASTag(tag="A1", neuter=True)])]
    assert result == expected
    
    # Test combined tags with slash
    result = data_utils.parse_usas_token_group("Z2/S2mf")
    expected = [USASTagGroup(tags=[
        USASTag(tag="Z2"),
        USASTag(tag="S2", male=True, female=True)
    ])]
    assert result == expected
    
    # Test multiple tag groups
    result = data_utils.parse_usas_token_group("L1 E3- O4.2-")
    expected = [
        USASTagGroup(tags=[USASTag(tag="L1")]),
        USASTagGroup(tags=[USASTag(tag="E3", number_negative_markers=1)]),
        USASTagGroup(tags=[USASTag(tag="O4.2", number_negative_markers=1)])
    ]
    assert result == expected
    
    # Test complex example from docstring
    result = data_utils.parse_usas_token_group("L1 E3- O4.2- X5.2+ A6.2- A1.7- A7- W3 L2 F1 S1.2.4- Z2 Z2/S2mf Z3 O4.3 G1.2 G1.2/S2mf")
    expected = [
        USASTagGroup(tags=[USASTag(tag="L1")]),
        USASTagGroup(tags=[USASTag(tag="E3", number_negative_markers=1)]),
        USASTagGroup(tags=[USASTag(tag="O4.2", number_negative_markers=1)]),
        USASTagGroup(tags=[USASTag(tag="X5.2", number_positive_markers=1)]),
        USASTagGroup(tags=[USASTag(tag="A6.2", number_negative_markers=1)]),
        USASTagGroup(tags=[USASTag(tag="A1.7", number_negative_markers=1)]),
        USASTagGroup(tags=[USASTag(tag="A7", number_negative_markers=1)]),
        USASTagGroup(tags=[USASTag(tag="W3")]),
        USASTagGroup(tags=[USASTag(tag="L2")]),
        USASTagGroup(tags=[USASTag(tag="F1")]),
        USASTagGroup(tags=[USASTag(tag="S1.2.4", number_negative_markers=1)]),
        USASTagGroup(tags=[USASTag(tag="Z2")]),
        USASTagGroup(tags=[
            USASTag(tag="Z2"),
            USASTag(tag="S2", male=True, female=True)
        ]),
        USASTagGroup(tags=[USASTag(tag="Z3")]),
        USASTagGroup(tags=[USASTag(tag="O4.3")]),
        USASTagGroup(tags=[USASTag(tag="G1.2")]),
        USASTagGroup(tags=[
            USASTag(tag="G1.2"),
            USASTag(tag="S2", male=True, female=True)
        ])
    ]
    assert result == expected

    # Test that it can parse the whole USAS tagset
    for usas_tag in data_utils.load_usas_mapper(None, None).keys():
        result = data_utils.parse_usas_token_group(usas_tag)
        expected = [USASTagGroup(tags=[USASTag(tag=usas_tag)])]
        assert result == expected
    
    # Test edge cases
    result = data_utils.parse_usas_token_group("")
    expected = []
    assert result == expected
    
    result = data_utils.parse_usas_token_group(" ")
    assert result == expected

    result = data_utils.parse_usas_token_group("   ")
    assert result == expected

    result = data_utils.parse_usas_token_group("PUNCT")
    expected = [USASTagGroup(tags=[USASTag(tag="PUNCT")])]
    assert result == expected
    
    # Test invalid tag format
    with pytest.raises(ValueError):
        data_utils.parse_usas_token_group("invalid_tag")
    
    with pytest.raises(ValueError):
        data_utils.parse_usas_token_group("123")
    
    with pytest.raises(ValueError):
        data_utils.parse_usas_token_group("L1 invalid_tag")


@pytest.mark.parametrize("usas_tag_description_file_str",
                        [None, "test_usas_mapper.yaml"])
@pytest.mark.parametrize("tags_to_filter_out", [None, set(["Z99"])])
def test_load_usas_mapper(usas_tag_description_file_str: str | None,
                          tags_to_filter_out: set[str] | None,
                          get_test_data_utils_directory: Path) -> None:
    usas_tag_description_file: Path | None = None
    if usas_tag_description_file_str is not None:
        usas_tag_description_file = get_test_data_utils_directory / usas_tag_description_file_str
    usas_mapper = data_utils.load_usas_mapper(usas_tag_description_file,
                                                              tags_to_filter_out)
    assert isinstance(usas_mapper, dict)
    assert len(usas_mapper) > 0

    if tags_to_filter_out is None and usas_tag_description_file is None:
        assert len(usas_mapper) == 222
        assert "Z99" in usas_mapper
    elif tags_to_filter_out is not None and usas_tag_description_file is None:
        assert len(usas_mapper) == 221
        assert "Z99" not in usas_mapper
    elif usas_tag_description_file is not None:
        assert len(usas_mapper) == 1
    
    assert "A1.1.1" in usas_mapper
    if usas_tag_description_file is None:
        expected_title_description = (
            "title: General actions, making etc. description: "
            "General/abstract terms relating to an activity/action "
            "(e.g. act, adventure, approach, arise); a characteristic/feature "
            "(e.g. absorb, attacking, automatically); "
            "aconstruction/craft and/or the action of constructing/crafting "
            "(e.g. arrange, assemble, bolts, boring, break)"
        )
        assert expected_title_description == usas_mapper["A1.1.1"]
    else:
        assert "title: General Test description: Test Case" == usas_mapper["A1.1.1"]

    assert "A.1" not in usas_mapper

def test_load_usas_mapper_with_nonexistent_file() -> None:
    with pytest.raises(FileNotFoundError):
        data_utils.load_usas_mapper(Path(__file__).parent / "nonexistent.yaml", None)

def test_load_usas_mapper_with_directory_path() -> None:
    with pytest.raises(ValueError):
        data_utils.load_usas_mapper(Path(__file__).parent, None)

def test_load_usas_mapper_with_no_title(get_test_data_utils_directory: Path) -> None:
    with pytest.raises(KeyError):
        no_title_usas_tag_description_file = get_test_data_utils_directory / "test_usas_mapper_no_title.yaml"
        data_utils.load_usas_mapper(no_title_usas_tag_description_file, None)

def test_load_usas_mapper_with_no_description(get_test_data_utils_directory: Path) -> None:
    with pytest.raises(KeyError):
        no_description_usas_tag_description_file = get_test_data_utils_directory / "test_usas_mapper_no_description.yaml"
        data_utils.load_usas_mapper(no_description_usas_tag_description_file, None)

def test_load_usas_mapper_duplicate_key(get_test_data_utils_directory: Path) -> None:
    with pytest.raises(KeyError):
        duplicate_key_usas_tag_description_file = get_test_data_utils_directory / "test_usas_mapper_duplicate_key.yaml"
        data_utils.load_usas_mapper(duplicate_key_usas_tag_description_file, None)

