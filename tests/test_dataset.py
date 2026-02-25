"""Tests for the dataset module."""
import copy
from typing import Literal, TypedDict

import pytest

from usas_evaluation_framework.dataset import (
    EvaluationDataset,
    EvaluationTexts,
    TextLevel,
)


class EvaluationTextsData(TypedDict):
    text: str
    tokens: list[str]
    lemmas: list[str]
    pos_tags: list[str]
    semantic_tags: list[str]
    mwe_indexes: list[frozenset[int]]

def test_text_level_enum_values() -> None:
    """Test that TextLevel enum has the correct values."""
    assert TextLevel.sentence.value == "sentence"
    assert TextLevel.paragraph.value == "paragraph"
    assert TextLevel.document.value == "document"


def test_text_level_enum_members() -> None:
    """Test that TextLevel enum has the correct members."""
    assert TextLevel.sentence in TextLevel
    assert TextLevel.paragraph in TextLevel
    assert TextLevel.document in TextLevel


def test_text_level_enum_iteration() -> None:
    """Test that TextLevel enum can be iterated over."""
    members = list(TextLevel)
    assert len(members) == 3
    assert TextLevel.sentence in members
    assert TextLevel.paragraph in members
    assert TextLevel.document in members


def test_text_level_enum_from_string() -> None:
    """Test that TextLevel enum can be created from strings."""
    assert TextLevel("sentence") == TextLevel.sentence
    assert TextLevel("paragraph") == TextLevel.paragraph
    assert TextLevel("document") == TextLevel.document


def test_text_level_enum_invalid_value() -> None:
    """Test that TextLevel enum raises ValueError for invalid values."""
    try:
        TextLevel("invalid")
        assert False, "Expected ValueError for invalid enum value"
    except ValueError:
        pass

@pytest.fixture
def evaluation_texts_data() -> EvaluationTextsData:
    return {
        "text": "This is a test sentence.",
        "tokens": ["This", "is", "a", "test", "sentence", "."],
        "lemmas": ["This", "be", "a", "test", "sentence", "."],
        "pos_tags": ["DT", "VBZ", "DT", "NN", "NN", "."],
        "semantic_tags": ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"],
        "mwe_indexes": [frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5}), frozenset({6})]
    } 

def test_evaluation_texts_valid_initialization(evaluation_texts_data: EvaluationTextsData) -> None:
    texts = EvaluationTexts(
        text=evaluation_texts_data["text"],
        tokens=evaluation_texts_data["tokens"],
        lemmas=evaluation_texts_data["lemmas"],
        pos_tags=evaluation_texts_data["pos_tags"],
        semantic_tags=evaluation_texts_data["semantic_tags"],
        mwe_indexes=evaluation_texts_data["mwe_indexes"]
    )
    assert texts.text == evaluation_texts_data["text"]
    assert texts.tokens == evaluation_texts_data["tokens"]
    assert texts.lemmas == evaluation_texts_data["lemmas"]
    assert texts.pos_tags == evaluation_texts_data["pos_tags"]
    assert texts.semantic_tags == evaluation_texts_data["semantic_tags"]
    assert texts.mwe_indexes == evaluation_texts_data["mwe_indexes"]

def test_evaluation_texts__eq__(evaluation_texts_data: EvaluationTextsData) -> None:
    expected_evaluation_texts = EvaluationTexts(**evaluation_texts_data)
    assert expected_evaluation_texts == expected_evaluation_texts
    incorrect_key_values = [
        ("text", "Different text"),
        ("tokens", ["Different tokens"] * len(evaluation_texts_data["tokens"])),
        ("lemmas", ["Different lemmas"]* len(evaluation_texts_data["tokens"])),
        ("pos_tags", ["Different pos tags"]* len(evaluation_texts_data["tokens"])),
        ("semantic_tags", ["Different semantic tags"]* len(evaluation_texts_data["tokens"])),
        ("mwe_indexes", [frozenset({})]* len(evaluation_texts_data["tokens"]))
    ]
    for incorrect_key, incorrect_value in incorrect_key_values:
        temp_evaluation_texts_data = copy.deepcopy(evaluation_texts_data)
        temp_evaluation_texts_data[incorrect_key] = incorrect_value
        assert expected_evaluation_texts != EvaluationTexts(**temp_evaluation_texts_data)

    alt_evaluation_texts_data = dict(copy.deepcopy(evaluation_texts_data))
    alt_evaluation_texts_data["lemmas"] = None
    alt_evaluation_texts_data["pos_tags"] = None
    alt_evaluation_texts = EvaluationTexts(**alt_evaluation_texts_data)
    assert alt_evaluation_texts == alt_evaluation_texts

    assert alt_evaluation_texts != expected_evaluation_texts

@pytest.mark.parametrize("list_attribute_testing", ["lemmas", "pos_tags", "semantic_tags", "mwe_indexes"])
def test_evaluation_texts_mismatched_lengths(list_attribute_testing: Literal["lemmas", "pos_tags", "semantic_tags", "mwe_indexes"],
                                             evaluation_texts_data: EvaluationTextsData) -> None:
    lemmas = evaluation_texts_data["lemmas"]
    pos_tags = evaluation_texts_data["pos_tags"]
    semantic_tags = evaluation_texts_data["semantic_tags"]
    mwe_indexes = evaluation_texts_data["mwe_indexes"]
    match list_attribute_testing:
        case "lemmas":
            lemmas.pop()
        case "pos_tags":
            pos_tags.pop()
        case "semantic_tags":
            semantic_tags.pop()
        case "mwe_indexes":
            mwe_indexes.pop()
    with pytest.raises(ValueError):
        EvaluationTexts(
            text=evaluation_texts_data["text"],
            tokens=evaluation_texts_data["tokens"],
            lemmas=lemmas,
            pos_tags=pos_tags,
            semantic_tags=semantic_tags,
            mwe_indexes=mwe_indexes
        )

def test_evaluation_texts_none_values(evaluation_texts_data: EvaluationTextsData) -> None:
    texts = EvaluationTexts(
        text=evaluation_texts_data['text'],
        tokens=evaluation_texts_data['tokens'],
        lemmas=None,
        pos_tags=None,
        semantic_tags=None,
        mwe_indexes=None
    )
    assert texts.text == evaluation_texts_data['text']
    assert texts.tokens == evaluation_texts_data['tokens']
    assert texts.lemmas is None
    assert texts.pos_tags is None
    assert texts.semantic_tags is None
    assert texts.mwe_indexes is None


def test_evaluation_dataset_valid_initialization(evaluation_texts_data: EvaluationTextsData) -> None:
    texts = [
        EvaluationTexts(
            **evaluation_texts_data
        )
    ]
    dataset = EvaluationDataset(
        name="Test Dataset",
        text_level=TextLevel.sentence,
        labels_removed={"Z1", "Z2"},
        texts=texts
    )
    assert dataset.name == "Test Dataset"
    assert dataset.text_level == TextLevel.sentence
    assert dataset.labels_removed == {"Z1", "Z2"}
    assert dataset.texts == texts

def test_evaluation_dataset_empty_texts() -> None:
    dataset = EvaluationDataset(
        name="Test Dataset",
        text_level=TextLevel.sentence,
        labels_removed=None,
        texts=[]
    )
    assert dataset.name == "Test Dataset"
    assert dataset.text_level == TextLevel.sentence
    assert dataset.labels_removed is None
    assert dataset.texts == []
