from enum import Enum

from pydantic import BaseModel, model_validator


class TextLevel(str, Enum):
    """
    An enumeration of text levels, the value is a string representation of the
    member.

    Attributes:
        sentence: A sentence
        paragraph: A paragraph
        document: A document
    """
    sentence = "sentence"
    paragraph = "paragraph"
    document = "document"


class EvaluationTexts(BaseModel):
    """
    A representation of a text, with many optional fields.

    All list fields must be the same length if they are not None.

    Attributes:
        text (str): The text
        tokens (list[str]): The tokens of the text.
        lemmas (list[str] | None): The lemmas of the text. Default is `None`.
        pos_tags (list[str] | None): The POS tags of the text. Default is `None`.
        semantic_tags (list[list[str]] | None): The semantic tags of the text. Default is `None`.
            The inner list contains all the possible semantic tags for the token at the same index.
            The semantic tags are in order of likelihood, e.g. the first tag is the most likely tag.
        mwe_indexes (list[frozenset[int]] | None): The Multi Word Expression (MWE) indexes of the text.
            If the set is empty then the token is not part of a MWE, otherwise
            the set contains the MWE index and all tokens with the same index
            make up the whole MWE. Default is `None`.
    """
    text: str
    tokens: list[str]
    lemmas: list[str] | None
    pos_tags: list[str] | None
    semantic_tags: list[list[str]] | None
    mwe_indexes: list[frozenset[int]] | None

    @model_validator(mode='after')
    def check_lists_match(self) -> "EvaluationTexts":
        """
        Checks that the length of the tokens, lemmas, POS tags, semantic tags, and MWE indexes
        are all the same if they are not None. If they are not the same, raises a ValueError.

        Returns:
            The EvaluationTexts object
        Raises:
            ValueError: If the length of the tokens, lemmas, POS tags, semantic tags, and MWE indexes are not the same
        """
        number_tokens = len(self.tokens)
        if self.lemmas is not None and number_tokens != len(self.lemmas):
            raise ValueError(f"The number of tokens: {number_tokens} and "
                             f"lemmas must be the same: {len(self.lemmas)}")
        if self.pos_tags is not None and number_tokens != len(self.pos_tags):
            raise ValueError(f"The number of tokens: {number_tokens} "
                             f"and POS tags must be the same: {len(self.pos_tags)}")
        if self.semantic_tags is not None and number_tokens != len(self.semantic_tags):
            raise ValueError(f"The number of tokens: {number_tokens} and "
                             f"Semantic tags must be the same: {len(self.semantic_tags)}")
        if self.mwe_indexes is not None and number_tokens != len(self.mwe_indexes):
            raise ValueError(f"The number of tokens: {number_tokens} and "
                             f"MWE indexes must be the same: {len(self.mwe_indexes)}")
        return self


class EvaluationDataset(BaseModel):
    """
    A representation of a dataset, it can be used to hold either gold/true
    labels or predicted labels. The dataset is designed for evaluation and analysis.


    Attributes:
        name (str): The name of the dataset
        text_level (TextLevel): The text level of the `texts`, e.g. sentence, paragraph, or document.
        labels_removed (set[str] | None): The labels that were removed from the dataset. For example
            a specific semantic tag or semantic tags. Default is `None` indicating no labels were removed.
        texts (list[EvaluationTexts]): The texts of the dataset, this contains
            both the texts, tokens, and potentially lemmas, POS tags, semantic tags, and MWE indexes.
    """
    name: str
    text_level: TextLevel
    labels_removed: set[str] | None = None
    texts: list[EvaluationTexts]

    def __len__(self: "EvaluationDataset") -> int:
        """
        Returns the number of texts in the dataset.

        Returns:
            The number of texts
        """
        return len(self.texts)

    def text_tokens_equal(self: "EvaluationDataset", other: "EvaluationDataset") -> bool:
        """
        Returns True if the texts in the datasets are equal, False otherwise.
        This comparison is based on the tokens of each text at the same index
        being equal.

        Args:
            other: The other dataset whose texts are to be compared too.

        Returns:
            True if the texts are equal, False otherwise.
        """
        # Check if the number of texts in both datasets is the same
        if len(self) != len(other):
            return False

        # Compare each text's tokens at the same index
        for self_text, other_text in zip(self.texts, other.texts):
            if self_text.tokens != other_text.tokens:
                return False

        return True
    

