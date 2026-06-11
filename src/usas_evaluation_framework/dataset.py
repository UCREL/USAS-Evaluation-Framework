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


class DatasetStats(BaseModel):
    """
    Statistics about an EvaluationDataset.

    Attributes:
        num_texts (int): Number of texts in the dataset.
        num_tokens (int): Total number of tokens across all texts.
        num_semantic_tags (int | None): Total number of individual semantic tag strings
            assigned across all tokens. None if no texts have semantic tags.
        num_labelled_tokens (int | None): Number of tokens that have at least one semantic
            tag assigned. None if no texts have semantic tags.
        num_compound_semantic_tags (int | None): Number of semantic tag strings containing
            a '/' (e.g. 'B2/A1.1.1'). None if no texts have semantic tags.
        unique_semantic_tags (frozenset[str] | None): The set of distinct semantic tag strings
            that appear in the dataset. None if no texts have semantic tags.
        num_mwes (int | None): Number of distinct Multi Word Expressions across all texts.
            None if no texts have MWE indexes.
    """
    num_texts: int
    num_tokens: int
    num_semantic_tags: int | None
    num_labelled_tokens: int | None
    num_compound_semantic_tags: int | None
    unique_semantic_tags: frozenset[str] | None
    num_mwes: int | None


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
    language: str | None = None
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

    @classmethod
    def merge(
        cls,
        name: str,
        text_level: "TextLevel",
        language: "str | None",
        *datasets: "EvaluationDataset",
    ) -> "EvaluationDataset":
        """
        Merges one or more EvaluationDatasets into a new dataset.

        The caller supplies the name, text_level, and language for the result, so
        datasets with differing values for those fields can be combined freely.
        All source datasets must have identical labels_removed values.

        Args:
            name: The name for the merged dataset.
            text_level: The TextLevel for the merged dataset.
            language: The language for the merged dataset (can be None).
            *datasets: One or more EvaluationDataset instances to merge.

        Returns:
            A new EvaluationDataset containing all texts from the source datasets
            in the order they were supplied.

        Raises:
            ValueError: If no datasets are provided.
            ValueError: If datasets have differing labels_removed values.
        """
        if not datasets:
            raise ValueError("At least one dataset must be provided to merge.")

        first_labels_removed = datasets[0].labels_removed
        for dataset in datasets[1:]:
            if dataset.labels_removed != first_labels_removed:
                raise ValueError(
                    f"All datasets must have the same labels_removed value, "
                    f"but got {first_labels_removed!r} and {dataset.labels_removed!r}."
                )

        merged_texts = [text for dataset in datasets for text in dataset.texts]

        return cls(
            name=name,
            text_level=text_level,
            language=language,
            labels_removed=first_labels_removed,
            texts=merged_texts,
        )

    def stats(self: "EvaluationDataset") -> DatasetStats:
        """
        Returns statistics about the dataset.

        Returns:
            A DatasetStats object containing counts of texts, tokens, semantic tags,
            compound semantic tags, unique semantic tags, and MWEs.
        """
        num_tokens = sum(len(t.tokens) for t in self.texts)

        all_tags = [
            tag
            for t in self.texts
            if t.semantic_tags is not None
            for tag_list in t.semantic_tags
            for tag in tag_list
        ]
        if all_tags:
            num_semantic_tags: int | None = len(all_tags)
            num_labelled_tokens: int | None = sum(
                1
                for t in self.texts
                if t.semantic_tags is not None
                for tag_list in t.semantic_tags
                if tag_list
            )
            num_compound_semantic_tags: int | None = sum(1 for tag in all_tags if "/" in tag)
            unique_semantic_tags: frozenset[str] | None = frozenset(all_tags)
        else:
            num_semantic_tags = None
            num_labelled_tokens = None
            num_compound_semantic_tags = None
            unique_semantic_tags = None

        if any(t.mwe_indexes is not None for t in self.texts):
            num_mwes: int | None = sum(
                len({idx for fs in t.mwe_indexes for idx in fs})
                for t in self.texts
                if t.mwe_indexes is not None
            )
        else:
            num_mwes = None

        return DatasetStats(
            num_texts=len(self),
            num_tokens=num_tokens,
            num_semantic_tags=num_semantic_tags,
            num_labelled_tokens=num_labelled_tokens,
            num_compound_semantic_tags=num_compound_semantic_tags,
            unique_semantic_tags=unique_semantic_tags,
            num_mwes=num_mwes,
        )

