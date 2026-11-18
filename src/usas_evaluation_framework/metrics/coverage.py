from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Literal

from usas_evaluation_framework.data_utils import is_labelled
from usas_evaluation_framework.dataset import EvaluationDataset, EvaluationTexts


def coverage(y_pred: EvaluationDataset | Iterable[EvaluationTexts],
             label_filter: set[str] | None = None,
             exclude_labels: set[str] | None = None) -> float:
    r"""
    Calculate the coverage of a semantic tagger, the proportion of tokens in the
    dataset that have been assigned at least one semantic tag.

    This is the "real text token" coverage of `A domain-independent semantic
    tagger for the study of meaning associations in English text 
    <https://eprints.whiterose.ac.uk/id/eprint/82299/1/ADomainIndependentSemanticTagger.pdf>`__
    (Section 5), which answers "how many of the word tokens in a text are
    expected to be covered by the system?". For the coverage of distinct words
    see :func:`type_coverage`.

    A token counts as labelled if it has at least one tag that is not empty
    (``""``), not only whitespace, and not in ``label_filter``, see
    :func:`.is_labelled`. Without a filter this is the same as
    :attr:`.DatasetStats.num_labelled_tokens`. The filter matches the
    whole tag string, so filtering ``F2`` does not remove ``F2/O2``. Filtering
    lets tags that do not represent a real semantic tag, such as ``Z99``
    (unmatched), count as not covered.

    Tokens that are not meaningful units to measure, such as punctuation
    (``Z9``), can instead be left out of both the numerator and denominator
    with ``exclude_labels``, see :func:`is_excluded`. A filtered token counts
    against the tagger, an excluded token is not counted at all.

    .. math::

        \text{Coverage} = \frac{\text{number of labelled tokens}}{\text{number of tokens}}

    Where both counts are over the tokens that are not excluded.

    .. warning::

        Tokens are excluded based on the tagger's own predicted tags, so two
        taggers are only comparable with ``exclude_labels`` if they assign
        those tags to the same tokens. A word wrongly tagged ``Z9`` is
        excluded rather than counted against the tagger.

    The texts are read once, one at a time, and nothing is copied, so the extra
    memory used does not grow with the size of the dataset. To score a corpus
    too large to hold in memory, pass an iterable, such as a generator, that
    yields its texts one at a time.

    Args:
        y_pred: The texts containing the semantic tags predicted by the tagger,
            either as an :class:`.EvaluationDataset` or any iterable of
            :class:`.EvaluationTexts`.
        label_filter: Semantic tags that do not count as a label, e.g.
            ``{"Z99"}``. Default is `None`, no tags are filtered.
        exclude_labels: Semantic tags whose tokens are left out of the
            calculation, e.g. ``{"Z9"}``. Default is `None`, no tokens are
            excluded.

    Returns:
        float: The coverage, between 0.0 and 1.0.

    Raises:
        ValueError: If ``label_filter`` and ``exclude_labels`` share a tag.
        ValueError: If any text does not have semantic tags.
        ValueError: If there are no tokens, or every token is excluded.

    Examples:
        >>> from usas_evaluation_framework.dataset import (
        ...     EvaluationDataset, EvaluationTexts, TextLevel
        ... )
        >>> text = EvaluationTexts(
        ...     text="The cat sat .",
        ...     tokens=["The", "cat", "sat", "."],
        ...     lemmas=None,
        ...     pos_tags=None,
        ...     semantic_tags=[["Z5"], ["L2", "Z99"], ["Z99"], ["Z9"]],
        ...     mwe_indexes=None,
        ... )
        >>> y_pred = EvaluationDataset(
        ...     name="example", text_level=TextLevel.sentence, texts=[text]
        ... )
        >>> coverage(y_pred)
        1.0
        >>> coverage(y_pred, label_filter={"Z99"})
        0.75
        >>> coverage(y_pred, label_filter={"Z99", "Z9"})
        0.5

        Excluding punctuation leaves three tokens, two of them labelled:

        >>> round(coverage(y_pred, label_filter={"Z99"}, exclude_labels={"Z9"}), 4)
        0.6667

        Texts can also be streamed one at a time, e.g. from a generator:

        >>> coverage(t for t in [text, text])
        1.0
    """
    num_tokens = 0
    num_labelled_tokens = 0
    for _, labelled in iter_token_labels(y_pred, label_filter, exclude_labels):
        num_tokens += 1
        num_labelled_tokens += labelled

    if num_tokens == 0:
        raise ValueError("y_pred does not contain any tokens that are not excluded")

    return num_labelled_tokens / num_tokens


def type_coverage(y_pred: EvaluationDataset | Iterable[EvaluationTexts],
                  label_filter: set[str] | None = None,
                  exclude_labels: set[str] | None = None,
                  lowercase: bool = False,
                  average: Literal["binary", "macro"] = "binary") -> float:
    r"""
    Calculate the word type coverage of a semantic tagger, how well the unique
    words (token strings) in the dataset are covered by the tagger.

    Each word type is given a score between 0 and 1, and the type coverage is
    the mean of these scores over all word types:

    .. math::

        \text{Type Coverage} = \frac{1}{|W|} \sum_{w \in W} \text{score}(w)

    Where :math:`W` is the set of word types. How :math:`\text{score}(w)` is
    calculated depends on ``average``:

    - ``"binary"``: 1 if **at least one** occurrence of the word type is
      labelled, otherwise 0. This is the proportion of word types the tagger
      can tag at all, the "vocabulary type" coverage of `A domain-independent semantic
      tagger for the study of meaning associations in English text 
      <https://eprints.whiterose.ac.uk/id/eprint/82299/1/ADomainIndependentSemanticTagger.pdf>`__
      (Section 5), which answers "how many of the different word types in
      language are covered by the system?".
    - ``"macro"``: the proportion of the word type's occurrences that are
      labelled. A rule-based tagger may tag a word in one context but not
      another, for example when it is part of a Multi Word Expression, and
      this gives partial credit for such words. Every word type has equal
      weight, however often it occurs. Weighting each word type by its
      number of occurrences instead gives token :func:`coverage`. This is an
      extension of the paper above, where coverage is measured by lexicon
      lookup, so every occurrence of a word type is either covered or not and
      the two averages are the same.

    A token is labelled using the same rule as :func:`coverage`: it has at
    least one tag that is not empty (``""``), not only whitespace, and not in
    ``label_filter``. Tokens excluded by ``exclude_labels`` (see
    :func:`is_excluded`) are left out completely, so a word type whose every
    occurrence is excluded is not part of :math:`W`. The same warning as for
    :func:`coverage` applies: tokens are excluded based on the tagger's own
    predicted tags.

    The texts are read once, one at a time, so a large corpus can be streamed
    as with :func:`coverage`. The memory used grows with the number of unique
    word types, not the number of tokens.

    Args:
        y_pred: The texts containing the semantic tags predicted by the tagger,
            either as an :class:`.EvaluationDataset` or any iterable of
            :class:`.EvaluationTexts`.
        label_filter: Semantic tags that do not count as a label, e.g.
            ``{"Z99"}``. Default is `None`, no tags are filtered.
        exclude_labels: Semantic tags whose tokens are left out of the
            calculation, e.g. ``{"Z9"}``. Default is `None`, no tokens are
            excluded.
        lowercase: Whether to lowercase tokens before comparing them, so that
            ``The`` and ``the`` are the same word type. Default is `False`.
        average: How each word type is scored, ``"binary"`` or ``"macro"``,
            see above. Default is ``"binary"``.

    Returns:
        float: The word type coverage, between 0.0 and 1.0.

    Raises:
        ValueError: If ``average`` is not ``"binary"`` or ``"macro"``.
        ValueError: If ``label_filter`` and ``exclude_labels`` share a tag.
        ValueError: If any text does not have semantic tags.
        ValueError: If there are no tokens, or every token is excluded.

    Examples:
        >>> from usas_evaluation_framework.dataset import EvaluationTexts
        >>> text = EvaluationTexts(
        ...     text="The cat saw the dog",
        ...     tokens=["The", "cat", "saw", "the", "dog"],
        ...     lemmas=None,
        ...     pos_tags=None,
        ...     semantic_tags=[["Z5"], ["L2"], ["Z99"], ["Z99"], ["Z99"]],
        ...     mwe_indexes=None,
        ... )

        Five word types (``The``, ``cat``, ``saw``, ``the``, ``dog``), two covered:

        >>> type_coverage([text], label_filter={"Z99"})
        0.4

        Lowercased there are four word types, ``the`` is covered by its first
        occurrence:

        >>> type_coverage([text], label_filter={"Z99"}, lowercase=True)
        0.5

        With macro averaging ``the`` scores 0.5, as one of its two occurrences
        is labelled, so the type coverage is (0.5 + 1 + 0 + 0) / 4:

        >>> type_coverage([text], label_filter={"Z99"}, lowercase=True, average="macro")
        0.375
    """
    if average not in ("binary", "macro"):
        raise ValueError("average must be either 'binary' or 'macro'")

    occurrence_counts: Counter[str] = Counter()
    labelled_counts: Counter[str] = Counter()
    for token, labelled in iter_token_labels(y_pred, label_filter, exclude_labels):
        word_type = token.lower() if lowercase else token
        occurrence_counts[word_type] += 1
        if labelled:
            labelled_counts[word_type] += 1

    if not occurrence_counts:
        raise ValueError("y_pred does not contain any tokens that are not excluded")

    match average:
        case "binary":
            # Only word types with a labelled occurrence are keys of `labelled_counts`.
            total_score = float(len(labelled_counts))
        case "macro":
            total_score = sum(
                labelled_counts[word_type] / count
                for word_type, count in occurrence_counts.items()
            )

    return total_score / len(occurrence_counts)


def iter_token_labels(y_pred: EvaluationDataset | Iterable[EvaluationTexts],
                      label_filter: set[str] | None = None,
                      exclude_labels: set[str] | None = None,
                      ) -> Iterator[tuple[str, bool]]:
    """
    Yield each token in the texts that is not excluded, with whether it has
    been labelled.

    The texts are read once, one at a time, and nothing is copied. See
    :func:`.is_labelled` for when a token counts as labelled, and
    :func:`is_excluded` for when a token is excluded.

    Args:
        y_pred: The texts, either as an :class:`.EvaluationDataset` or any
            iterable of :class:`.EvaluationTexts`.
        label_filter: Semantic tags that do not count as a label. Default is
            `None`, no tags are filtered.
        exclude_labels: Semantic tags whose tokens are not yielded. Default is
            `None`, no tokens are excluded.

    Yields:
        tuple[str, bool]: The token and True if it is labelled, False otherwise.

    Raises:
        ValueError: If ``label_filter`` and ``exclude_labels`` share a tag.
            As this is a generator, it is raised when iteration starts.
        ValueError: If any text does not have semantic tags.

    Examples:
        >>> from usas_evaluation_framework.dataset import EvaluationTexts
        >>> text = EvaluationTexts(
        ...     text="Hi !",
        ...     tokens=["Hi", "!"],
        ...     lemmas=None,
        ...     pos_tags=None,
        ...     semantic_tags=[["Q2.2"], ["Z9"]],
        ...     mwe_indexes=None,
        ... )
        >>> list(iter_token_labels([text], label_filter={"Z9"}))
        [('Hi', True), ('!', False)]
        >>> list(iter_token_labels([text], exclude_labels={"Z9"}))
        [('Hi', True)]
    """
    filtered_labels = label_filter or frozenset()
    excluded_labels = exclude_labels or frozenset()
    shared_labels = filtered_labels & excluded_labels
    if shared_labels:
        raise ValueError(
            f"label_filter and exclude_labels must not share any tags, "
            f"but both contain: {sorted(shared_labels)}"
        )

    match y_pred:
        case EvaluationDataset():
            texts: Iterable[EvaluationTexts] = y_pred.texts
        case _:
            texts = y_pred

    for i, text in enumerate(texts):
        if text.semantic_tags is None:
            raise ValueError(f"y_pred text {i} does not have semantic_tags")
        for token, token_tags in zip(text.tokens, text.semantic_tags, strict=True):
            if is_excluded(token_tags, excluded_labels):
                continue
            yield token, is_labelled(token_tags, filtered_labels)


def is_excluded(token_tags: list[str],
                exclude_labels: set[str] | frozenset[str] = frozenset()) -> bool:
    """
    Return True if a token should be left out of the coverage calculation.

    A token is excluded if it has at least one tag that is not empty and
    **every** such tag is in ``exclude_labels``. A token with
    no tags (``[]`` or ``[""]``) is never excluded. The match is on the whole
    tag string, so excluding ``Z9`` does not exclude ``Z9/Z99``.

    Args:
        token_tags: The semantic tags of a single token.
        exclude_labels: The semantic tags whose tokens are excluded. Default
            is an empty set, no tokens are excluded.

    Returns:
        bool: True if the token is excluded, False otherwise.

    Examples:
        >>> is_excluded(["Z9"], {"Z9"})
        True
        >>> is_excluded(["Z9", ""], {"Z9"})
        True
        >>> is_excluded(["Z9", "A1"], {"Z9"})
        False
        >>> is_excluded(["Z9/Z99"], {"Z9"})
        False
        >>> is_excluded([""], {"Z9"})
        False
    """
    non_empty_tags = [tag for tag in token_tags if tag.strip()]
    return bool(non_empty_tags) and all(tag in exclude_labels for tag in non_empty_tags)
