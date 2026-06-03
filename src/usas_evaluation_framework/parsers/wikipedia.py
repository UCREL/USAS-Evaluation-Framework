import csv
import logging
from pathlib import Path

from usas_evaluation_framework.data_utils import (
    create_inner_list,
    parse_usas_token_group,
)
from usas_evaluation_framework.dataset import (
    EvaluationDataset,
    EvaluationTexts,
    TextLevel,
)
from usas_evaluation_framework.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class SpanishWikipediaUSAS(BaseParser):
    """
    Parser for the Spanish Wikipedia USAS corpus that contains human-annotated
    USAS semantic tags.

    The corpus is stored as a CSV file with the following columns:

    - ``id``: globally unique token identifier in the format
      ``{language}|{article}|{sentence_id}|{token_id}``
    - ``sentence id``: sentence index (resets per article)
    - ``token id``: token index within the sentence
    - ``token``: the token text
    - ``lemma``: the lemma of the token
    - ``POS``: part-of-speech tag
    - ``predicted USAS``: predicted USAS tags, multiple tags separated by ``"; "``
    - ``predicted MWE``: predicted MWE group identifier (numeric, empty if none)
    - ``corrected USAS``: human-corrected USAS tag (single tag, empty if
      predicted was correct)
    - ``corrected MWE``: corrected MWE group identifier (empty if predicted
      was correct)
    - ``Comments``: annotator comments (ignored during parsing)

    Sentences are delimited by empty rows in the CSV.  The resolved USAS tag
    for each token is determined as follows:

    1. If ``corrected USAS`` is non-empty, the first ``;``-separated tag is used.
    2. Else if the POS tag is ``PUNCT``, the USAS tag ``PUNCT`` is assigned.
    3. Otherwise the first ``;``-separated tag from ``predicted USAS`` is used.

    MWE group resolution: ``corrected MWE`` is used when non-empty; otherwise no
    MWE grouping is applied (``predicted MWE`` is ignored).  MWE group identifiers
    are remapped to sequential integers (1, 2, ...) within each sentence as sometimes 
    it is the case that MWE within a sentence start an integer that is not 1, e.g. 
    they carry on the MWE number from the previous sentence.

    This corpus supports MWEs and provides lemmas and POS tags.

    The main parsing method is :meth:`parse`.
    """

    @staticmethod
    def _resolve_usas_tag(predicted_usas: str, corrected_usas: str, pos: str) -> str:
        """
        Resolve and normalise the USAS tag for a single token.

        Args:
            predicted_usas: Raw predicted USAS field (may contain multiple
                ``;``-separated tags).
            corrected_usas: Raw corrected USAS field (single tag or empty).
            pos: POS tag for the token.
        Returns:
            Normalised USAS tag string, ``'PUNCT'`` for punctuation tokens, or
            ``''`` when no tag can be determined.
        Raises:
            ValueError: If the resolved raw tag cannot be parsed as a USAS tag.
        """
        if corrected_usas:
            raw_tag = corrected_usas.split(';')[0].strip()
        elif pos == 'PUNCT':
            return 'PUNCT'
        elif predicted_usas:
            raw_tag = predicted_usas.split(';')[0].strip()
        else:
            return ''

        if not raw_tag:
            return ''

        try:
            groups = parse_usas_token_group(raw_tag)
            return '/'.join(t.tag for t in groups[0].tags)
        except ValueError as e:
            raise ValueError(f"Invalid USAS tag '{raw_tag}'") from e

    @staticmethod
    def parse(
        dataset_path: Path,
        label_validation: set[str] | None = None,
        label_filter: set[str] | None = None,
    ) -> EvaluationDataset:
        """
        Parse the Spanish Wikipedia USAS corpus into the Evaluation Dataset format.

        If a token does not have a corrected USAS tag, the first ``;``-separated,
        but does have a POS tag of ``PUNCT``, then ``PUNCT`` is used. ``PUNCT``
        is always included in the label validation set.
        
        Empty tags (produced by label filtering), which can include ``PUNCT``,
        are not checked against ``label_validation``.

        For multi-tag labels such as ``L1/F1``, the complete multi-tag string
        must appear in ``label_filter`` for the token to be filtered out.

        Args:
            dataset_path: Path to the Spanish Wikipedia USAS CSV file.
            label_validation: Optional set of valid semantic labels.  When
                supplied, every resolved tag (excluding ``PUNCT`` and ``''``) is
                checked against this set.
            label_filter: Optional set of labels to suppress.  Matching tokens
                receive an empty-string tag.
        Returns:
            EvaluationDataset: Parsed dataset named ``'Spanish Wikipedia USAS'``
            at sentence-level granularity.  Tokens, lemmas, POS tags, semantic
            tags, and MWE indexes are all populated.
        Raises:
            ValueError: If a token cannot be parsed or a tag fails label
                validation.
        """
        dataset_name = "Spanish Wikipedia USAS"
        text_level = TextLevel.sentence

        logger.info(f"Parsing the {dataset_name} dataset found at: {dataset_path}")
        logger.info(f"Using label validation: {label_validation is not None}")
        logger.info(f"Using label filtering: {label_filter is not None}")

        evaluation_texts: list[EvaluationTexts] = []
        sentence_rows: list[list[str]] = []
        current_sentence_key: str | None = None

        def flush(rows: list[list[str]]) -> None:
            if not rows:
                return

            tokens: list[str] = []
            lemmas: list[str] = []
            pos_tags: list[str] = []
            usas_tags: list[str] = []
            raw_mwe_ids: list[int | None] = []

            for row in rows:
                # Pad short rows so index accesses are safe
                while len(row) < 10:
                    row.append('')

                token = row[3].strip()
                lemma = row[4].strip()
                pos = row[5].strip()
                predicted_usas = row[6].strip()
                # predicted_mwe = row[7].strip()
                corrected_usas = row[8].strip()
                corrected_mwe = row[9].strip()

                try:
                    usas_tag = SpanishWikipediaUSAS._resolve_usas_tag(
                        predicted_usas, corrected_usas, pos
                    )
                except ValueError as e:
                    raise ValueError(
                        f"Error resolving USAS tag for token '{token}': {e}"
                    ) from e

                if corrected_mwe:
                    mwe_id: int | None = int(float(corrected_mwe))
                else:
                    mwe_id = None

                tokens.append(token)
                lemmas.append(lemma)
                pos_tags.append(pos)
                usas_tags.append(usas_tag)
                raw_mwe_ids.append(mwe_id)

            # Validate that no token text is itself a parseable USAS tag
            for token in tokens:
                token_is_a_tag = True
                try:
                    parse_usas_token_group(token)
                except ValueError:
                    token_is_a_tag = False
                if token_is_a_tag:
                    raise ValueError(
                        f"Token '{token}' appears to be a USAS tag rather than a word token"
                    )

            # Label filtering
            if label_filter is not None:
                usas_tags = ['' if tag in label_filter else tag for tag in usas_tags]

            # Label validation
            if label_validation is not None:
                for usas_tag in usas_tags:
                    match usas_tag:
                        case 'PUNCT' | '':
                            continue
                    for sub_tag in usas_tag.split('/'):
                        if sub_tag not in label_validation:
                            raise ValueError(
                                f"Semantic tag '{sub_tag}' is not in the label validation set"
                            )

            # Remap sentence-local MWE IDs to sequential integers (1, 2, ...)
            unique_ids = sorted(set(m for m in raw_mwe_ids if m is not None))
            id_to_seq = {m: i + 1 for i, m in enumerate(unique_ids)}
            mwe_indexes: list[frozenset[int]] = [
                frozenset({id_to_seq[m]}) if m is not None else frozenset()
                for m in raw_mwe_ids
            ]

            evaluation_texts.append(
                EvaluationTexts(
                    text=' '.join(tokens),
                    tokens=tokens,
                    lemmas=lemmas,
                    pos_tags=pos_tags,
                    semantic_tags=create_inner_list(usas_tags),
                    mwe_indexes=mwe_indexes,
                )
            )

        with dataset_path.open('r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # skip header row

            for row in reader:
                # Empty row → sentence boundary
                if not any(field.strip() for field in row):
                    flush(sentence_rows)
                    sentence_rows: list[list[str]] = []
                    current_sentence_key = None
                    continue

                row_id = row[0]
                parts = row_id.split('|')
                if len(parts) != 4:
                    raise ValueError(
                        f"Token ID '{row_id}' does not match expected format "
                        "'<language>|<article>|<sentence_id>|<token_id>'"
                    )
                sentence_key = '|'.join(parts[:3])

                # The elif occurs when the data starts a new file in the dataset
                if current_sentence_key is None:
                    current_sentence_key = sentence_key
                elif current_sentence_key != sentence_key:
                    flush(sentence_rows)
                    sentence_rows = []
                    current_sentence_key = sentence_key

                sentence_rows.append(row)

        flush(sentence_rows)  # flush final sentence

        logger.info(f"Finished parsing the {dataset_name} dataset")
        return EvaluationDataset(
            name=dataset_name,
            text_level=text_level,
            labels_removed=label_filter,
            texts=evaluation_texts,
        )
