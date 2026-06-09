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


class NAACL2015USAS(BaseParser):
    """
    Parser for NAACL 2015 USAS corpora (Chinese, Italian, and others).

    The corpus is stored as a CSV file with the following columns:

    - ``id``: token identifier in the format ``{article}|{sentence_id}|{token_id}``
    - ``token``: the token text
    - ``corrected_usas``: human-corrected USAS tag (single tag or multi-tag
      ``/``-separated, or empty when no tag is applicable)
    - ``mwe`` *(optional)*: integer group identifier for Multi Word Expressions.
      Blank means the token is not part of any MWE; a positive integer (stored
      as a float string such as ``1.0``) means the token belongs to that MWE
      group. All tokens sharing the same integer form one MWE.

    Sentences are delimited by empty rows in the CSV.

    The resolved USAS tag for each token is determined as follows:

    1. If ``corrected_usas`` is non-empty, it is used directly as the tag.
    2. Otherwise an empty string is assigned.

    When the ``mwe`` column is absent, ``mwe_indexes`` is ``None`` on every
    returned :class:`EvaluationTexts`.  When the column is present,
    ``mwe_indexes`` is a list of :class:`frozenset` — ``frozenset()`` for
    tokens not in any MWE, ``frozenset({n})`` for tokens in MWE group *n*.

    The main parsing method is :meth:`parse`.
    """

    @staticmethod
    def _resolve_usas_tag(corrected_usas: str) -> str:
        """
        Resolve and normalise the USAS tag for a single token.

        Args:
            corrected_usas: Raw corrected USAS field (single tag, multi-tag
                ``/``-separated, or empty).
        Returns:
            Normalised USAS tag string, or ``''`` when no tag is present.
        Raises:
            ValueError: If the tag cannot be parsed as a valid USAS tag.
        """
        raw_tag = corrected_usas.strip()
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
        dataset_name: str | None = "NAACL 2015",
        language: str | None = None,
    ) -> EvaluationDataset:
        """
        Parse a NAACL 2015 USAS corpus CSV into the Evaluation Dataset format.

        The CSV must contain columns ``id``, ``token``, and ``corrected_usas``.
        If an ``mwe`` column is also present, MWE information is read and
        propagated to ``mwe_indexes`` on each :class:`EvaluationTexts`.

        Empty tags are not checked against ``label_validation``.

        For multi-tag labels such as ``Q4.2/I3.2/S2``, each sub-tag is checked
        individually against ``label_validation``.  The complete multi-tag string
        must appear in ``label_filter`` for the token to be filtered out.

        Args:
            dataset_path: Path to the corpus CSV file.
            label_validation: Optional set of valid semantic labels.  When
                supplied, every resolved tag (excluding ``''``) is checked
                against this set per sub-tag.
            label_filter: Optional set of labels to suppress.  Matching tokens
                receive an empty-string tag.
            dataset_name: Name for the returned dataset.
                Defaults to ``'NAACL 2015'``.
            language: Language of the corpus. Defaults to ``None``.
        Returns:
            EvaluationDataset: Parsed dataset at sentence-level granularity.
            Tokens and semantic tags are populated; lemmas and POS tags are
            ``None``.  ``mwe_indexes`` is ``None`` when no ``mwe`` column is
            present, otherwise a list of frozensets per sentence.
        Raises:
            ValueError: If a token ID does not match the expected format, a
                tag fails label validation, or an MWE cell contains a value
                that cannot be parsed as a positive integer.
        """
        if dataset_name is None:
            dataset_name = "NAACL 2015"

        text_level = TextLevel.sentence

        logger.info(f"Parsing the {dataset_name} dataset found at: {dataset_path}")
        logger.info(f"Using label validation: {label_validation is not None}")
        logger.info(f"Using label filtering: {label_filter is not None}")

        evaluation_texts: list[EvaluationTexts] = []
        sentence_rows: list[list[str]] = []
        current_sentence_key: str | None = None
        has_mwe = False  # updated after reading the header

        def flush(rows: list[list[str]]) -> None:
            if not rows:
                return

            tokens: list[str] = []
            usas_tags: list[str] = []
            mwe_idx_list: list[frozenset[int]] = []

            for row in rows:
                while len(row) < 3:
                    row.append('')

                token = row[1].strip()
                corrected_usas = row[2].strip()

                try:
                    usas_tag = NAACL2015USAS._resolve_usas_tag(corrected_usas)
                except ValueError as e:
                    raise ValueError(
                        f"Error resolving USAS tag for token '{token}': {e}"
                    ) from e

                tokens.append(token)
                usas_tags.append(usas_tag)

                if has_mwe:
                    mwe_raw = row[3].strip() if len(row) > 3 else ''
                    if mwe_raw:
                        try:
                            mwe_int = int(float(mwe_raw))
                            if mwe_int <= 0:
                                raise ValueError(
                                    f"MWE index must be a positive integer, got '{mwe_raw}'"
                                )
                            mwe_idx_list.append(frozenset({mwe_int}))
                        except (ValueError, OverflowError) as exc:
                            raise ValueError(
                                f"Invalid MWE value '{mwe_raw}' for token '{token}'"
                            ) from exc
                    else:
                        mwe_idx_list.append(frozenset())

            # Re-index MWE groups to start at 1 within this sentence, so that
            # annotator-level indexes that span sentence boundaries are normalised.
            if has_mwe:
                remap: dict[int, int] = {}
                next_local = 1
                remapped: list[frozenset[int]] = []
                for fs in mwe_idx_list:
                    if fs:
                        orig = next(iter(fs))
                        if orig not in remap:
                            remap[orig] = next_local
                            next_local += 1
                        remapped.append(frozenset({remap[orig]}))
                    else:
                        remapped.append(frozenset())
                mwe_idx_list = remapped

            # Label filtering
            if label_filter is not None:
                usas_tags = ['' if tag in label_filter else tag for tag in usas_tags]

            # Label validation
            if label_validation is not None:
                for usas_tag in usas_tags:
                    if usas_tag == '':
                        continue
                    for sub_tag in usas_tag.split('/'):
                        if sub_tag not in label_validation:
                            raise ValueError(
                                f"Semantic tag '{sub_tag}' is not in the label validation set"
                            )

            evaluation_texts.append(
                EvaluationTexts(
                    text=' '.join(tokens),
                    tokens=tokens,
                    lemmas=None,
                    pos_tags=None,
                    semantic_tags=create_inner_list(usas_tags),
                    mwe_indexes=mwe_idx_list if has_mwe else None,
                )
            )

        with dataset_path.open('r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            has_mwe = len(header) > 3 and header[3].strip() == 'mwe'

            for row in reader:
                # Empty row → sentence boundary
                if not any(field.strip() for field in row):
                    flush(sentence_rows)
                    sentence_rows = []
                    current_sentence_key = None
                    continue

                row_id = row[0]
                parts = row_id.split('|')
                if len(parts) != 3:
                    raise ValueError(
                        f"Token ID '{row_id}' does not match expected format "
                        "'<article>|<sentence_id>|<token_id>'"
                    )
                sentence_key = '|'.join(parts[:2])

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
            language=language,
            labels_removed=label_filter,
            texts=evaluation_texts,
        )
