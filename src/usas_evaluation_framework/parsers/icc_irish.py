import csv
import logging
import re
from pathlib import Path

from usas_evaluation_framework import data_utils
from usas_evaluation_framework.dataset import (
    EvaluationDataset,
    EvaluationTexts,
    TextLevel,
)
from usas_evaluation_framework.parsers.base import BaseParser

logger = logging.getLogger(__name__)

class ICCIrishParser(BaseParser):
    """
    Parser for the International Comparable Corpus (ICC) Irish corpus that
    contains human-annotated USAS semantic tags.

    This corpus does contain / support MWEs.

    The corpus should be in TSV format with the following columns:
    - TOKEN: The actual word/token text
    - UPOS: The Universal Dependency POS tag
    - MWE: Semantic tag(s) in USAS format, this format can contain
        more than one USAS tag of which we expect them to be separated by semi-colon
        followed by an optional space `; `.
    - USAS: A boolean indicating if the token is a sentence break, if True
        then the next token in the CSV is the start of a new sentence.

    The main parsing method is `parse`
    """

    @staticmethod
    def parse(dataset_path: Path,
              label_validation: set[str] | None = None,
              label_filter: set[str] | None = None,
              ) -> EvaluationDataset:
        """
        Parses the ICC Irish corpus into the Evaluation Dataset format, for
        easy evaluation of USAS WSD models.
        
        If the label filter is used then the semantic label for that token will be
        an empty string, this empty string will not be raised as a validation error
        through label validation if this is not None.

        For multi-tag labels, e.g. `F2/O2`, the full multi-tag label must be in
        the label filter for it to be filtered out, e.g. if `F2` is in the label
        filter then it will not affect the multi-tag label `F2/O2` only `F2`
        labels will be filtered out.

        The label validation does not need to include the `PUNCT` label as this is
        always validated and is expected to be a valid semantic tag.

        NOTE: when parsing the ICC Irish corpus many rules are used to create a USAS
        validated dataset. Some of the token semantic tags are removed as they
        are not valid tags and are replaced with an empty string or if possible
        with a valid tag.

        NOTE: We convert all punctuation from the `Z9` USAS tag to the `PUNCT`
        tag, this is to be consistent with how the other datasets are parsed.

        NOTE: we do not parse the lemma, UPOS, or PAROLE tag information, this is a TODO for
        future work.
        
        Args:
            dataset_path: Path to the ICC Irish corpus, which should be in TSV format.
            label_validation: A set of labels that the semantic/dataset labels should
                be validated against. Defaults to `None` in which case no validation
                is performed.
            label_filter: A set of labels from the dataset that should be filtered out.
                Defaults to `None` in which case no filtering is performed.
        Returns:
            EvaluationDataset: The parsed and formatted dataset. The name of the
                dataset is set to `ICCIrish` and the text level is set to
                `paragraph`. The text returned for this corpus is just the tokens
                joined together by a single space for the given paragraph/text. We assume
                each TSV file is one paragraph, thus the length of returned
                EvaluationDataset.texts is 1.
        Raises:
            ValueError: If it cannot parse the data due to formatting or a
                label cannot be validated when label validation is used.
            ValueError: If UPOS is PUNCT and the USAS is not "Z9".
        """

        mwe_index_re = re.compile(r"^\((\d+),\s*(\d+)\)$")

        def validate_mwe_indexes(mwe_index_string: str,
                                 previous_mwe_index: tuple[int, int],
                                 previous_mwe_index_value: int,
                                 all_previous_mwe_indexes: set[tuple[int, int]],
                                 in_mwe: bool) -> tuple[frozenset[int], tuple[int, int], bool, int]:
            mwe_match = mwe_index_re.findall(mwe_index_string)
            if len(mwe_match) != 1 or len(mwe_match[0]) != 2:
                raise ValueError(
                    f"Error expected one MWE index pair: {mwe_index_string} in the format `(10, 11)`"
                )
            current_mwe_index = (int(mwe_match[0][0]), int(mwe_match[0][1]))
            if current_mwe_index[0] >= current_mwe_index[1]:
                raise ValueError(
                    f"Error expected first MWE index to be less than the second MWE index: {mwe_index_string}"
                )
            if current_mwe_index == previous_mwe_index:
                if (current_mwe_index[1] - 1) <= current_mwe_index[0]:
                    raise ValueError(
                        f"Error cannot have an MWE that is a single token: {mwe_index_string}"
                    )
                return frozenset({previous_mwe_index_value}), current_mwe_index, in_mwe, previous_mwe_index_value
            
            if current_mwe_index in all_previous_mwe_indexes:
                raise ValueError(
                    f"Error expected MWE index pairs to occur in ascending order: {mwe_index_string}"
                )
            for previous_mwe_index in all_previous_mwe_indexes:
                if current_mwe_index[0] < previous_mwe_index[1]:
                    raise ValueError(
                        f"Error expected MWE index pairs to be non-overlapping: {mwe_index_string}"
                    )
            all_previous_mwe_indexes.add(current_mwe_index)
            if in_mwe:
                in_mwe = False
                previous_mwe_index_value += 1
            if (current_mwe_index[1] - 1) != current_mwe_index[0]:
                in_mwe = True
                return frozenset({previous_mwe_index_value}), current_mwe_index, in_mwe, previous_mwe_index_value

            return frozenset({}), current_mwe_index, in_mwe, previous_mwe_index_value
            
            


        def validate_label(label: str, pos: str) -> str:
            label_groups = data_utils.parse_usas_token_group(label)
            if len(label_groups) != 1:
                raise ValueError(
                    f"Error expected only one label group: {label_groups}"
                )
            label_group = label_groups[0]

            label_tag = "PUNCT"
            if pos == "PUNCT":
                if label_group.tags[0].tag != "Z9":
                    raise ValueError(
                        "Error expected USAS tag to be `Z9` when POS is `PUNCT`"
                    )
            else:
                label_tag_string_list = []
                for _label_tag in label_group.tags:
                    
                    if _label_tag.tag != "PUNCT" and label_validation is not None:
                        if _label_tag.tag not in label_validation:
                            raise ValueError(
                                f"Error expected label: {_label_tag.tag} to be in label validation set: {label_validation}"
                            )
                    label_tag_string_list.append(_label_tag.tag)
                label_tag = "/".join(label_tag_string_list)
            
            if label_filter is not None:
                if label_tag in label_filter:
                    label_tag = ""
            return label_tag

        def validate_token(token: str) -> None:
            if not token:
                raise ValueError(
                    f"Error expected token not to be empty: {token}"
                )
            # Validate token is not a USAS tag
            token_is_a_tag = True
            try:
                data_utils.parse_usas_token_group(token)
            except ValueError:
                token_is_a_tag = False
            if token_is_a_tag:
                raise ValueError(
                    f"Error expected token is a tag: {token}"
                )

        dataset_name = "ICCIrish"
        text_level = TextLevel.paragraph

        logger.info(f"Parsing the {dataset_name} dataset found at: {dataset_path}")
        
        using_label_validation = True if label_validation is not None else False
        logger.info(f"Using label validation: {using_label_validation}")
    
        using_label_filtering = True if label_filter is not None else False
        logger.info(f"Using label filtering: {using_label_filtering}")

        evaluation_texts: list[EvaluationTexts] = []
        required_fields = set({"TOKEN", "UPOS", "MWE", "USAS"})

        
        tokens: list[str] = []
        semantic_tags: list[str] = []
        mwe_indexes: list[frozenset[int]] = []
        
        previous_mwe_index: tuple[int, int] = (-1, -1)
        in_mwe = False
        mwe_index_value: int = 1
        all_mwe_indexes: set[tuple[int, int]] = set()
        with dataset_path.open("r", encoding="utf-8", newline="") as tsv_file:
            

            tsv_reader = csv.DictReader(
                tsv_file, delimiter="\t", quoting=csv.QUOTE_NONE
            )

            fields = tsv_reader.fieldnames
            if fields is None:
                raise ValueError(
                    f"Could not read field names from the {dataset_name} dataset"
                )
            elif not required_fields.issubset(set(fields)):
                raise ValueError(
                    f"Missing required fields from the {dataset_name} dataset: {required_fields.difference(set(fields))}"
                )
                

            for row_index, row in enumerate(tsv_reader):
                token = row["TOKEN"]
                try:
                    validate_token(token)
                except ValueError as e:
                    raise ValueError(
                        f"Error parsing the {dataset_name} dataset: for row_index: {row_index} for row: {row} {e}"
                    )

                usas_label = row["USAS"]

                skip_validation = False
                match (row_index, usas_label, token):
                    case (108, "G1.2/S7", "sochpholaitiúla"):
                        skip_validation = True
                    case (110, "X5", "Díríonn"):
                        skip_validation = True
                
                if not skip_validation:
                    try:
                        pos = row["UPOS"]
                        usas_label = validate_label(usas_label, pos)
                    except ValueError as e:
                        raise ValueError(
                            f"Error parsing the {dataset_name} dataset: for row_index: {row_index} for row: {row} {e}"
                        )
                else:
                    usas_label = ""

                mwe_index_string = row["MWE"]
                try:
                    current_mwe_index, previous_mwe_index, in_mwe, mwe_index_value = validate_mwe_indexes(mwe_index_string, previous_mwe_index, mwe_index_value, all_mwe_indexes, in_mwe)
                    all_mwe_indexes.add(previous_mwe_index)
                except ValueError as e:
                    raise ValueError(
                        f"Error parsing the {dataset_name} dataset: for row_index: {row_index} for row: {row} {e}"
                    )
                
                
                
                tokens.append(token)
                semantic_tags.append(usas_label)
                mwe_indexes.append(current_mwe_index)


        logger.info(f"Finished parsing the {dataset_name} dataset")

        semantic_tags_with_inner_list = data_utils.create_inner_list(semantic_tags)
        evaluation_texts = [
            EvaluationTexts(text=" ".join(tokens),
                            tokens=tokens,
                            lemmas=None,
                            pos_tags=None,
                            semantic_tags=semantic_tags_with_inner_list,
                            mwe_indexes=mwe_indexes)
        ]
        return EvaluationDataset(name=dataset_name,
                                 text_level=text_level,
                                 labels_removed=label_filter,
                                 texts=evaluation_texts)