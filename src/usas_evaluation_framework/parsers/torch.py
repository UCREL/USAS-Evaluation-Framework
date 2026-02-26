import csv
import logging
from pathlib import Path
from typing import cast

from usas_evaluation_framework import data_utils
from usas_evaluation_framework.dataset import (
    EvaluationDataset,
    EvaluationTexts,
    TextLevel,
)
from usas_evaluation_framework.parsers.base import BaseParser

logger = logging.getLogger(__name__)

class TorchParser(BaseParser):
    """
    Parser for the ToRCH corpus that contains human-annotated USAS semantic tags.

    The ToRCH corpus does not contain MWEs.

    The corpus should be in CSV format with the following columns:
    - Token: The actual word/token text
    - Corrected-USAS: Semantic tag(s) in USAS format, this format can contain
        more than one USAS tag of which we expect them to be separated by semi-colon
        followed by an optional space `; `.
    - sentence-break: A boolean indicating if the token is a sentence break, if True
        then the next token in the CSV is the start of a new sentence.

    The main parsing method is `parse`
    """

    @staticmethod
    def parse(dataset_path: Path,
              label_validation: set[str] | None = None,
              label_filter: set[str] | None = None,
              ) -> EvaluationDataset:
        """
        Parses the ToRCH corpus into the Evaluation Dataset format, for
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

        NOTE: when parsing the ToRCH corpus many rules are used to create a USAS
        validated dataset. Some of the token semantic tags are removed as they
        are not valid tags and are replaced with an empty string or if possible
        with a valid tag. In addition if there are no corrected USAS tags and the
        predicted USAS tags are `PUNCT` then the semantic tags will also be `PUNCT`.
        
        Args:
            dataset_path: Path to the ToRCH corpus, should be in CSV format.
            label_validation: A set of labels that the semantic/dataset labels should
                be validated against. Defaults to `None` in which case no validation
                is performed. NOTE: label validation is not performed on USAS tags
                that will not be returned, i.e. all USAS tags after the first `;`.
            label_filter: A set of labels from the dataset that should be filtered out.
                Defaults to `None` in which case no filtering is performed.
        Returns:
            EvaluationDataset: The parsed and formatted dataset. The name of the
                dataset is set to `Torch` and the text level is set to
                `sentence`. The text returned for this corpus is just the tokens
                joined together by a single space for each sentence.
        Raises:
            ValueError: If it cannot parse the data due to formatting or a
                label cannot be validated when label validation is used.
        """
        quantifier_row_indexes = set(
            [
                23,
                53,
                88,
                92,
                111,
                148,
                165,
                191,
                252,
                285,
                321,
                389,
                535,
                559,
                620,
                680,
                791,
                820,
                834,
                885,
                914,
                941,
                1026,
                1036,
                1049,
                1102,
                1109,
                1113,
                1117,
                1125,
                1129,
                1136,
                1162,
                1174,
                1199,
            ]
        )

        def label_string_to_labels(label_string: str) -> list[str]:
            label_string = label_string
            labels: list[str] | None = None
            if "；" in label_string:
                labels = label_string.split("；")
            elif "," in label_string:
                labels = label_string.split(",")
            else:
                labels = label_string.split()
            cleaned_labels = []
            for label in labels:
                label = label.strip()
                if label:
                    cleaned_labels.append(label)
            return cleaned_labels

        def validate_first_label(labels: list[str]) -> str:
            if len(labels) == 0:
                raise ValueError("Error expected at least one label")
            label = labels[0]
            label_groups = data_utils.parse_usas_token_group(label)
            if len(label_groups) != 1:
                raise ValueError(
                    f"Error expected only one label group: {label_groups}"
                )
            label_group = label_groups[0]
            
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

        dataset_name = "Torch"
        text_level = TextLevel.sentence

        logger.info(f"Parsing the {dataset_name} dataset found at: {dataset_path}")
        
        using_label_validation = True if label_validation is not None else False
        logger.info(f"Using label valdation: {using_label_validation}")
    
        using_label_filtering = True if label_filter is not None else False
        logger.info(f"Using label filtering: {using_label_filtering}")

        evaluation_texts: list[EvaluationTexts] = []

        string_to_bool = {"true": True, "false": False}

        expected_field_names = set({"Token", "Corrected-USAS", "sentence-break"})
        tokens: list[str] = []
        semantic_tags: list[str] = []
        with dataset_path.open("r", encoding="utf-8", newline="") as dataset_fp:
            dataset_csv_reader = csv.DictReader(dataset_fp)
            csv_field_names = dataset_csv_reader.fieldnames
            if not isinstance(csv_field_names, list):
                raise ValueError(
                    f"Error expected at least the following field names: {expected_field_names} "
                    f"but got: {csv_field_names}"
                )
            if set(csv_field_names).intersection(expected_field_names) != expected_field_names:
                raise ValueError(
                    f"Error expected at least the following field names: {expected_field_names} "
                    f"but got: {csv_field_names}"
                )
            
            for row_index, dataset_row in enumerate(dataset_csv_reader, start=2):
                token = cast(str, dataset_row["Token"].strip())
                corrected_usas_string = cast(str, dataset_row["Corrected-USAS"].strip())
                
                # Rules to overcome annotation issues
                if row_index in quantifier_row_indexes and corrected_usas_string == "N":
                    corrected_usas_string = "N5"
                
                if corrected_usas_string == "":
                    if "Predicted-USAS" in dataset_row:
                        predicted_usas_string = cast(str, dataset_row["Predicted-USAS"].strip())
                        if predicted_usas_string == "PUNCT":
                            corrected_usas_string = predicted_usas_string
                
                # Rules to overcome annotation issues
                skip_validation = False
                match (row_index, corrected_usas_string):
                    case (78, "A1") | (663, "") | (1457, "E4") | (1705, "N99") | (1706, "N99") | (1768, "N99"):
                        skip_validation = True
                
                if not skip_validation:
                    corrected_usas_labels = label_string_to_labels(corrected_usas_string)
                    correct_usas_label: str | None = None
                    try:
                        correct_usas_label = validate_first_label(corrected_usas_labels)
                    except ValueError as e:
                        raise ValueError(
                                f"Error parsing the {dataset_name} dataset at row: {row_index} with {dataset_row} "
                            ) from e
                else:
                    correct_usas_label = ""

                try:
                    validate_token(token)
                except ValueError as e:
                    raise ValueError(
                            f"Error parsing the {dataset_name} dataset at row: {row_index} with {dataset_row} "
                        ) from e
                
                tokens.append(token)
                semantic_tags.append(correct_usas_label)


                is_sentence_break = cast(bool, string_to_bool[dataset_row["sentence-break"].strip().lower()])
                if is_sentence_break:
                    evaluation_text = EvaluationTexts(text=" ".join(tokens),
                                                      tokens=tokens,
                                                      lemmas=None,
                                                      pos_tags=None,
                                                      semantic_tags=semantic_tags,
                                                      mwe_indexes=[frozenset({})] * len(tokens))
                    evaluation_texts.append(evaluation_text)
                    tokens = []
                    semantic_tags = []
        if tokens:
            evaluation_text = EvaluationTexts(text=" ".join(tokens),
                                              tokens=tokens,
                                              lemmas=None,
                                              pos_tags=None,
                                              semantic_tags=semantic_tags,
                                              mwe_indexes=[frozenset({})] * len(tokens))
            evaluation_texts.append(evaluation_text)
        

        logger.info(f"Finished parsing the {dataset_name} dataset")

        return EvaluationDataset(name=dataset_name,
                                 text_level=text_level,
                                 labels_removed=label_filter,
                                 texts=evaluation_texts)
                
