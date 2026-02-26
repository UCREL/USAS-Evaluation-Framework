import logging
import re
from pathlib import Path
from typing import TypedDict

from usas_evaluation_framework.data_utils import (
    parse_usas_token_group,
)
from usas_evaluation_framework.dataset import (
    EvaluationDataset,
    EvaluationTexts,
    TextLevel,
)
from usas_evaluation_framework.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class MWEData(TypedDict):
    """
    Data structure for storing MWE information.

    Attributes:
        total_tokens (int): The total number of tokens in the MWE.
        token_indices (list[int]): A list of token indexes that are part of the MWE.
    """
    total_tokens: int
    token_indices: list[int]


class EnglishBenedict(BaseParser):
    """
    Parser for the Benedict English corpus that contains human-annotated USAS semantic tags.

    This parser handles the English version of the Benedict corpus, which uses a specific
    token format: '<Token>_<USAS-Label><MWE>?' where:
    - Token: The actual word/token text
    - USAS-Label: Semantic tag(s) in USAS format (e.g., 'F2/O4.5', 'A1.1.1')
    - MWE: Optional Multi-Word Expression identifier in format '[i<ID>.<TOTAL>.<INDEX>]'

    This corpus does contain / support MWEs.

    The parser provides comprehensive validation, normalization, and parsing capabilities
    for evaluating USAS Word Sense Disambiguation models.

    The main parsing method is `parse`
    """


    @staticmethod
    def validate_text_string_format(text: str) -> tuple[str, list[str], list[str]]:
        """
        Given a text whereby when split by whitespace each text element
        has the following format '<Token>_<USAS-Label><MWE>?' whereby the USAS-Label
        should be validated by `parse_usas_token_group`, whereby the MWE is optional
        and will be validated later by `get_mwe_indexes`. This validator ensures
        that the text string as a whole is valid according to the token text and
        USAS label format. It returns; the text string, the tokens,
        and the USAS labels.

        The USAS labels returned take into account the required conversion of
        `PUNC`, `-`, `.`, `,`, and `!` within the USAS label section of the text
        string to the USAS label `PUNCT`.

        Args:
            text: The text to validate, e.g. `Turkish_F2/O4.5[i86.2.1 grind_F2/O4.5[i86.2.2 
                -_- extremely_A13.3 finely_O4.5 ground_A1.1.1 ,_PUNC dust-like_O4.1`

        Returns:
            tuple[str, list[str], list[str]]: Format validated text string, tokens, and the USAS labels.
        Raises:
            ValueError: If the text string is not valid.
            ValueError: If the USAS label is not valid
        Examples:
            >>> EnglishBenedict.validate_text_string('Turkish_F2/O4.5[i86.2.1 grind_F2/O4.5[i86.2.2 -_- extremely_A13.3 finely_O4.5 ground_A1.1.1 ,_PUNC dust-like_O4.1')
            ('Turkish_F2/O4.5[i86.2.1 grind_F2/O4.5[i86.2.2 -_- extremely_A13.3 finely_O4.5 ground_A1.1.1 ,_PUNC dust-like_O4.1',
             ['Turkish', 'grind', '-', 'extremely', 'finely', 'ground', ',', 'dust-like'],
             ['F2/O4.5', 'F2/O4.5', 'PUNCT', 'A13.3', 'O4.5', 'A1.1.1', 'PUNCT', 'O4.1'])
        """
        if not text.strip():
            raise ValueError(f"Empty or whitespace-only text string: '{text}'")

        tokens = text.split()
        token_texts: list[str] = []
        usas_tags: list[str] = []
        special_punct_USAS_tags = set(['PUNC', '-', '.', ',', '!'])

        for token in tokens:
            # Split token into text and USAS/MWE parts
            if '_' not in token:
                raise ValueError(f"Invalid token format in text: {text}, expected "
                                 f"a single underscore in token: {token}")
            
            
            parts = token.split('_')
            if len(parts) != 2:
                raise ValueError(f"Invalid token format in text: {text}, expected "
                                 f"exactly one underscore in token: {token}")
            
            token_text, usas_mwe_info = parts
            
            # Check if token_text is empty
            if not token_text:
                raise ValueError(f"Token text is empty in token: {token} for text: {text}")
            token_texts.append(token_text)
            
            # Extract USAS tag (before any MWE marker)
            # MWE markers start with [i, so we need to extract everything before that
            mwe_start = usas_mwe_info.find('[i')
            if mwe_start == 0:
                # No USAS tag, only MWE - this is invalid
                raise ValueError(f"Token has MWE but no USAS tag: {token} for text: {text}")
            
            usas_tag_string = usas_mwe_info if mwe_start == -1 else usas_mwe_info[:mwe_start]
            
            # Check if USAS tag string is one of the special PUNCT USAS tags to be converted
            # We need to do this BEFORE validation since these are not valid USAS tags
            if usas_tag_string in special_punct_USAS_tags:
                usas_tags.append('PUNCT')
            else:
                # Validate USAS tag
                if not usas_tag_string:
                    raise ValueError(f"USAS tag is empty in token: {token} for text: {text}")
                try:
                    usas_tag_groups = parse_usas_token_group(usas_tag_string)
                    # We only want the first USAS tag group as there should only be one USAS tag
                    # group by definition. We then want each tag from that first group.
                    # There can be multiple tags because of multi tag membership, e.g.
                    # F2/O2
                    usas_tag = "/".join([usas_tag.tag for usas_tag in usas_tag_groups[0].tags])
                    usas_tags.append(usas_tag)
                except ValueError as e:
                    raise ValueError(f"Invalid USAS tag '{usas_tag_string}' in token: {token} for text: {text}") from e

        return (text, token_texts, usas_tags)

    

    @staticmethod
    def get_mwe_indexes(text: str) -> list[frozenset[int]]:
        """
        Given a text in a format of '<Token>_<USAS-Label><MWE>?' whereby the USAS-Label
        is a combination of alphanumerical characters with zero or more forward slash separations
        e.g. `Z1/O2`, `A1.1.1`, or `A13.3/O2` and the MWE identifier is optional and has
        the following format `[i<ID>.<TOTAL_TOKENS.<TOKEN_INDEX>` whereby the first set of digits
        represents a unique ID, the second set represents the number of tokens
        in the MWE, and the last set represents the token index in the MWE, e.g.
        `[i86.2.1` indicates it is the 86th unique MWE in the text, the MWE has 2
        tokens in total and this is the first token. The return should be a list of
        set values whereby an empty set indicates the token is not part of a MWE
        and a non-empty set contains a unique identifier representing the MWE the
        token is part of.

        Args:
            text: The text to parse, e.g. `Turkish_F2/O4.5[i86.2.1 grind_F2/O4.5[i86.2.2 
                -_- extremely_A13.3 finely_O4.5 ground_A1.1.1 ,_PUNC dust-like_O4.1`
        Returns:
            list[frozenset[int]]: A list of frozenset values where each frozenset
                contains a unique identifier representing the MWE the token is part of
                or an empty set if the token is not part of a MWE.
        Raises:
            ValueError: If a token doesn't contain exactly one underscore separating the token text from the USAS tag/MWE information.
            ValueError: If the MWE format is invalid (e.g., non-numeric values in the MWE identifier).
            ValueError: If a token has multiple MWE assignments (nesting not supported).
            ValueError: If the MWE expected total tokens does not match the number of tokens in the MWE.
            ValueError: If a token has an incomplete MWE format (starts with `[i` but doesn't match the expected pattern).

        Examples:
            >>> get_mwe_indexes("Turkish_F2/O4.5[i86.2.1 grind_F2/O4.5[i86.2.2 
                -_- extremely_A13.3 finely_O4.5 ground_A1.1.1 ,_PUNC dust-like_O4.1")
            [frozenset({1}), frozenset({1}), frozenset({}), frozenset({}), frozenset({}), frozenset({}), frozenset({}), frozenset({}) ]

        """
        if not text.strip():
            return []

        tokens: list[str] = text.split()
        result: list[frozenset[int]] = []
        mwe_data: dict[int, MWEData] = {}  # mwe_id -> {'total_tokens': int, 'token_indices': list[int]}
        token_mwe_mapping: list[int | None] = []  # token_index -> mwe_id (this is either an int or None)
        
        mwe_pattern = re.compile(r'\[i(\d+)\.(\d+)\.(\d+)')

        for token_index, token in enumerate(tokens):
            text_usas_mwe_information = token.split("_")
            if len(text_usas_mwe_information) != 2:
                raise ValueError(f"Invalid token format in text: {text}, expected "
                                 f"a single underscore in token: {token} that separates "
                                 f"the token and the USAS tag / MWE information")
            _, usas_mwe_information = text_usas_mwe_information
            # Find MWE pattern in the usas_mwe_information
            mwe_match = mwe_pattern.search(usas_mwe_information)
            if mwe_match:
                # Validate format - should have exactly 3 groups
                # This should always be the case because of the regex
                if len(mwe_match.groups()) != 3:
                    raise ValueError(f"Invalid MWE format in token: {token}")

                # This should never raise an error because of the regex
                try:
                    mwe_id = int(mwe_match.group(1))
                    total_tokens = int(mwe_match.group(2))
                    _ = int(mwe_match.group(3))
                except ValueError:
                    raise ValueError(f"Invalid MWE format - non-numeric values in: {token} for text: {text}")

                # Check for multiple MWE assignments to same token (nesting not supported)
                if len(mwe_pattern.findall(usas_mwe_information)) > 1:
                    raise ValueError(f"Multiple MWE assignments not supported in token: {token} for text: {text}")

                # Store MWE data
                if mwe_id not in mwe_data:
                    mwe_data[mwe_id] = {
                        'total_tokens': total_tokens,
                        'token_indices': []
                    }

                mwe_data[mwe_id]['token_indices'].append(token_index)
                token_mwe_mapping.append(mwe_id)
            else:
                # Check if token has invalid MWE format (starts with [ but doesn't match pattern)
                if '[i' in usas_mwe_information and not usas_mwe_information.endswith(']'):
                    raise ValueError(f"Invalid MWE format in token: {token} for text: {text}")
                token_mwe_mapping.append(None)

        # Validate MWE consistency
        for mwe_id, data in mwe_data.items():
            actual_token_count = len(data['token_indices'])
            expected_token_count = data['total_tokens']

            if actual_token_count != expected_token_count:
                raise ValueError(
                    f"MWE {mwe_id} has {actual_token_count} tokens but expected {expected_token_count} for text: {text}"
                )

        # Create mapping from MWE IDs to sequential integers
        # (187 -> 1, 188 -> 2, etc.)
        unique_mwe_ids: list[int] = sorted(mwe_data.keys())
        mwe_id_to_int: dict[int, int] = {}
        for idx, mwe_id in enumerate(unique_mwe_ids):
            mwe_id_to_int[mwe_id] = idx + 1

        # Build result
        for mwe_id in token_mwe_mapping:
            if mwe_id is not None:
                result.append(frozenset({mwe_id_to_int[mwe_id]}))
            else:
                result.append(frozenset())

        return result

    @staticmethod
    def parse(dataset_path: Path,
              label_validation: set[str] | None = None,
              label_filter: set[str] | None = None
              ) -> EvaluationDataset:
        """
        Parses the Benedict English corpus into the Evaluation Dataset format, for
        easy evaluation of USAS WSD models.

        All semantic tags / USAS tags that are `,`, `.`, `-`, `!`, or `PUNC` are
        converted to the USAS tag `PUNCT`.

        The label validation does not need to include the `PUNCT` label as this is
        always validated and is expected to be a valid semantic tag.
        
        If the label filter is used then the semantic label for that token will be
        an empty string, this empty string will not be raised as a validation error
        through label validation if this is not None.

        For multi-tag labels, e.g. `F2/O2`, the full multi-tag label must be in
        the label filter for it to be filtered out, e.g. if `F2` is in the label
        filter then it will not affect the multi-tag label `F2/O2` only `F2`
        labels will be filtered out.
        
        Args:
            dataset_path: Path to the Benedict English corpus.
            label_validation: A set of labels that the semantic/dataset labels should
                be validated against. Defaults to `None` in which case no validation
                is performed.
            label_filter: A set of labels from the dataset that should be filtered out.
                Defaults to `None` in which case no filtering is performed.
        Returns:
            EvaluationDataset: The parsed and formatted dataset. The name of the
                dataset is set to `Benedict English` and the text level is set to
                `sentence`. The text returned for this corpus is the original text
                for the given sentence, which includes USAS tags and MWE index information.
        Raises:
            ValueError: If it cannot parse the data due to formatting or a
                label cannot be validated when label validation is used.
        """
        dataset_name = "Benedict English"
        text_level = TextLevel.sentence

        logger.info(f"Parsing the {dataset_name} dataset found at: {dataset_path}")
        
        using_label_validation = True if label_validation is not None else False
        logger.info(f"Using label valdation: {using_label_validation}")
    
        using_label_filtering = True if label_filter is not None else False
        logger.info(f"Using label filtering: {using_label_filtering}")

        evaluation_texts: list[EvaluationTexts] = []

        with dataset_path.open("r", encoding="utf-8") as dataset_fp:
            for line_index, line in enumerate(dataset_fp):
                line = line.strip()
                if not line:
                    continue
                logger.debug(f"Line index: {line_index}")
                
                validated_line, tokens, usas_tags =  EnglishBenedict.validate_text_string_format(line)

                logger.debug(f"Number of tokens in line: {len(tokens)}")

                for token in tokens:
                    token_is_a_tag = True
                    try:
                        parse_usas_token_group(token)
                    except ValueError:
                        token_is_a_tag = False
                    if token_is_a_tag:
                        raise ValueError(
                            f"Error expected token is a tag: {line_index}/{token} (line_index/token): `{line}`"
                        )
                # label filtering
                if label_filter is not None:
                    tmp_usas_tags: list[str] = []
                    for usas_tag in usas_tags:
                        if usas_tag in label_filter:
                            tmp_usas_tags.append("")
                        else:
                            tmp_usas_tags.append(usas_tag)
                    usas_tags = tmp_usas_tags
                
                # label validation
                if label_validation is not None:
                    for usas_tag in usas_tags:
                        match usas_tag:
                            case "PUNCT" | "":
                                continue
                        for usas_sub_tag in usas_tag.split("/"):
                            if usas_sub_tag not in label_validation:
                                raise ValueError(
                                    f"Error semantic tag is not in the label validation: {line_index}-{usas_sub_tag} (line_index/semantic_tag): `{line}`"
                                )

                mwe_indexes = EnglishBenedict.get_mwe_indexes(validated_line)

                evaluation_text = EvaluationTexts(text=validated_line,
                                                  tokens=tokens,
                                                  lemmas=None,
                                                  pos_tags=None,
                                                  semantic_tags=usas_tags,
                                                  mwe_indexes=mwe_indexes)
                evaluation_texts.append(evaluation_text)
                

        logger.info(f"Finished parsing the {dataset_name} dataset")
        return EvaluationDataset(
            name=dataset_name,
            text_level=text_level,
            labels_removed=label_filter,
            texts=evaluation_texts
        )

class FinnishBenedict(BaseParser):
    """
    Parser for the Benedict Finnish corpus that contains human-annotated USAS semantic tags.

    This parser handles the Finnish version of the Benedict corpus, which uses a specific
    token format: '<Token>_<USAS-Label>(_i)?' where:
    - Token: The actual word/token text
    - USAS-Label: Semantic tag(s) in USAS format (e.g., 'F2/O2', 'A3+')
    - i: Optional marker indicating the token is part of a Multi-Word Expression

    This corpus does contain / support MWEs.

    The parser provides comprehensive validation, normalization, and parsing capabilities
    for evaluating USAS Word Sense Disambiguation models.

    The main parsing method is `parse`
    """

    @staticmethod
    def validate_text_string_format(text: str) -> EvaluationTexts:
        """
        Given a text whereby when split by whitespace each text element
        has the following format '<Token>_<USAS-Label>(_i)?' whereby the USAS-Label
        should be validated by `parse_usas_token_group`, the optional `_i`
        indicates if the token is part of a Multi Word Expression.
        The whole text string is validated to ensure it is in this
        '<Token>_<USAS-Label>(_i)?' format.
        It returns the following information within the data structure `EvaluationTexts`:
        * the text string
        * the tokens.
        * the USAS labels.
        * the MWE indexes.

        This validation also handles edge cases with punctuation;
        * If the text element only contains the following it will have the
            USAS label `PUNCT`:
            * `-`
            * `.`
            * `,`
            * `!`
            * `:`
            * `(`
            * `)`
            * `?`
            * `"`

        Args:
            text: The text to validate, e.g. `Vac_F2/O2_i pot_F2/O2_i on_A3+`

        Returns:
            EvaluationTexts: Format validated text string, tokens, USAS labels, and MWE indexes.
        Raises:
            ValueError: If the text string is not valid.
            ValueError: If the USAS label is not valid.
        Examples:
            >>> FinnishBenedict.validate_text_string('Vac_F2/O2_i pot_F2/O2_i on_A3+')
            EvaluationTexts(text='Vac_F2/O2_i pot_F2/O2_i on_A3+',
                            tokens=['Vac', 'pot', 'on'],
                            lemmas=None,
                            pos_tags=None,
                            semantic_tags=['F2/O2', 'F2/O2', 'A3'],
                            mwe_indexes=[frozenset({2})])
        """
        special_punctuation = set({"-", ".", ",", "!", ":", "(", ")", '"', "?"})
        text = text.strip()
        if not text:
            raise ValueError(f"Error the text string is empty: `{text}`")
        all_token_usas_mwe = text.split()
        
        all_tokens: list[str] = []
        all_usas_tags: list[str] = []
        all_mwe_indexes: list[frozenset[int]] = []
        mwe_index = 0
        is_mwe = False
        for token_usas_mwe in all_token_usas_mwe:
            segmented_token_usas_mwe = token_usas_mwe.split("_")
            token = ""
            usas_tag_string = ""
            match len(segmented_token_usas_mwe):
                case 1:
                    token = segmented_token_usas_mwe[0]
                    if token in special_punctuation:
                        usas_tag_string = "PUNCT"
                    else:
                        raise ValueError(
                            f"Error the text string is not valid: `{text}` "
                            f"contains a single token {token} that is not punctuation {special_punctuation}."
                        )
                    is_mwe = False
                case 2:
                    token = segmented_token_usas_mwe[0]
                    usas_tag_string = segmented_token_usas_mwe[1]
                    is_mwe = False
                case 3:
                    token = segmented_token_usas_mwe[0]
                    usas_tag_string = segmented_token_usas_mwe[1]
                    mwe_token = segmented_token_usas_mwe[2]
                    if mwe_token != "i":
                        raise ValueError(
                            f"Error the text string is not valid: `{text}` "
                            f"as the MWE index token should be `i` and not "
                            f"{mwe_token} for the token {token}."
                        )
                    if not is_mwe:
                        mwe_index += 1
                    is_mwe = True
                case _:
                    raise ValueError(
                            f"Error the text string is not valid: `{text}` "
                            f"as the token {token} contains more than two underscores."
                        )

            if token.strip() == "":
                raise ValueError(
                    f"Error the text string is not valid: `{text}` "
                    f"as the token {token} is empty."
                )
            try:
                usas_tag = usas_tag_string
                if usas_tag_string != "PUNCT":
                    usas_tag_groups = parse_usas_token_group(usas_tag_string)
                    # We only want the first USAS tag group as there should only be one USAS tag
                    # group by definition. We then want each tag from that first group.
                    # There can be multiple tags because of multi tag membership, e.g.
                    # F2/O2
                    usas_tag = "/".join([usas_tag.tag for usas_tag in usas_tag_groups[0].tags])
                all_usas_tags.append(usas_tag)
            except ValueError as e:
                raise ValueError(f"Invalid USAS tag '{usas_tag_string}' in token: {token} for text: {text}") from e
            
            all_tokens.append(token)
            if is_mwe:
                all_mwe_indexes.append(frozenset({mwe_index}))
            else:
                all_mwe_indexes.append(frozenset({}))

        return EvaluationTexts(
            text=text,
            tokens=all_tokens,
            lemmas=None,
            pos_tags=None,
            semantic_tags=all_usas_tags,
            mwe_indexes=all_mwe_indexes
        )


    @staticmethod
    def parse(dataset_path: Path,
              label_validation: set[str] | None = None,
              label_filter: set[str] | None = None) -> EvaluationDataset:
        """
        Parses the Benedict Finnish corpus into the Evaluation Dataset format, for
        easy evaluation of USAS WSD models.

        All tokens that are "-", ".", ",", "!", ":", "(", ")", '"', "?" will be
        kept as the tokens and given the USAS tag `PUNCT`.

        The label validation does not need to include the `PUNCT` label as this is
        always validated and is expected to be a valid semantic tag.
        
        If the label filter is used then the semantic label for that token will be
        an empty string, this empty string will not be raised as a validation error
        through label validation if this is not None.

        For multi-tag labels, e.g. `F2/O2`, the full multi-tag label must be in
        the label filter for it to be filtered out, e.g. if `F2` is in the label
        filter then it will not affect the multi-tag label `F2/O2` only `F2`
        labels will be filtered out.
        
        Args:
            dataset_path: Path to the Benedict Finnish corpus.
            label_validation: A set of labels that the semantic/dataset labels should
                be validated against. Defaults to `None` in which case no validation
                is performed.
            label_filter: A set of labels from the dataset that should be filtered out.
                Defaults to `None` in which case no filtering is performed.
        Returns:
            EvaluationDataset: The parsed and formatted dataset. The name of the
                dataset is set to `Benedict Finnish` and the text level is set to
                `sentence`. The text returned for this corpus is the original text
                for the given sentence, which includes USAS tags and MWE index information.
        Raises:
            ValueError: If it cannot parse the data due to formatting or a
                label cannot be validated when label validation is used.
        """
        dataset_name = "Benedict Finnish"
        text_level = TextLevel.sentence

        logger.info(f"Parsing the {dataset_name} dataset found at: {dataset_path}")
        
        using_label_validation = True if label_validation is not None else False
        logger.info(f"Using label valdation: {using_label_validation}")
    
        using_label_filtering = True if label_filter is not None else False
        logger.info(f"Using label filtering: {using_label_filtering}")

        evaluation_texts: list[EvaluationTexts] = []

        with dataset_path.open("r", encoding="utf-8") as dataset_fp:
            for line_index, line in enumerate(dataset_fp):
                line = line.strip()
                if not line:
                    continue
                logger.debug(f"Line index: {line_index}")
                
                validated_evaluation_text =  FinnishBenedict.validate_text_string_format(line)
                tokens = validated_evaluation_text.tokens

                logger.debug(f"Number of tokens in line: {len(tokens)}")

                tmp_usas_tags: list[str] = []
                usas_tags = validated_evaluation_text.semantic_tags
                assert isinstance(usas_tags, list)
                for index, token in enumerate(tokens):
                    usas_tag = usas_tags[index]

                    # Validate token is not a USAS tag
                    token_is_a_tag = True
                    try:
                        parse_usas_token_group(token)
                    except ValueError:
                        token_is_a_tag = False
                    if token_is_a_tag:
                        raise ValueError(
                            f"Error expected token is a tag: {line_index}/{token} (line_index/token): `{line}`"
                        )
                    
                    # label filtering
                    if label_filter is not None and usas_tag in label_filter:
                        usas_tag = ""
                    
                    # label validation
                    if label_validation is not None:
                        match usas_tag:
                            case "PUNCT" | "":
                                pass
                            case _:
                                for usas_sub_tag in usas_tag.split("/"):
                                    if usas_sub_tag not in label_validation:
                                        raise ValueError(
                                            f"Error semantic tag is not in the label validation: {line_index}-{usas_sub_tag}"
                                            f" (line_index/semantic_tag): `{line}`"
                                        )
                    tmp_usas_tags.append(usas_tag)
                
                

                evaluation_text = EvaluationTexts(text=validated_evaluation_text.text,
                                                  tokens=tokens,
                                                  lemmas=None,
                                                  pos_tags=None,
                                                  semantic_tags=tmp_usas_tags,
                                                  mwe_indexes=validated_evaluation_text.mwe_indexes)
                evaluation_texts.append(evaluation_text)
                

        logger.info(f"Finished parsing the {dataset_name} dataset")
        return EvaluationDataset(
            name=dataset_name,
            text_level=text_level,
            labels_removed=label_filter,
            texts=evaluation_texts
        )