import logging
from pathlib import Path

from usas_evaluation_framework import data_utils
from usas_evaluation_framework.dataset import (
    EvaluationDataset,
    EvaluationTexts,
    TextLevel,
)
from usas_evaluation_framework.parsers.base import BaseParser

logger = logging.getLogger(__name__)

class CorcenccParser(BaseParser):
    """
    Parser for the CorCenCC corpus that contains human-annotated USAS semantic tags.

    The CorCenCC corpus does not contain MWEs.

    The corpus should contain a sentence on each new line. Each sentence should
    contain token data that is separated by a space, in which the token data
    should be in the following format:
    {Token}|{Lemma}|{Core POS}|{True CorCenCC Basic POS}|{Predicted CorCenCC Enriched POS}|{Predicted CorCenCC Basic POS}|{USAS Tag}

    Example token data:

    `A|a|pron|Rha|Rhaperth|Rha|Z5`

    Token - The annotated token.
    Lemma - The predicted lemma. The prediction was made from the CyTag tagger.
    Core POS - Mapping from the True CorCenCC Basic POS tag to the Core POS tag. This mapping is based off table A.1 in Leveraging Pre-Trained Embeddings for Welsh Taggers.
    True CorCenCC Basic POS - Human annotated CorCenCC Basic POS tag.
    Predicted CorCenCC Enriched POS - this has come from running the CyTag tagger. As this tag has been predicted it may be different to the True CorCenCC Basic POS.
    Predicted CorCenCC Basic POS - this has come from running the CyTag tagger. As this tag has been predicted it may be different to the True CorCenCC Basic POS.
    USAS Tag - The USAS tag/label.

    The main parsing method is `parse`
    """

    @staticmethod
    def parse(dataset_path: Path,
              label_validation: set[str] | None = None,
              label_filter: set[str] | None = None,
              ) -> EvaluationDataset:
        """
        Parses the CorCenCC corpus into the Evaluation Dataset format, for
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

        NOTE: when parsing the CorCenCC corpus many rules are used to create a USAS
        validated dataset. Some of the token semantic tags are removed as they
        are not valid tags and are replaced with an empty string or if possible
        with a valid tag.

        NOTE: we do not parse the lemma or POS tag information, this is a TODO for
        future work.
        
        Args:
            dataset_path: Path to the CorCenCC corpus, which should be in txt format.
            label_validation: A set of labels that the semantic/dataset labels should
                be validated against. Defaults to `None` in which case no validation
                is performed. NOTE: label validation is not performed on USAS tags
                that will not be returned, i.e. all USAS tags after the first `;`.
            label_filter: A set of labels from the dataset that should be filtered out.
                Defaults to `None` in which case no filtering is performed.
        Returns:
            EvaluationDataset: The parsed and formatted dataset. The name of the
                dataset is set to `Corcencc` and the text level is set to
                `sentence`. The text returned for this corpus is just the tokens
                joined together by a single space for each sentence.
        Raises:
            ValueError: If it cannot parse the data due to formatting or a
                label cannot be validated when label validation is used.
        """
        def validate_label(label: str) -> str:
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
            if token == "S4C":
                return
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

        dataset_name = "Corcencc"
        text_level = TextLevel.sentence

        logger.info(f"Parsing the {dataset_name} dataset found at: {dataset_path}")
        
        using_label_validation = True if label_validation is not None else False
        logger.info(f"Using label valdation: {using_label_validation}")
    
        using_label_filtering = True if label_filter is not None else False
        logger.info(f"Using label filtering: {using_label_filtering}")

        evaluation_texts: list[EvaluationTexts] = []

        with dataset_path.open("r", encoding="utf-8") as fp:
            
            for line_index, line in enumerate(fp):
                line = line.strip()
                if not line:
                    continue
                
                tokens: list[str] = []
                semantic_tags: list[str] = []
                for token_index, token_entry in enumerate(line.split()):
                    token_data = token_entry.split("|")
                    if len(token_data) != 7:
                        raise ValueError(
                            "CorCenCC data is not in the expected format, expected "
                            f"7 columns but found {len(token_data)} "
                            f"columns. Line: {line_index}: {line}"
                        )

                    token = token_data[0].strip()
                    usas_label = token_data[6].strip()

                    skip_validation = False

                    match (usas_label, line_index, token_index, token):
                        case ("I3", 3, 10, "gweithio") | ("I3", 18, 17, "waith") | ("I3", 37, 16, "gweithio") | ("I3", 38, 18, "gwaith") | ("I3", 57, 5, "gwaith") | ("I3", 62, 25, "swyddi") | ("I3", 101, 40, "swyddi") | ("I3", 133, 12, "weithio") | ("I3", 225, 7, "waith") | ("I3", 227, 8, "waith") | ("I3", 259, 21, "ddyletswyddau") | ("I3", 286, 24, "gwaith") | ("I3", 295, 4, "waith") | ("I3", 335, 23, "waith") | ("I3", 336, 0, "Gwaith") | ("I3", 337, 2, "gwaith") | ("I3", 341, 20, "waith") | ("I3", 351, 4, "gwaith") | ("I3", 397, 15, "weithiodd"):
                            usas_label = "I3.1"
                        case ("A1", 112, 15, "broses") | ("A1", 191, 15, "broses") | ("A.1.1.1", 578, 3, "peiriant"):
                            usas_label = "A1.1.1"
                        case ("A.1.5.1", 147, 9, "ddefnydd") | ("A.1.5.1", 159, 25, "ddefnyddio"):
                            usas_label = "A1.5.1"
                        case ("T.1.1.1/S2/P1", 248, 20, "cyn-fyfyrwyr"):
                            usas_label = "T1.1.1/S2/P1"
                        case ("T.1.1.2", 326, 24, "eleni") | ("T.1.1.2", 329, 4, "eleni") | ("T.1.1.2", 333, 1, "eleni") | ("T.1.1.2", 391, 43, "gyfoes"):
                            usas_label = "T1.1.2"
                        case ("A4", 345, 12, "nodweddu"):
                            usas_label = "A4.1"
                        case ("Q2/T.1.1.1", 381, 0, "Geirdarddiad"):
                            usas_label = "Q2/T1.1.1"
                        case ("Q2.2/S.1.2.4", 511, 5, "croesawyd"):
                            usas_label = "Q2.2/S1.2.4"
                        case ("Q.21", 512, 0, "Dywedodd"):
                            usas_label = "Q2.1"

                    match usas_label:
                        case "I3":
                            match (line_index, token_index, token):
                                case (19, 2, "swyddogaethau") | (19, 20, "swyddogaethau")  | (26, 9, "rôl")  | (67, 37, "gweithle")  | (96, 1, "rôl") | (270, 7, "swyddogaeth") | (295, 9, "yrfa") | (295, 17, "gweithiodd") | (304, 18, "gyrfa") | (460, 2, "gyrfa"):
                                    skip_validation = True
                        case "I3/S7":
                            match (line_index, token_index, token):
                                case (74, 3, "ddyletswydd") | (21, 15, "ddyletswydd"):
                                    skip_validation = True
                        case "N5.1/I3":
                            match (line_index, token_index, token):
                                case (23, 8, "adran") | (27, 15, "adran") | (35, 2, "adran") | (40, 7, "adran") | (43, 8, "adran") | (484, 8, "Adran") | (491, 8, "Adran"):
                                    skip_validation = True
                        case "!ERR":
                            match (line_index, token_index, token):
                                case (23, 37, "welliannau") | (29, 16, "gwelliannau") | (63, 18, "gwelliannau") | (486, 29, "Newyddion"):
                                    skip_validation = True
                        case "A11":
                            match (line_index, token_index, token):
                                case (46, 26, "allweddol") | (80, 8, "hollbwysig") | (131, 5, "hollbwysig") | (195, 15, "brif") | (196, 13, "gwerthfawr") | (406, 26, "enwocaf") | (434, 29, "statws") | (448, 24, "statws") | (453, 15, "allweddol") | (470, 25, "hollbwysig") | (476, 10, "bennaf") | (484, 4, "bennaf") | (502, 10, "seiliedig"):
                                    skip_validation = True
                        case "I3/S2mf":
                            match (line_index, token_index, token):
                                case (86, 15, "gweithredwyr"):
                                    skip_validation = True
                        case "I3/S7.1":
                            match (line_index, token_index, token):
                                case (486, 6, "ddyletswydd"):
                                    skip_validation = True
                        case "S7/X6":
                            match (line_index, token_index, token):
                                case (94, 4, "benodir"):
                                    skip_validation = True
                        case "A11/A10":
                            match (line_index, token_index, token):
                                case (104, 14, "swyddogol") | (161, 17, "swyddogol"):
                                    skip_validation = True
                        case "X5-":
                            match (line_index, token_index, token):
                                case (115, 17, "frasamcanu"):
                                    skip_validation = True
                        case "Q1/Y2":
                            match (line_index, token_index, token):
                                case (137, 25, "negeseuon"):
                                    skip_validation = True
                        case "A4":
                            match (line_index, token_index, token):
                                case (151, 3, "math") | (217, 4, "thema") | (223, 29, "Themâu") | (223, 45, "themâu") | (273, 3, "teipoleg") | (366, 19, "fath"):
                                    skip_validation = True
                        case "S7.1+/S.1F":
                            match (line_index, token_index, token):
                                case (394, 16, "frenhiniaeth") | (437, 9, "frenhines"):
                                    skip_validation = True
                        case "A1":
                            match (line_index, token_index, token):
                                case (83, 18, "gyffredinol") | (253, 17, "cyffredinol") | (514, 19, "gyffredinol"):
                                    skip_validation = True
                        case "A11/A4.2/S7.1":
                            match (line_index, token_index, token):
                                case (237, 22, "brif") | (242, 8, "brif"):
                                    skip_validation = True
                        case "A4.2/A11":
                            match (line_index, token_index, token):
                                case (248, 9, "arbennig") | (264, 21, "arbennig") | (282, 5, "Arbennig") | (299, 3, "arbennig"):
                                    skip_validation = True
                        case "A11/S7.1":
                            match (line_index, token_index, token):
                                case (252, 5, "prif") | (253, 20, "prif") | (281, 22, "swyddogol") | (311, 12, "prif") | (311, 22, "prif"):
                                    skip_validation = True
                        case "T.13":
                            match (line_index, token_index, token):
                                case (255, 14, "hyd") | (308, 8, "dal"):
                                    skip_validation = True
                        case "N.37":
                            match (line_index, token_index, token):
                                case (263, 7, "isaf"):
                                    skip_validation = True
                        case "S7.1/A11/A14":
                            match (line_index, token_index, token):
                                case (270, 6, "prif"):
                                    skip_validation = True
                        case "A14/A11":
                            match (line_index, token_index, token):
                                case (293, 22, "prif"):
                                    skip_validation = True
                        case "S2/I3/Q4":
                            match (line_index, token_index, token):
                                case (301, 10, "gyflwynydd"):
                                    skip_validation = True
                        case "A1.8/A11":
                            match (line_index, token_index, token):
                                case (309, 14, "eicon"):
                                    skip_validation = True
                        case "A11/A2.1":
                            match (line_index, token_index, token):
                                case (315, 35, "drobwynt"):
                                    skip_validation = True
                        case "I3/S5":
                            match (line_index, token_index, token):
                                case (334, 18, "gydweithio"):
                                    skip_validation = True
                        case "E4-/S5-":
                            match (line_index, token_index, token):
                                case (363, 9, "unigrwydd") | (365, 28, "unigrwydd"):
                                    skip_validation = True
                        case "A1.1.1/E4-":
                            match (line_index, token_index, token):
                                case (415, 3, "penyd"):
                                    skip_validation = True
                        case "I3/W3":
                            match (line_index, token_index, token):
                                case (420, 24, "chwareli"):
                                    skip_validation = True
                        case "E4-/I1-/G1.2-":
                            match (line_index, token_index, token):
                                case (424, 8, "Dirwasgiad"):
                                    skip_validation = True
                        case "Q2.2/S7.1/A11+":
                            match (line_index, token_index, token):
                                case (444, 7, "seremonïol"):
                                    skip_validation = True
                        case "S1.1.3/Q.2":
                            match (line_index, token_index, token):
                                case (482, 19, "gyfarfod") | (514, 1, "cyfarfod"):
                                    skip_validation = True
                        case "Q2/X4":
                            match (line_index, token_index, token):
                                case (495, 49, "sail"):
                                    skip_validation = True
                        case "A10/A11/Q2":
                            match (line_index, token_index, token):
                                case (497, 23, "swyddogol") | (499, 5, "swyddogol"):
                                    skip_validation = True
                        case "H1/I3":
                            match (line_index, token_index, token):
                                case (513, 8, "Swyddfa")  | (519, 4, "Swyddfa"):
                                    skip_validation = True
                        
                        

                    if not skip_validation:
                        #print(usas_label, line_index, token_index, token)
                        try:
                            usas_label = validate_label(usas_label)
                        except ValueError as e:
                            raise ValueError(
                                f"Error parsing the {dataset_name} dataset at for line: {line_index}: {line}"
                            ) from e
                    else:
                        usas_label = ""

                    try:
                        validate_token(token)
                    except ValueError as e:
                        raise ValueError(
                                f"Error parsing the {dataset_name} dataset at for line: {line_index}: {line}"
                            ) from e

                    tokens.append(token)
                    semantic_tags.append(usas_label)

                evaluation_text = EvaluationTexts(text=" ".join(tokens),
                                                tokens=tokens,
                                                lemmas=None,
                                                pos_tags=None,
                                                semantic_tags=semantic_tags,
                                                mwe_indexes=[frozenset({})] * len(tokens))

                evaluation_texts.append(evaluation_text)


        logger.info(f"Finished parsing the {dataset_name} dataset")
        return EvaluationDataset(
            name=dataset_name,
            text_level=text_level,
            labels_removed=label_filter,
            texts=evaluation_texts
        )
                
