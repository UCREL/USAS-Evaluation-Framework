import logging
import re
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

TAG_RE = re.compile(r"^[A-Z](\d+)((\.\d+)+)?")
PUNCT_RE = re.compile(r"^PUNCT")
POSITIVE_MARKERS_RE = re.compile(r"\++")
NEGATIVE_MARKERS_RE = re.compile(r"\-+")


# A regular expression that was used to capture edge cases in the original
# C version of the USAS tagger:
# ALT_TAG_RE = re.compile(r"^[a-z](\d+)((\.\d+)+)?")

class USASTag(BaseModel):
    """
    Represents all of the properties associated with a USAS tag.

    Attributes:
        tag: The USAS tag.
        number_positive_markers: The number of positive markers.
        number_negative_markers: The number of negative markers.
        rarity_marker_1: True if the USAS tag contains the rarity marker %.
        rarity_marker_2: True if the USAS tag contains the rarity marker @.
        female: True if the USAS tag contains the female marker denoted by `f`.
        male: True if the USAS tag contains the male marker denoted by `m`.
        antecedents: True if the USAS tag contains the antecedents marker denoted by `c`.
        neuter: True if the USAS tag contains the neuter marker denoted by `n`.
        idiom: False, currently not supported and therefore is always False.
    """

    tag: str = Field(title="USAS Tag", description="USAS Tag", examples=["A1.1.1"])
    number_positive_markers: int = Field(
        0,
        title="Positive Markers",
        description="Number of positive markers.",
        examples=[0, 1, 2, 3],
    )
    number_negative_markers: int = Field(
        0,
        title="Negative Markers",
        description="Number of negative markers.",
        examples=[0, 1, 2, 3],
    )
    rarity_marker_1: bool = Field(
        False, title="Rare Marker 1", description="Rarity marker 1 indicated by %"
    )
    rarity_marker_2: bool = Field(
        False, title="Rare Marker 2", description="Rarity marker 2 indicated by @"
    )
    female: bool = Field(False, title="Female", description="Female")
    male: bool = Field(False, title="Male", description="Male")
    antecedents: bool = Field(
        False,
        title="Antecedents",
        description="Potential antecedents of conceptual anaphors (neutral for number)",
    )
    neuter: bool = Field(False, title="Neuter", description="Neuter")
    idiom: Literal[False] = Field(False, title="Idiom",
                                  description="Is it an idiom, currently not supported and is always False.")


class USASTagGroup(BaseModel):
    """
    Represents a grouping of one or more USAS tags that are associated to a
    token.

    Attributes:
        tags: A list of USAS tags that are associated to a token. This grouping
            of USAS tags is a way of representing multi tag membership.
    """

    _tags_description = (
        "A grouping of one or more USAS tags whereby if more "
        "than one exists then the word is an equal member of "
        "all semantic tags/categories"
    )
    _tags_examples = [
        [USASTag(tag="A1.1.1")],
        [
            USASTag(tag="E2", number_negative_markers=1),
            USASTag(tag="S7.1", number_positive_markers=1),
        ],
    ]
    tags: list[USASTag] = Field(
        title="USAS Tags", description=_tags_description, examples=_tags_examples
    )


def get_all_mwe_token_indexes(mwe_index_slices: list[tuple[int, int]]) -> frozenset[int]:
    """
    Given a list of tuples that represent the start and end indexes of a
    Multi Word Expression (MWE), it returns a frozenset of all the token indexes
    that are part of the MWE. If the MWE is a single token then it returns a
    frozenset of length 1 which is start index.

    Args:
        mwe_index_slices: A list of tuples that represent the start and end indexes of a
            Multi Word Expression (MWE).

    Returns:
        A frozenset of all the token indexes that are part of the MWE, even if
            the MWE is a single token.
    """
    all_mwe_token_indexes = set()
    for mwe_index_range in mwe_index_slices:
        all_mwe_token_indexes.update(range(*mwe_index_range))
    return frozenset(all_mwe_token_indexes)


def parse_usas_token_group(usas_tag_group_text: str) -> list[USASTagGroup]:
    r"""
    Given a the string that represents the USAS tags whereby each USAS tag is
    separated by whitespace it is converted into a structured format.

    This whitespace separation of USAS tags is the format that is produced by the
    original C version of the USAS tagger when it outputs USAS tags for a given
    token or meaningful word unit like a Multi Word Expression (MWE).

    The whitespace separation can be one or more spaces, i.e. `    ` or ` `

    A USAS tag can be also be `PUNCT` which represents punctuation.

    Complex examples of `usas_tag_group_text`:
    `L1 E3- O4.2- X5.2+ A6.2- A1.7- A7- W3 L2 F1 S1.2.4- Z2 Z2/S2mf Z3 O4.3 G1.2 G1.2/S2mf`

    Args:
        usas_tag_group_text: The string that represents the USAS tags
            produced by the USAS tagger for one token.
    Returns:
        list[USASTagGroup]: structured format of the USAS tags.
    Raises:
        ValueError: If the USAS tags within the given text cannot be parsed
            as a USAS tag, whereby each USAS tag after whitespace and `/` split
            should match the following regex: `[A-Z](\d+)((\.\d+)+)?` or `PUNCT`.
    """

    def parse_usas_tag(usas_tag_text: str) -> USASTag:
        r"""
        Given a single USAS tag text, e.g. `X5.2+` it is converted into
        a structured format.

        Note: a USAS tag text should not contain a `/`,
        e.g. `G1.2/S2mf` as this contains two USAS tags that represent
        a combined semantic meaning of a token.

        Args:
            usas_tag_text: Single USAS tag text
        Returns:
            USASTag: A structured format of the USAS tag.
        Raises:
            ValueError: If it cannot match the given text with the USAS tag
                regex, which is `[A-Z](\d+)((\.\d+)+)?` or `PUNCT`.
        """

        tag_match = TAG_RE.match(usas_tag_text)
        punct_match = PUNCT_RE.match(usas_tag_text)
        tag = ""

        if tag_match:
            tag = tag_match.group()
            usas_tag_text = TAG_RE.sub("", usas_tag_text)
        elif punct_match:
            tag = punct_match.group()
            usas_tag_text = PUNCT_RE.sub("", usas_tag_text)
        else:
            raise ValueError(
                f"Cannot find the tag for this USAS tag text: {usas_tag_text}"
            )

        number_positive_markers = 0
        positive_marker_match = POSITIVE_MARKERS_RE.search(usas_tag_text)
        if positive_marker_match:
            number_positive_markers = len(positive_marker_match.group())
            usas_tag_text = POSITIVE_MARKERS_RE.sub("", usas_tag_text)

        number_negative_markers = 0
        negative_marker_match = NEGATIVE_MARKERS_RE.search(usas_tag_text)
        if negative_marker_match:
            number_negative_markers = len(negative_marker_match.group())
            usas_tag_text = NEGATIVE_MARKERS_RE.sub("", usas_tag_text)

        is_male = False
        if "m" in usas_tag_text:
            is_male = True

        is_female = False
        if "f" in usas_tag_text:
            is_female = True

        contain_rare_marker_1 = False
        if "%" in usas_tag_text:
            contain_rare_marker_1 = True

        contain_rare_marker_2 = False
        if "@" in usas_tag_text:
            contain_rare_marker_2 = True

        contains_antecedent = False
        if "c" in usas_tag_text:
            contains_antecedent = True

        contains_neuter = False
        if "n" in usas_tag_text:
            contains_neuter = True

        # Currently do not support finding idioms
        is_idiom = False

        return USASTag(
            tag=tag,
            male=is_male,
            female=is_female,
            rarity_marker_1=contain_rare_marker_1,
            rarity_marker_2=contain_rare_marker_2,
            number_positive_markers=number_positive_markers,
            number_negative_markers=number_negative_markers,
            antecedents=contains_antecedent,
            neuter=contains_neuter,
            idiom=is_idiom,
        )

    token_usas_tags: list[USASTagGroup] = []

    for usas_tag_group in re.findall(r"\S+", usas_tag_group_text):
        usas_tags: list[USASTag] = []
        for usas_tag_text in usas_tag_group.split("/"):
            usas_tags.append(parse_usas_tag(usas_tag_text))
        token_usas_tags.append(USASTagGroup(tags=usas_tags))

    return token_usas_tags


def load_usas_mapper(usas_tag_descriptions_file: Path | None,
                     tags_to_filter_out: set[str] | None
                     ) -> dict[str, str]:
    """
    Returns a dictionary of USAS tags and their descriptions.

    Args:
        usas_tag_descriptions_file: The path to the YAML file that
            contains the USAS tags and their descriptions. If None then the
            function will use the USAS tags and description file that is located
            within the package at `usas_csv_auto_labeling/data/usas/usas_mapper.yaml`.
        tags_to_filter_out: A set of USAS tags to filter out.

    Returns:
        dict[str, str]: A dictionary of USAS tags and their descriptions.
    
    Raises:
        FileNotFoundError: If the `usas_tag_descriptions_file` is not found.
        ValueError: If the `usas_tag_descriptions_file` is not a file.
    """

    def _get_usas_tag_descriptions(usas_tag_name: str,
                                   usas_tag_dict: dict[str, Any],
                                   collected_tag_descriptions: dict[str, str]
                                   ) -> dict[str, str]:
        """
        A recursive function that loops through the `usas_tag_dict` and returns a
        dictionary of the USAS tag and as a value it's description. Each USAS tag
        that is found is added to the `collected_tag_descriptions` dictionary.
        Once all the USAS tags are found, the `collected_tag_descriptions` is
        returned.

        The description is made of the USAS tag title and description in the
        following format: `title: <title> description: <description>`

        Args:
            usas_tag_name: The name of the USAS tag.
            usas_tag_dict: A dictionary containing the raw
                USAS tag data that is read from the YAML file.
            collected_tag_descriptions: A dictionary of all
                USAS tags and their descriptions.

        Returns:
            dict[str, str]: A dictionary of USAS tags and their descriptions.
        """
        if "title" in usas_tag_dict and "description" in usas_tag_dict:
            title_description = f"title: {usas_tag_dict['title']} description: {usas_tag_dict['description']}"
            if usas_tag_name in collected_tag_descriptions:
                raise KeyError(f"Duplicate usas tag name found: {usas_tag_name} "
                               "when reading the following data: "
                               f"{usas_tag_dict}, currently found usas tags: "
                               f"{collected_tag_descriptions}")
            collected_tag_descriptions[usas_tag_name] = title_description.strip()
        elif "title" in usas_tag_dict:
            raise KeyError("No description key found when it is expected for: "
                           f"{usas_tag_name} {usas_tag_dict}")
        elif "description" in usas_tag_dict:
            raise KeyError("No title key found when it is expected for: "
                           f"{usas_tag_name} {usas_tag_dict}")

        keys_to_ignore = set(["title", "description"])
        for child_usas_tag_name, child_usas_tag_dict in usas_tag_dict.items():
            if child_usas_tag_name not in keys_to_ignore:
                collected_tag_descriptions = _get_usas_tag_descriptions(child_usas_tag_name,
                                                                          child_usas_tag_dict,
                                                                                        collected_tag_descriptions)
        return collected_tag_descriptions

    usas_tag_descriptions_file_path: Path = Path()
    if usas_tag_descriptions_file is None:
        usas_tag_descriptions_file_str = str(files("usas_evaluation_framework").joinpath("data/usas/usas_mapper.yaml"))
        usas_tag_descriptions_file_path = Path(usas_tag_descriptions_file_str)
    else:
        usas_tag_descriptions_file_path = usas_tag_descriptions_file

    if usas_tag_descriptions_file_path.exists() is False:
        raise FileNotFoundError(f"USAS tag descriptions file not found at: "
                                f"{usas_tag_descriptions_file_path}")
    elif usas_tag_descriptions_file_path.is_file() is False:
        raise ValueError(f"USAS tag descriptions file is not a file: "
                         f"{usas_tag_descriptions_file_path}")

    usas_mapping: dict[str, str] = {}
    with usas_tag_descriptions_file_path.open("r") as usas_mapper_fp:
        usas_mapping_data = usas_mapper_fp.read()
        for high_level_usas_tag, high_level_usas_tag_dict in yaml.safe_load(usas_mapping_data).items():
            usas_mapping = _get_usas_tag_descriptions(high_level_usas_tag,
                                                      high_level_usas_tag_dict,
                                                      usas_mapping)
    if tags_to_filter_out:
        tmp_usas_mapping = {}
        for key, value in usas_mapping.items():
            if key in tags_to_filter_out:
                continue
            tmp_usas_mapping[key] = value
        usas_mapping = tmp_usas_mapping
    return usas_mapping