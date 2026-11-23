"""
Upload parsed USAS evaluation corpora to one Hugging Face Hub dataset repository.

Each corpus is parsed with its parser from this library and pushed as its own
subset (config) of the repository, named ``<language>_<corpus>``, for example
``english_benedict`` or ``chinese_torch``. Every subset shares the same schema,
one row per text:

``text`` (string), ``tokens`` (list of strings), ``lemmas`` and ``pos_tags``
(list of strings or null), ``semantic_tags`` (list of lists of strings, one
list per token) and ``mwe_indexes`` (list of lists of integers, one list per
token, empty when the token is not part of a multi word expression).

After the subsets are pushed, a dataset card (``README.md``) is generated and
pushed. It explains how each subset was parsed, how to load it back into this
library and where the raw data can be found. The card's prose is in
``dataset_card_template.md`` next to this script; its ``$name`` placeholders
are filled in with :class:`string.Template`, so a literal ``$`` in the template
must be written ``$$``.

Corpora are given as ``--dataset SUBSET=PATH`` options, or ``--all`` uploads
every subset from its default raw data file in this repository.

The Hugging Face token is read from the ``HF_TOKEN`` environment variable,
which can be set in a ``.env`` file in the current directory.

The NAACL 2015 Chinese, NAACL 2015 Portuguese and ICC Irish corpora must not be
redistributed. They are not uploadable subsets, and the script refuses any
input path that points at them. The dataset card lists them as available on
request.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Annotated, Any, Literal, TypeAlias, cast

import typer
from datasets import Dataset, Features, List, Value
from datasets.info import DatasetInfosDict
from datasets.splits import SplitDict, SplitInfo
from datasets.utils.metadata import MetadataConfigs
from dotenv import load_dotenv
from huggingface_hub import DatasetCard, DatasetCardData, HfApi
from huggingface_hub.errors import (
    HfHubHTTPError,
    LocalTokenNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
)
from requests.exceptions import HTTPError
from usas_validator.utils import load_usas_mapper

from usas_evaluation_framework.dataset import EvaluationDataset, TextLevel
from usas_evaluation_framework.parsers.base import BaseParser
from usas_evaluation_framework.parsers.benedict import EnglishBenedict, FinnishBenedict
from usas_evaluation_framework.parsers.corcencc import CorcenccParser
from usas_evaluation_framework.parsers.danish_wikipedia import DanishWikipediaUSAS
from usas_evaluation_framework.parsers.dutch_wikipedia import DutchWikipediaUSAS
from usas_evaluation_framework.parsers.english_wikipedia import EnglishWikipediaUSAS
from usas_evaluation_framework.parsers.hindi_wikipedia import HindiWikipediaUSAS
from usas_evaluation_framework.parsers.igbo_wikipedia import IgboWikipediaUSAS
from usas_evaluation_framework.parsers.naacl_2015 import NAACL2015USAS
from usas_evaluation_framework.parsers.spanish_wikipedia import SpanishWikipediaUSAS
from usas_evaluation_framework.parsers.torch import TorchParser

GITHUB_URL = "https://github.com/UCREL/USAS-Evaluation-Framework"
GITHUB_BLOB_URL = f"{GITHUB_URL}/blob/main"
PARSERS_DOCS_URL = "https://ucrel.github.io/USAS-Evaluation-Framework/api/parsers.html"
CARD_TEMPLATE = Path(__file__).with_name("dataset_card_template.md")
# The repository root, which each corpus's default raw data path is relative to.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Hugging Face size categories, each with its exclusive upper bound on the
# number of rows.
SIZE_CATEGORIES: tuple[tuple[int, str], ...] = (
    (1_000, "n<1K"),
    (10_000, "1K<n<10K"),
    (100_000, "10K<n<100K"),
    (1_000_000, "100K<n<1M"),
    (10_000_000, "1M<n<10M"),
    (100_000_000, "10M<n<100M"),
    (1_000_000_000, "100M<n<1B"),
)

# Tags filtered out when computing the dataset card statistics, as in
# notebooks/dataset_statistics.ipynb. The uploaded data is not filtered.
STATS_LABEL_FILTER = frozenset({"Z9", "Z99"})

CC_BY_NC_SA_4 = "cc-by-nc-sa-4.0"
CC_BY_SA_4 = "cc-by-sa-4.0"
CC_BY_NC_SA_3 = "cc-by-nc-sa-3.0"
CC_BY_SA_3 = "cc-by-sa-3.0"

WIKIPEDIA_NOTES = (
    "Medical Wikipedia articles. The gold tag is the `corrected USAS` column, or the "
    "`predicted USAS` tag when the annotator left `corrected USAS` empty. Every token, "
    "punctuation included, has a human checked tag (`Z9` for punctuation). MWEs come "
    "from the `corrected MWE` column. Lemmas and POS tags are included."
)
IGBO_WIKIPEDIA_NOTES = (
    WIKIPEDIA_NOTES.removesuffix(" Lemmas and POS tags are included.")
    + " The corpus has no lemmas or POS tags; every lemma and POS tag is the placeholder `-`."
)


@dataclass(frozen=True)
class CorpusSpec:
    """
    How to parse and describe one corpus that may be uploaded as a subset.

    Attributes:
        parser: The parser class used to parse the corpus.
        language: The language of the corpus, e.g. ``'English'``.
        language_code: The ISO 639-1 code of the language, for the card metadata.
        corpus: The display name of the corpus.
        text_level: The text level the parser produces, shown on the card even
            for subsets that were not parsed in this run.
        licences: The Hugging Face licence identifiers that apply to the corpus.
        raw_data: Where the raw data can be found, relative to the repository root.
            It is also the default input file used by ``--all``.
        parser_source: The parser's source file, relative to the GitHub repository root.
        parsing_notes: Markdown notes on how the raw data was parsed.
        parse_kwargs: Extra keyword arguments passed to the parser's ``parse`` method.
    """
    parser: type[BaseParser]
    language: str
    language_code: str
    corpus: str
    text_level: TextLevel
    licences: tuple[str, ...]
    raw_data: str
    parser_source: str
    parsing_notes: str
    parse_kwargs: dict[str, Any] = field(default_factory=dict)


CORPORA: dict[str, CorpusSpec] = {
    "english_benedict": CorpusSpec(
        parser=EnglishBenedict,
        language="English",
        language_code="en",
        corpus="Benedict",
        text_level=TextLevel.sentence,
        licences=(CC_BY_NC_SA_4,),
        raw_data="tests/data/parsers/benedict/english/benedict_english_corpus.txt",
        parser_source="src/usas_evaluation_framework/parsers/benedict.py",
        parsing_notes=(
            "Each line is a sentence of `<token>_<USAS tag><MWE>?` items, where the "
            "optional MWE marker is `[i<id>.<total>.<index>]`. The tags `PUNC`, `-`, `.`, "
            "`,` and `!` are converted to `Z9`; punctuation tags are inferred from these "
            "symbols rather than human checked. **The `text` column is the original "
            "annotated line, so it contains the gold USAS tags and MWE markers**; use "
            "`tokens` as model input."
        ),
    ),
    "finnish_benedict": CorpusSpec(
        parser=FinnishBenedict,
        language="Finnish",
        language_code="fi",
        corpus="Benedict",
        text_level=TextLevel.sentence,
        licences=(CC_BY_NC_SA_4,),
        raw_data="tests/data/parsers/benedict/finnish/benedict_finnish_corpus.txt",
        parser_source="src/usas_evaluation_framework/parsers/benedict.py",
        parsing_notes=(
            "Each line is a sentence of `<token>_<USAS tag>(_i)?` items, where `_i` "
            "marks a token that is part of an MWE; consecutive tokens with the same tag "
            "and `_i` marker form one MWE. Punctuation tags (`Z9`) are inferred from known "
            "punctuation symbols rather than human checked. **The `text` column is the "
            "original annotated line, so it contains the gold USAS tags and MWE markers**; "
            "use `tokens` as model input."
        ),
    ),
    "welsh_corcencc": CorpusSpec(
        parser=CorcenccParser,
        language="Welsh",
        language_code="cy",
        corpus="CorCenCC",
        text_level=TextLevel.sentence,
        licences=(CC_BY_NC_SA_4,),
        raw_data="tests/data/parsers/corcencc/corcencc_corpus.txt",
        parser_source="src/usas_evaluation_framework/parsers/corcencc.py",
        parsing_notes=(
            "One sentence per line of `token|lemma|core POS|...|USAS tag` items. Lemmas "
            "(predicted by CyTag) and the core POS tags are included. The corpus has no "
            "MWEs. `text` is the tokens joined by spaces."
        ),
    ),
    "chinese_torch": CorpusSpec(
        parser=TorchParser,
        language="Chinese",
        language_code="zh",
        corpus="ToRCH",
        text_level=TextLevel.sentence,
        licences=(CC_BY_NC_SA_4,),
        raw_data="tests/data/parsers/torch/torch_corpus.csv",
        parser_source="src/usas_evaluation_framework/parsers/torch.py",
        parsing_notes=(
            "CSV with `Token`, `Corrected-USAS` and `sentence-break` columns. A token "
            "may have several gold tags separated by `\"; \"`. Some known annotation "
            "errors are fixed by the parser. The corpus has no MWEs. `text` is the "
            "tokens joined by spaces."
        ),
    ),
    "italian_naacl_2015": CorpusSpec(
        parser=NAACL2015USAS,
        language="Italian",
        language_code="it",
        corpus="NAACL 2015",
        text_level=TextLevel.sentence,
        licences=(CC_BY_NC_SA_3, CC_BY_SA_3),
        raw_data="tests/data/parsers/naacl_2015_italian/naacl_2015_italian_corpus.csv",
        parser_source="src/usas_evaluation_framework/parsers/naacl_2015.py",
        parsing_notes=(
            "CSV with `id`, `token`, `corrected_usas` and `mwe` columns, converted from "
            "the original Excel annotation files with `scripts/naacl_2015_excel_to_csv.py "
            "--language italian`. MWE "
            "groups are re-indexed to start at 1 within each sentence. `text` is the "
            "tokens joined by spaces."
        ),
        parse_kwargs={"dataset_name": "NAACL 2015 Italian", "language": "Italian"},
    ),
    "english_medical_wikipedia": CorpusSpec(
        parser=EnglishWikipediaUSAS,
        language="English",
        language_code="en",
        corpus="Medical Wikipedia",
        text_level=TextLevel.sentence,
        licences=(CC_BY_SA_4,),
        raw_data="tests/data/parsers/english_wikipedia/english_wikipedia_corpus.csv",
        parser_source="src/usas_evaluation_framework/parsers/wikipedia.py",
        parsing_notes=WIKIPEDIA_NOTES,
    ),
    "spanish_medical_wikipedia": CorpusSpec(
        parser=SpanishWikipediaUSAS,
        language="Spanish",
        language_code="es",
        corpus="Medical Wikipedia",
        text_level=TextLevel.sentence,
        licences=(CC_BY_SA_4,),
        raw_data="tests/data/parsers/spanish_wikipedia/spanish_wikipedia_corpus.csv",
        parser_source="src/usas_evaluation_framework/parsers/wikipedia.py",
        parsing_notes=WIKIPEDIA_NOTES,
    ),
    "dutch_medical_wikipedia": CorpusSpec(
        parser=DutchWikipediaUSAS,
        language="Dutch",
        language_code="nl",
        corpus="Medical Wikipedia",
        text_level=TextLevel.sentence,
        licences=(CC_BY_SA_4,),
        raw_data="tests/data/parsers/dutch_wikipedia/dutch_wikipedia_corpus.csv",
        parser_source="src/usas_evaluation_framework/parsers/wikipedia.py",
        parsing_notes=WIKIPEDIA_NOTES,
    ),
    "danish_medical_wikipedia": CorpusSpec(
        parser=DanishWikipediaUSAS,
        language="Danish",
        language_code="da",
        corpus="Medical Wikipedia",
        text_level=TextLevel.sentence,
        licences=(CC_BY_SA_4,),
        raw_data="tests/data/parsers/danish_wikipedia/danish_wikipedia_corpus.csv",
        parser_source="src/usas_evaluation_framework/parsers/wikipedia.py",
        parsing_notes=WIKIPEDIA_NOTES,
    ),
    "hindi_medical_wikipedia": CorpusSpec(
        parser=HindiWikipediaUSAS,
        language="Hindi",
        language_code="hi",
        corpus="Medical Wikipedia",
        text_level=TextLevel.sentence,
        licences=(CC_BY_SA_4,),
        raw_data="tests/data/parsers/hindi_wikipedia/hindi_wikipedia_corpus.csv",
        parser_source="src/usas_evaluation_framework/parsers/wikipedia.py",
        parsing_notes=WIKIPEDIA_NOTES,
    ),
    "igbo_medical_wikipedia": CorpusSpec(
        parser=IgboWikipediaUSAS,
        language="Igbo",
        language_code="ig",
        corpus="Medical Wikipedia",
        text_level=TextLevel.sentence,
        licences=(CC_BY_NC_SA_4,),
        raw_data="tests/data/parsers/igbo_wikipedia/igbo_wikipedia_corpus.csv",
        parser_source="src/usas_evaluation_framework/parsers/wikipedia.py",
        parsing_notes=IGBO_WIKIPEDIA_NOTES,
    ),
}


@dataclass(frozen=True)
class OnRequestCorpus:
    """
    A corpus that must not be uploaded, with statistics for the dataset card.

    The statistics are copied from ``notebooks/dataset_statistics.ipynb``, which
    computes them with the tags in :data:`STATS_LABEL_FILTER` filtered out, as
    :func:`subset_table` does for the uploaded subsets.

    Attributes:
        name: The display name of the corpus.
        licence: A short description of the licence.
        blocked_names: Directory or file name prefixes that identify the corpus's
            raw data. Input paths with a matching part are refused.
        text_level: The text level of the parsed corpus.
        texts: The number of texts.
        tokens: The number of tokens.
        labelled_tokens: The number of labelled tokens.
        multi_tag_tokens: The number of compound semantic tags (multi tag
            membership), see :attr:`.DatasetStats.num_compound_semantic_tags`.
        unique_tags: The number of unique semantic tags.
        mwes: The number of multi word expressions.
    """
    name: str
    licence: str
    blocked_names: tuple[str, ...]
    text_level: str
    texts: int
    tokens: int
    labelled_tokens: int
    multi_tag_tokens: int
    unique_tags: int
    mwes: int


ON_REQUEST_CORPORA: dict[str, OnRequestCorpus] = {
    "chinese_naacl_2015": OnRequestCorpus(
        name="NAACL 2015 Chinese",
        licence=(
            "Non-commercial academic and education use only, under the "
            "[LCMC End User License](https://www.lancaster.ac.uk/fass/projects/corpus/LCMC/lcmc/lcmc_license.htm)"
        ),
        blocked_names=("naacl_2015_chinese",),
        text_level="sentence",
        texts=35,
        tokens=1056,
        labelled_tokens=512,
        multi_tag_tokens=56,
        unique_tags=115,
        mwes=0,
    ),
    "portuguese_naacl_2015": OnRequestCorpus(
        name="NAACL 2015 Portuguese",
        licence="Unknown (texts from the [Lácio-Ref dataset](https://ucrel.lancs.ac.uk/publications/CL2003/papers/aluisio.pdf))",
        blocked_names=("naacl_2015_portuguese",),
        text_level="sentence",
        texts=39,
        tokens=1232,
        labelled_tokens=781,
        multi_tag_tokens=26,
        unique_tags=119,
        mwes=18,
    ),
    "irish_icc": OnRequestCorpus(
        name="ICC Irish",
        licence="No redistribution, full details available upon request",
        blocked_names=("icc_irish", "ICC-GA-"),
        text_level="paragraph",
        texts=3,
        tokens=707,
        labelled_tokens=617,
        multi_tag_tokens=60,
        unique_tags=101,
        mwes=24,
    ),
}

FEATURES = Features({
    "text": Value("string"),
    "tokens": List(Value("string")),
    "lemmas": List(Value("string")),
    "pos_tags": List(Value("string")),
    "semantic_tags": List(List(Value("string"))),
    "mwe_indexes": List(List(Value("int32"))),
})

app = typer.Typer(help="Upload parsed USAS evaluation corpora to the Hugging Face Hub.")


def blocked_corpus(path: Path) -> OnRequestCorpus | None:
    """
    Return the on request corpus that a path belongs to, if any.

    Args:
        path: An input data path.

    Returns:
        The matching on request corpus, or ``None`` if the path is allowed.
    """
    parts = path.resolve().parts
    for corpus in ON_REQUEST_CORPORA.values():
        if any(part.startswith(corpus.blocked_names) for part in parts):
            return corpus
    return None


def parse_dataset_options(dataset_options: list[str]) -> dict[str, list[Path]]:
    """
    Parse the ``SUBSET=PATH`` options into the paths for each subset.

    Args:
        dataset_options: The ``--dataset`` option values.

    Returns:
        The input paths for each subset, in the order given.

    Raises:
        typer.BadParameter: If an option is malformed, names an unknown or on
            request subset, points at a missing file, or points at data that
            must not be uploaded.
    """
    subset_paths: dict[str, list[Path]] = {}
    for option in dataset_options:
        subset, separator, raw_path = option.partition("=")
        if not separator or not subset or not raw_path:
            raise typer.BadParameter(f"Expected SUBSET=PATH, got '{option}'.")

        if subset in ON_REQUEST_CORPORA:
            raise typer.BadParameter(
                f"'{subset}' ({ON_REQUEST_CORPORA[subset].name}) must not be uploaded; "
                "it is only available on request."
            )
        if subset not in CORPORA:
            valid_subsets = ", ".join(CORPORA)
            raise typer.BadParameter(f"Unknown subset '{subset}'. Valid subsets: {valid_subsets}.")

        path = Path(raw_path)
        check_input_path(path)
        subset_paths.setdefault(subset, []).append(path)
    return subset_paths


def check_input_path(path: Path) -> None:
    """
    Check that an input path is a file that may be uploaded.

    Args:
        path: An input data path.

    Raises:
        typer.BadParameter: If the path points at data that must not be
            uploaded, or is not a file.
    """
    blocked = blocked_corpus(path)
    if blocked is not None:
        raise typer.BadParameter(
            f"'{path}' looks like {blocked.name} data, which must not be uploaded."
        )
    if not path.is_file():
        raise typer.BadParameter(f"'{path}' is not a file.")


def default_dataset_paths() -> dict[str, list[Path]]:
    """
    Return the default input path of every subset, for ``--all``.

    Each path is the corpus's :attr:`CorpusSpec.raw_data`, resolved against
    the repository root, so ``--all`` works from any directory.

    Returns:
        The default input path of each subset.

    Raises:
        typer.BadParameter: If a default path is missing or points at data
            that must not be uploaded.
    """
    subset_paths: dict[str, list[Path]] = {}
    for subset, spec in CORPORA.items():
        path = REPO_ROOT / spec.raw_data
        check_input_path(path)
        subset_paths[subset] = [path]
    return subset_paths


def parse_corpus(
    spec: CorpusSpec,
    paths: list[Path],
    label_validation: set[str],
    label_filter: set[str] | None = None,
) -> EvaluationDataset:
    """
    Parse every path of a corpus and merge them into one dataset.

    Args:
        spec: The corpus to parse.
        paths: The input files, each parsed separately.
        label_validation: The valid USAS tags.
        label_filter: Tags to filter out, or ``None`` to keep every tag.

    Returns:
        The parsed dataset.
    """
    datasets = [
        spec.parser.parse(path, label_validation=label_validation, label_filter=label_filter, **spec.parse_kwargs)
        for path in paths
    ]
    if len(datasets) == 1:
        return datasets[0]
    first = datasets[0]
    return EvaluationDataset.merge(first.name, first.text_level, first.language, *datasets)


def to_hf_dataset(dataset: EvaluationDataset) -> Dataset:
    """
    Convert an evaluation dataset into a Hugging Face dataset, one row per text.

    Each MWE index set is converted to a sorted list.

    Args:
        dataset: The dataset to convert.

    Returns:
        The Hugging Face dataset.
    """
    rows: list[dict[str, Any]] = []
    for text in dataset.texts:
        mwe_indexes = None
        if text.mwe_indexes is not None:
            mwe_indexes = [sorted(indexes) for indexes in text.mwe_indexes]
        rows.append({
            "text": text.text,
            "tokens": text.tokens,
            "lemmas": text.lemmas,
            "pos_tags": text.pos_tags,
            "semantic_tags": text.semantic_tags,
            "mwe_indexes": mwe_indexes,
        })
    return Dataset.from_list(rows, features=FEATURES)


Visibility: TypeAlias = Literal["missing", "inaccessible", "private", "public", "unknown"]


def can_create_repo(repo_id: str, token: str | None) -> bool:
    """
    Return whether the token's user could create the repository.

    The repository can be created if it has no namespace (it is then created
    under the user), or if its namespace is the user or one of their
    organisations.

    Args:
        repo_id: The Hugging Face dataset repository ID.
        token: The Hugging Face token. If ``None``, the token saved by
            ``hf auth login`` is used, if there is one.

    Returns:
        ``True`` if the repository's namespace belongs to the token's user,
        ``False`` otherwise, including when there is no token.

    Raises:
        typer.Exit: If the token is invalid.
        HfHubHTTPError: If the Hub cannot be queried for another reason, for
            example a rate limit or a server error.
    """
    namespace, separator, _ = repo_id.rpartition("/")
    try:
        user = HfApi().whoami(token=token)
    except LocalTokenNotFoundError:
        return False
    except HfHubHTTPError:
        raise
    except HTTPError as error:
        # whoami re-raises a 401 (invalid token) as a plain requests HTTPError,
        # not as an HfHubHTTPError; every other error is an HfHubHTTPError.
        typer.echo(f"The Hugging Face token is invalid: {error}", err=True)
        raise typer.Exit(code=1) from error
    if not separator:
        return True
    namespaces = {user["name"], *(org["name"] for org in user.get("orgs", []))}
    return namespace in namespaces


def repo_visibility(repo_id: str, token: str | None) -> Visibility:
    """
    Return the visibility of a dataset repository on the Hub.

    The Hub gives the same "not found" error for a repository that does not
    exist and for one that the token cannot access, so in that case the
    repository is only treated as missing if the token's user could create it
    (see :func:`can_create_repo`).

    Args:
        repo_id: The Hugging Face dataset repository ID.
        token: The Hugging Face token, needed to see a private repository.

    Returns:
        ``'missing'`` if the repository was not found and the token's user
        could create it, ``'inaccessible'`` if it was not found and they
        could not (it may exist but is hidden from the token),
        ``'private'`` or ``'public'`` if it exists, or ``'unknown'`` if the
        Hub does not say.

    Raises:
        typer.Exit: If the token is invalid, from :func:`can_create_repo`.
        HfHubHTTPError: If the Hub cannot be queried for another reason, for
            example a rate limit or a server error.
    """
    try:
        info = HfApi().dataset_info(repo_id, token=token)
    except RepositoryNotFoundError:
        return "missing" if can_create_repo(repo_id, token) else "inaccessible"
    match info.private:
        case True:
            return "private"
        case False:
            return "public"
        case _:
            return "unknown"


def visibility_message(
    repo_id: str,
    visibility: Visibility,
    private: bool,
) -> tuple[bool, str] | None:
    """
    Check the ``--private`` option against the repository's visibility.

    ``push_to_hub`` only applies ``private`` when it creates the repository,
    so uploading to an existing public repository would make the data public
    even with ``--private``. That upload is refused unless ``--no-private`` is
    given, because data that has been public cannot be made private again.

    Args:
        repo_id: The Hugging Face dataset repository ID.
        visibility: The repository's visibility, from :func:`repo_visibility`.
        private: The ``--private`` option.

    Returns:
        ``None`` if the repository's visibility already matches ``private``
        (it is private and ``private`` is ``True``, or it is public or
        unknown and ``private`` is ``False``), so the upload can go ahead
        with no message.

        Otherwise a ``(refused, message)`` tuple, where ``message`` explains
        the visibility and ``refused`` is:

        * ``True`` if the upload must be refused: the repository cannot be
          accessed, or it is public or its visibility is unknown and
          ``private`` is ``True``.
        * ``False`` if the upload can go ahead and ``message`` is only
          information: the repository does not exist and will be created
          with the requested visibility, or it is private and ``private``
          is ``False``.
    """
    requested = "private" if private else "public"
    match visibility, private:
        case "inaccessible", _:
            return True, (
                f"Cannot access {repo_id}: it was not found, and its namespace does not belong to the "
                "token's user or their organisations, so it either exists but the token cannot read it, "
                "or it cannot be created with this token."
            )
        case "missing", _:
            return False, f"{repo_id} does not exist; it will be created as {requested}."
        case "public", True:
            return True, (
                f"{repo_id} is public, so uploading would make the data public even though --private "
                "is set. Pass --no-private to upload to a public repository."
            )
        case "unknown", True:
            return True, (
                f"Could not tell whether {repo_id} is private, so it may be public. "
                "Pass --no-private to upload anyway."
            )
        case "private", False:
            return False, f"{repo_id} is private and stays private; --no-private does not change it."
        case _:
            return None


def load_existing_card(repo_id: str, token: str | None) -> DatasetCard:
    """
    Load the dataset card currently on the Hub, or an empty card if there is none.

    Only used by a dry run, so that its card starts from the same metadata as
    an upload's. Nothing is written to the Hub.

    Args:
        repo_id: The Hugging Face dataset repository ID.
        token: The Hugging Face token, needed only for a private repository.

    Returns:
        The existing card, or an empty card if the repository or its card
        does not exist or cannot be read.
    """
    try:
        return DatasetCard.load(repo_id, repo_type="dataset", token=token)
    except (HfHubHTTPError, OfflineModeIsEnabled) as error:
        typer.echo(f"Could not read the dataset card of {repo_id} ({type(error).__name__}); starting from an empty card.")
        # An empty YAML mapping, as a card with no metadata block logs a warning.
        return DatasetCard("---\n{}\n---\n")


def record_subset(card_data: DatasetCardData, repo_id: str, subset: str, split: str, dataset: Dataset) -> None:
    """
    Record a subset in the card metadata as ``Dataset.push_to_hub`` does.

    ``push_to_hub`` adds a ``configs`` entry that points at the subset's data
    files and a ``dataset_info`` entry with its features and number of rows.
    A dry run uploads nothing, so it records them with this function instead.
    The byte sizes are estimates, because nothing is written as Parquet.

    Args:
        card_data: The card metadata, updated in place.
        repo_id: The Hugging Face dataset repository ID.
        subset: The subset name.
        split: The split name.
        dataset: The subset's data.
    """
    num_bytes = dataset.data.nbytes
    info = dataset.info.copy()
    info.config_name = subset
    info.dataset_size = num_bytes
    info.splits = SplitDict({
        split: SplitInfo(split, num_bytes=num_bytes, num_examples=len(dataset), dataset_name=repo_id.split("/")[-1])
    })
    DatasetInfosDict({subset: info}).to_dataset_card_data(card_data)
    data_files = [{"split": split, "path": f"{subset}/{split}-*"}]
    MetadataConfigs({subset: {"data_files": data_files}}).to_dataset_card_data(card_data)


def subset_num_texts(card_data: DatasetCardData, split: str) -> dict[str, int]:
    """
    Read the number of texts (rows) of each subset's split from the card metadata.

    Args:
        card_data: The card metadata, with the ``dataset_info`` entries that
            ``push_to_hub`` or :func:`record_subset` wrote.
        split: The split name.

    Returns:
        The number of texts of each subset that has the split.
    """
    num_texts: dict[str, int] = {}
    for subset, info in DatasetInfosDict.from_dataset_card_data(card_data).items():
        split_info = info.splits.get(split) if info.splits else None
        if split_info is not None:
            num_texts[subset] = split_info.num_examples
    return num_texts


def size_category(num_rows: int) -> str:
    """
    Return the Hugging Face size category for a number of rows.

    Args:
        num_rows: The number of rows in the whole dataset.

    Returns:
        The size category, e.g. ``'1K<n<10K'``.
    """
    for upper_bound, category in SIZE_CATEGORIES:
        if num_rows < upper_bound:
            return category
    return "n>1B"


STATS_HEADER = "Texts | Tokens | Labelled tokens | Multi tag membership | Unique semantic tags | MWEs"


def multi_tag_membership(multi_tag_tokens: int, labelled_tokens: int) -> str:
    """
    Format the multi tag membership count with its percentage of labelled tokens.

    Args:
        multi_tag_tokens: The number of compound semantic tags.
        labelled_tokens: The number of labelled tokens.

    Returns:
        The count and percentage, e.g. ``'56 (10.9%)'``.

    Examples:
        >>> multi_tag_membership(56, 512)
        '56 (10.9%)'
        >>> multi_tag_membership(0, 0)
        '0 (0.0%)'
    """
    percentage = multi_tag_tokens / labelled_tokens * 100 if labelled_tokens else 0.0
    return f"{multi_tag_tokens} ({percentage:.1f}%)"


def stats_cells(dataset: EvaluationDataset) -> str:
    """
    Build the statistics cells of a table row, in the order of :data:`STATS_HEADER`.

    Args:
        dataset: The dataset, parsed with :data:`STATS_LABEL_FILTER` filtered out.

    Returns:
        The markdown cells, separated by ``" | "``.
    """
    stats = dataset.stats()
    labelled_tokens = stats.num_labelled_tokens or 0
    unique_tags = stats.unique_semantic_tags or frozenset()
    cells = [
        stats.num_texts,
        stats.num_tokens,
        labelled_tokens,
        multi_tag_membership(stats.num_compound_semantic_tags or 0, labelled_tokens),
        len(unique_tags),
        stats.num_mwes or 0,
    ]
    return " | ".join(str(cell) for cell in cells)


def subset_table(repo_subsets: list[str], stats_datasets: dict[str, EvaluationDataset], num_texts: dict[str, int]) -> str:
    """
    Build the markdown table of subsets in the repository.

    Args:
        repo_subsets: Every subset in the repository.
        stats_datasets: The datasets parsed in this run with
            :data:`STATS_LABEL_FILTER` filtered out, by subset. Subsets
            uploaded in an earlier run only have their number of texts.
        num_texts: The number of texts of each subset, from the card metadata,
            used for subsets uploaded in an earlier run.

    Returns:
        The markdown table.
    """
    lines = [
        f"| Subset | Language | Corpus | Text level | {STATS_HEADER} | Licence |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for subset in repo_subsets:
        spec = CORPORA[subset]
        licences = ", ".join(spec.licences)
        dataset = stats_datasets.get(subset)
        if dataset is None:
            cells = f"{num_texts.get(subset, '-')} | - | - | - | - | -"
        else:
            cells = stats_cells(dataset)
        lines.append(
            f"| `{subset}` | {spec.language} | {spec.corpus} | {spec.text_level.value} | "
            f"{cells} | {licences} |"
        )
    return "\n".join(lines)


def on_request_table() -> str:
    """
    Build the markdown table of corpora that are only available on request.

    Returns:
        The markdown table.
    """
    lines = [
        f"| Corpus | Text level | {STATS_HEADER} | Licence |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for corpus in ON_REQUEST_CORPORA.values():
        membership = multi_tag_membership(corpus.multi_tag_tokens, corpus.labelled_tokens)
        lines.append(
            f"| {corpus.name} | {corpus.text_level} | {corpus.texts} | {corpus.tokens} | "
            f"{corpus.labelled_tokens} | {membership} | {corpus.unique_tags} | "
            f"{corpus.mwes} | {corpus.licence} |"
        )
    return "\n".join(lines)


def parsing_sections(repo_subsets: list[str]) -> str:
    """
    Build the markdown section on how each subset was parsed.

    Args:
        repo_subsets: Every subset in the repository.

    Returns:
        One markdown section per subset.
    """
    sections: list[str] = []
    for subset in repo_subsets:
        spec = CORPORA[subset]
        sections.append(
            f"### `{subset}`\n\n"
            f"* Parser: `{spec.parser.__name__}` "
            f"([source]({GITHUB_BLOB_URL}/{spec.parser_source}), [docs]({PARSERS_DOCS_URL}))\n"
            f"* Raw data: [{spec.raw_data}]({GITHUB_BLOB_URL}/{spec.raw_data})\n\n"
            f"{spec.parsing_notes}"
        )
    return "\n\n".join(sections)


def card_body(
    repo_id: str,
    split: str,
    repo_subsets: list[str],
    stats_datasets: dict[str, EvaluationDataset],
    num_texts: dict[str, int],
) -> str:
    """
    Build the markdown body of the dataset card from ``dataset_card_template.md``.

    Args:
        repo_id: The Hugging Face dataset repository ID.
        split: The split name every subset was pushed to.
        repo_subsets: Every subset in the repository.
        stats_datasets: The datasets parsed in this run with
            :data:`STATS_LABEL_FILTER` filtered out, by subset.
        num_texts: The number of texts of each subset, from the card metadata.

    Returns:
        The markdown body.

    Raises:
        KeyError: If the template has a placeholder that is not filled in.
    """
    example_subset = repo_subsets[0]
    example_spec = CORPORA[example_subset]
    template = Template(CARD_TEMPLATE.read_text(encoding="utf-8"))
    return template.substitute(
        github_url=GITHUB_URL,
        github_blob_url=GITHUB_BLOB_URL,
        repo_id=repo_id,
        split=split,
        subset_table=subset_table(repo_subsets, stats_datasets, num_texts),
        parsing_sections=parsing_sections(repo_subsets),
        on_request_table=on_request_table(),
        example_subset=example_subset,
        example_name=f"{example_spec.corpus} {example_spec.language}",
        example_language=example_spec.language,
        example_text_level=example_spec.text_level.name,
    )


def card_metadata(repo_subsets: list[str], num_texts: dict[str, int]) -> dict[str, Any]:
    """
    Build the dataset card metadata that this script controls.

    Args:
        repo_subsets: Every subset in the repository.
        num_texts: The number of texts of each subset, for the size category.

    Returns:
        The metadata fields.
    """
    specs = [CORPORA[subset] for subset in repo_subsets]
    languages = sorted({spec.language_code for spec in specs})
    total_texts = sum(num_texts.get(subset, 0) for subset in repo_subsets)
    return {
        "pretty_name": "USAS Evaluation Datasets",
        "license": sorted({licence for spec in specs for licence in spec.licences}),
        "language": languages,
        "multilinguality": ["multilingual" if len(languages) > 1 else "monolingual"],
        "size_categories": [size_category(total_texts)],
        "task_categories": ["token-classification"],
        "tags": ["usas", "semantic-tagging", "word-sense-disambiguation"],
    }


def update_card(card: DatasetCard, repo_id: str, split: str, stats_datasets: dict[str, EvaluationDataset]) -> None:
    """
    Fill in the metadata and body of a dataset card, for an upload or a dry run.

    The card's ``configs`` metadata must already list every subset in the
    repository, as it does after ``push_to_hub`` or :func:`record_subset`.

    Args:
        card: The dataset card, updated in place.
        repo_id: The Hugging Face dataset repository ID.
        split: The split name every subset was pushed to.
        stats_datasets: The datasets parsed in this run with
            :data:`STATS_LABEL_FILTER` filtered out, by subset.
    """
    card_data = cast(DatasetCardData, card.data)
    config_names = MetadataConfigs.from_dataset_card_data(card_data)
    repo_subsets = sorted(subset for subset in config_names if subset in CORPORA)
    num_texts = subset_num_texts(card_data, split)
    for key, value in card_metadata(repo_subsets, num_texts).items():
        card_data[key] = value
    card.text = card_body(repo_id, split, repo_subsets, stats_datasets, num_texts)


def print_stats(subset: str, dataset: EvaluationDataset) -> None:
    """
    Print a short summary of a parsed dataset.

    Args:
        subset: The subset name.
        dataset: The parsed dataset.
    """
    stats = dataset.stats()
    typer.echo(
        f"{subset}: {stats.num_texts} texts, {stats.num_tokens} tokens, "
        f"{stats.num_labelled_tokens or 0} labelled tokens, {stats.num_mwes or 0} MWEs"
    )


@app.command()
def main(
    repo_id: Annotated[str, typer.Argument(help="Hugging Face dataset repository ID, e.g. 'ucrelnlp/USAS-Eval'.")],
    dataset: Annotated[
        list[str] | None,
        typer.Option(
            "--dataset",
            "-d",
            help=(
                "SUBSET=PATH of a corpus to upload. Repeat for each corpus; repeat a subset to "
                f"merge several files into it. Subsets: {', '.join(CORPORA)}."
            ),
        ),
    ] = None,
    all_datasets: Annotated[
        bool,
        typer.Option(
            "--all",
            help=(
                "Upload every subset from its default raw data file in this repository (see the "
                "README). Cannot be combined with --dataset."
            ),
        ),
    ] = False,
    private: Annotated[
        bool,
        typer.Option(
            help=(
                "Create the repository as private if it does not exist. The visibility of an "
                "existing repository is not changed (change it in the repository's settings on "
                "the Hub), so with --private an upload to an existing public repository is "
                "refused; pass --no-private to upload to it."
            )
        ),
    ] = True,
    split: Annotated[str, typer.Option(help="Split name for every subset.")] = "test",
    dry_run: Annotated[
        bool,
        typer.Option(
            help=(
                "Check everything without uploading: parse and validate each corpus, print its "
                "statistics, report whether the repository's visibility would refuse the upload, "
                "and write the dataset card that an upload would push to --card-output. The card "
                "includes any subsets already on the Hub if the existing card can be read (a "
                "private repository needs HF_TOKEN); otherwise no token is needed."
            )
        ),
    ] = False,
    card_output: Annotated[
        Path,
        typer.Option(help="Where --dry-run writes the dataset card."),
    ] = Path("dataset_card.md"),
) -> None:
    """
    Parse each corpus and upload it as its own subset of one Hugging Face dataset repository.

    Args:
        repo_id: Hugging Face dataset repository ID.
        dataset: SUBSET=PATH values of the corpora to upload.
        all_datasets: Upload every subset from its default raw data file.
        private: Whether to create the repository as private.
        split: Split name for every subset.
        dry_run: Parse and write the card locally without uploading.
        card_output: Where a dry run writes the dataset card.

    Raises:
        typer.BadParameter: If neither or both of ``--dataset`` and ``--all``
            are given, or an input path is invalid.
        typer.Exit: If ``HF_TOKEN`` is not set when uploading, if the token
            is invalid, if the repository cannot be accessed, or if
            ``--private`` is set and the repository is public or its
            visibility is unknown.
    """
    match bool(dataset), all_datasets:
        case True, True:
            raise typer.BadParameter("Use either --dataset or --all, not both.")
        case False, False:
            raise typer.BadParameter("Give at least one --dataset SUBSET=PATH, or --all.")
        case False, True:
            subset_paths = default_dataset_paths()
        case _:
            subset_paths = parse_dataset_options(dataset or [])

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not dry_run and not token:
        typer.echo("HF_TOKEN is not set; set it in the environment or in a .env file.", err=True)
        raise typer.Exit(code=1)

    # Checked before parsing, so that a refused upload fails straight away.
    if dry_run:
        try:
            visibility_check = visibility_message(repo_id, repo_visibility(repo_id, token), private)
        except (HfHubHTTPError, OfflineModeIsEnabled) as error:
            visibility_check = False, f"Could not check the visibility of {repo_id} ({type(error).__name__})."
        if visibility_check is not None:
            refused, message = visibility_check
            typer.echo(f"An upload would be refused: {message}" if refused else message)
    else:
        try:
            visibility_check = visibility_message(repo_id, repo_visibility(repo_id, token), private)
        except HfHubHTTPError as error:
            typer.echo(f"Cannot access {repo_id} ({type(error).__name__}: {error}).", err=True)
            raise typer.Exit(code=1) from error
        if visibility_check is not None:
            refused, message = visibility_check
            typer.echo(message, err=refused)
            if refused:
                raise typer.Exit(code=1)

    label_validation = set(load_usas_mapper(None, None).keys())

    parsed: dict[str, EvaluationDataset] = {}
    stats_datasets: dict[str, EvaluationDataset] = {}
    hf_datasets: dict[str, Dataset] = {}
    for subset, paths in subset_paths.items():
        spec = CORPORA[subset]
        parsed[subset] = parse_corpus(spec, paths, label_validation)
        # Parsed again with tags filtered out, only for the card statistics.
        stats_datasets[subset] = parse_corpus(spec, paths, label_validation, set(STATS_LABEL_FILTER))
        hf_datasets[subset] = to_hf_dataset(parsed[subset])
        print_stats(subset, parsed[subset])

    if dry_run:
        # Start from the card on the Hub and record each subset in it as
        # push_to_hub would, so that the card matches the one an upload makes.
        card = load_existing_card(repo_id, token)
        for subset, hf_dataset in hf_datasets.items():
            record_subset(cast(DatasetCardData, card.data), repo_id, subset, split, hf_dataset)
        update_card(card, repo_id, split, stats_datasets)
        card_output.write_text(card.content, encoding="utf-8")
        typer.echo(f"Dry run: nothing uploaded. Dataset card written to {card_output}")
        return

    for subset, hf_dataset in hf_datasets.items():
        typer.echo(f"Uploading {subset} to {repo_id}")
        hf_dataset.push_to_hub(repo_id, config_name=subset, split=split, token=token, private=private)

    # push_to_hub records each subset in the card's `configs` and `dataset_info`
    # metadata, so start from that card, which also covers earlier uploads.
    card = DatasetCard.load(repo_id, repo_type="dataset", token=token)
    update_card(card, repo_id, split, stats_datasets)
    card.push_to_hub(repo_id, repo_type="dataset", token=token)
    typer.echo(f"Uploaded {len(parsed)} subset(s) and the dataset card to {repo_id}")


if __name__ == "__main__":
    app()
