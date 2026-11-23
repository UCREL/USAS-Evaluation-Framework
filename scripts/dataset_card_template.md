
# USAS Evaluation Datasets

Human annotated (gold) [USAS](https://ucrel.lancs.ac.uk/usas/) semantic tag
datasets for evaluating USAS semantic taggers / word sense disambiguation
models. Each subset is one corpus in one language, parsed with the
[USAS Evaluation Framework]($github_url) from UCREL at Lancaster University.


## Subsets

Every subset has a single `$split` split.

$subset_table

The statistics are computed as in
[`notebooks/dataset_statistics.ipynb`]($github_blob_url/notebooks/dataset_statistics.ipynb),
with the `Z9` (punctuation), and `Z99` (unmatched)
tags filtered out; the tokens are not filtered. Labelled tokens are tokens
with at least one semantic tag that is not filtered out. Multi tag membership is
the number of compound semantic tags (such as `F2/O2`), and its percentage of
labelled tokens. The number of Unique Semantic Tags reflect the number of unique USAS tags that are
present in the test data, as you can see out of the possible 232 [USAS tags](https://ucrel.lancs.ac.uk/usas/), not all USAS tags are
represented in any of the test corpora, which is not unexpected as some USAS tags
are domain specific and some are very rare. MWEs are the number of Multi Word Expressions that have
been labelled in the dataset.

## Data fields

One row per text (a sentence or paragraph, see the text level above):

* `text` (`string`): the text. For most corpora this is the tokens joined by
  spaces; see the parsing notes below for exceptions.
* `tokens` (`list[string]`): the tokens.
* `lemmas` (`list[string]` or `null`): the lemma of each token, if the corpus has them.
* `pos_tags` (`list[string]` or `null`): the POS tag of each token, if the corpus has them.
* `semantic_tags` (`list[list[string]]`): the gold USAS tags of each token. Most
  tokens have one tag. A tag can be a compound tag such as `F2/O2`. An empty
  string `""` means the token has no label.
* `mwe_indexes` (`list[list[int]]` or `null`): for each token, the IDs of the
  multi word expressions (MWEs) it belongs to; tokens that share an ID form one
  MWE. An empty list means the token is not part of an MWE. `null` means the
  corpus has no MWE annotation. IDs are only unique to their given text row.

## How the data was parsed

Each raw corpus was parsed with its parser from the USAS Evaluation Framework,
with every tag validated against the [USAS tagset](https://ucrel.lancs.ac.uk/usas/usas_guide.pdf)
([usas_validator.utils.load_usas_mapper](https://ucrel.github.io/USAS-Validator/api/utils.html#usas_validator.utils.load_usas_mapper))
and no tags filtered out. Punctuation has the USAS tag `Z9`.

$parsing_sections

## Usage

With the Hugging Face `datasets` [library](https://github.com/huggingface/datasets):

```python
from datasets import load_dataset

$example_subset = load_dataset("$repo_id", "$example_subset", split="$split")
```

To use a subset with the [USAS Evaluation Framework]($github_url), to make uses of USAS Evaluation Frameworks' metrics functionality, convert the rows back into an `EvaluationDataset`:

```python
from datasets import load_dataset

from usas_evaluation_framework.dataset import EvaluationDataset, EvaluationTexts, TextLevel

rows = load_dataset("$repo_id", "$example_subset", split="$split")
texts = [
    EvaluationTexts(
        text=row["text"],
        tokens=row["tokens"],
        lemmas=row["lemmas"],
        pos_tags=row["pos_tags"],
        semantic_tags=row["semantic_tags"],
        mwe_indexes=None if row["mwe_indexes"] is None else [frozenset(ids) for ids in row["mwe_indexes"]],
    )
    for row in rows
]
dataset = EvaluationDataset(
    name="$example_name",
    text_level=TextLevel.$example_text_level,
    language="$example_language",
    texts=texts,
)
```

Alternatively, parse the raw data directly with the framework's parsers, which
also let you filter and validate labels.

## Raw data

The raw annotated files are in the
[USAS Evaluation Framework repository]($github_url) under
[`tests/data/parsers/`]($github_url/tree/main/tests/data/parsers); each subset
above links to its file.

## Datasets available on request

The following datasets are not included because their licences do not allow
redistribution. They can be obtained by an email request to Paul Rayson or
Andrew Moore at Lancaster University of which Paul and Andrew may need to contact the original authors; the data is free, but you must agree to
the licence terms below.

These statistics are from
[`notebooks/dataset_statistics.ipynb`]($github_blob_url/notebooks/dataset_statistics.ipynb)
and are computed in the same way as the subset table above.

$on_request_table

## Licence

Each subset has its own licence, given in the subset table above:

* `cc-by-nc-sa-4.0`: [Creative Commons Attribution Non Commercial Share Alike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en)
* `cc-by-sa-4.0`: [Creative Commons Attribution Share Alike 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en)
* `cc-by-nc-sa-3.0` and `cc-by-sa-3.0`: dual licensed under [Creative Commons Attribution Non-Commercial Share Alike 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/deed.en) and [Creative Commons Attribution Share Alike 3.0](https://creativecommons.org/licenses/by-sa/3.0/deed.en)

## Acknowledgements and Citations

We are very grateful to everyone who has created the evaluation datasets used in the forthcoming paper. Over the years many people have contributed to them, and they can be credited through the citations below:

* Carmen Dayrell for manually checking the Portuguese NAACL 2015 evaluation data.
* Daisy Lal for annotating the English and Hindi Wikipedia data.
* Ignatius Ezeani for annotating the Igbo Wikipedia data.
* Bettina Mølri Knudsen and Mette Holz Meinhardt Gregersen for annotating the Danish Wikipedia data.
* Hugo Sanjurjo-Gonzalez for annotating the Spanish Wikipedia data.
* Erik van Mulligen for annotating the Dutch Wikipedia data.
* Tim Czerniak, Elaine Uí Dhonnchadha, and Gearóid Ó Donnchadha for annotating the Irish data.
* Yan Yabo and Xiaobin Yang for annotating the ToRCH Chinese data.

Cite the following papers for each evaluation dataset:

* Wikipedia datasets: forthcoming paper.
* NAACL 2015 Chinese, Italian and Portuguese: Piao et al. (2015). [Development of the Multilingual Semantic Annotation System](https://aclanthology.org/N15-1137/). In *Proceedings of NAACL-HLT 2015*.
* ToRCH Chinese: Moore et al. (2026). [Creating a Hybrid Rule and Neural Network Based Semantic Tagger using Silver Standard Data: the PyMUSAS framework for Multilingual Semantic Annotation](https://aclanthology.org/2026.lrec-1.926/). In *Proceedings of LREC 2026*.
* Irish: Czerniak and Uí Dhonnchadha (2024). [Towards Semantic Tagging for Irish](https://aclanthology.org/2024.lrec-main.1446/). In *Proceedings of LREC 2024*.
* Welsh CorCenCC: Ezeani et al. (2019). [Leveraging Pre-Trained Embeddings for Welsh Taggers](https://aclanthology.org/W19-4332/). In *Proceedings of the 4th Workshop on Representation Learning for NLP (RepL4NLP)*.
* Finnish and English Benedict: Löfberg et al. (2003). [Porting an English Semantic Tagger to the Finnish Language](https://ucrel.lancs.ac.uk/publications/CL2003/papers/lofberg.pdf). In *Proceedings of Corpus Linguistics 2003*.

