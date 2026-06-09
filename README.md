# USAS-Evaluation-Framework

Evaluation metrics and datasets for USAS Semantic Tagging


## Setup

You can either use the dev container with your favourite editor, e.g. VSCode. Or you can create your setup locally below we demonstrate both.

In both cases they share the same tools, of which these tools are:
* [uv](https://docs.astral.sh/uv/) for Python packaging and development
* [make](https://www.gnu.org/software/make/) (OPTIONAL) for automation of tasks, not strictly required but makes life easier.

### Dev Container

A [dev container](https://containers.dev/) uses a docker container to create the required development environment, the Dockerfile we use for this dev container can be found at [./.devcontainer/Dockerfile](./.devcontainer/Dockerfile). To run it locally it requires docker to be installed, you can also run it in a cloud based code editor, for a list of supported editors/cloud editors see [the following webpage.](https://containers.dev/supporting)

To run for the first time on a local VSCode editor (a slightly more detailed and better guide on the [VSCode website](https://code.visualstudio.com/docs/devcontainers/tutorial)):
1. Ensure docker is running.
2. Ensure the VSCode [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension is installed in your VSCode editor.
3. Open the command pallete `CMD + SHIFT + P` and then select `Dev Containers: Rebuild and Reopen in Container`

You should now have everything you need to develop, `uv`, `make`, for VSCode various extensions like `Pylance`, etc.

If you have any trouble see the [VSCode website.](https://code.visualstudio.com/docs/devcontainers/tutorial).

### Local

To run locally first ensure you have the following tools installted locally:
* [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python packaging and development. (version `0.9.6`)
* [make](https://www.gnu.org/software/make/) (OPTIONAL) for automation of tasks, not strictly required but makes life easier.
  * Ubuntu: `apt-get install make`
  * Mac: [Xcode command line tools](https://mac.install.guide/commandlinetools/4) includes `make` else you can use [brew.](https://formulae.brew.sh/formula/make)
  * Windows: Various solutions proposed in this [blog post](https://earthly.dev/blog/makefiles-on-windows/) on how to install on Windows, inclduing `Cygwin`, and `Windows Subsystem for Linux`.

When developing on the project you will want to install the Python package locally in editable format with all the extra requirements, this can be done like so:

```bash
uv sync --all-extras
```

### Linting

Linting and formatting with [ruff](https://docs.astral.sh/ruff/) it is a replacement for tools like Flake8, isort, Black etc, and we us [ty](https://github.com/astral-sh/ty) for type checking.

To run the linting:

``` bash
make lint
```

### Tests

To run the tests (uses pytest and coverage) and generate a coverage report:

``` bash
make test
```

To test the parsing of the Irish ICC dataset, i.e. to fully test the `usas_evaluation_framework.parsers.icc_irish.ICCIrishParser.parse` method fully it requires downloading the Irish ICC human annotated dataset files too: `tests/data/parsers/icc_irish`, e.g. `tests/data/parsers/icc_irish/ICC-GA-WPH-001-the_wire.tsv`.

## Evaluation metrics and splits

### Metrics

#### Top-N Accuracy

* Micro
* Macro

#### Coverage

This only applies to the rule based methods

### Splits

#### Tokens within a lexicon

#### Unseen tokens from the training dataset

#### Unseen token/semantic label from the training dataset

#### Top level categories

Metrics scores for the 21 top level categories

#### Named Entities

## Notes

In the future we should be able to replace [./src/usas_evaluation_framework/data_utils.py](./src/usas_evaluation_framework/data_utils.py) with [https://github.com/UCREL/USAS-Validator](https://github.com/UCREL/USAS-Validator)

## Scripts

### Wikipedia Medical 2026 Excel data to CSV

**Note** to use this script you need to instal the `excel-conversion` extra: `uv pip install ".[excel-conversion]"`

If you want to convert the annotated Wikipedia Medical 2026 Excel data to CSV, this script will do so; it assumes that one folder of excel files (each folder we assumed to only contain one language at time of processing this data) is given as input and the output is a single CSV file containing all of the Excel file data;

To note that all of the fields are parsed into the CSV unchanged apart from special cases listed in the help output below, for more detail have a read of the script itself; [./scripts/medical_wikipedia_excel_to_csv.py script](./scripts/medical_wikipedia_excel_to_csv.py):

``` bash
uv run scripts/medical_wikipedia_excel_to_csv.py --help
                                                                                                                                                                                                                                                                                                                                                                                                                                        
 Usage: medical_wikipedia_excel_to_csv.py [OPTIONS] FOLDER                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                                        
 Parse all Excel files in FOLDER into a single CSV file.                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                                        
 Args:                                                                                                                                                                                                                                                                                                                                                                                                                                  
     folder: Folder containing Excel files to parse.                                                                                                                                                                                                                                                                                                                                                                                    
     output: Path for the output CSV file. Defaults to output.csv.                                                                                                                                                                                                                                                                                                                                                                      
     sheet: Sheet name or 0-based index to read. Defaults to the first sheet.                                                                                                                                                                                                                                                                                                                                                           
     add_source: Add a 'source_file' column with the originating filename.                                                                                                                                                                                                                                                                                                                                                              
     recursive: Search subdirectories recursively.                                                                                                                                                                                                                                                                                                                                                                                      
     punct_to_z9: Replace PUNCT tags with Z9 in the corrected USAS column.                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                                                                                        
 Raises:                                                                                                                                                                                                                                                                                                                                                                                                                                
     typer.Exit: If no Excel files are found or a file cannot be read.                                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                                                                                                        
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    folder      DIRECTORY  Folder containing Excel files to parse. [required]                                                                                                                                                                                                                                                                                                                                                       │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --output              -o                      PATH  Path for the output CSV file. [default: output.csv]                                                                                                                                                                                                                                                                                                                              │
│ --sheet               -s                      TEXT  Sheet name or 0-based index to read from each file. Defaults to the first sheet.                                                                                                                                                                                                                                                                                                 │
│ --add-source              --no-add-source           Add a 'source_file' column with the originating filename. [default: no-add-source]                                                                                                                                                                                                                                                                                               │
│ --recursive           -r  --no-recursive            Search for Excel files recursively in subdirectories. [default: no-recursive]                                                                                                                                                                                                                                                                                                    │
│ --punct-to-z9             --no-punct-to-z9          Replace PUNCT tags in 'corrected USAS' with Z9. Also fills empty 'corrected USAS' cells with Z9 when 'predicted USAS' or 'POS' is PUNCT. [default: no-punct-to-z9]                                                                                                                                                                                                               │
│ --install-completion                                Install completion for the current shell.                                                                                                                                                                                                                                                                                                                                        │
│ --show-completion                                   Show completion for the current shell, to copy it or customize the installation.                                                                                                                                                                                                                                                                                                 │
│ --help                                              Show this message and exit.                                                                                                                                                                                                                                                                                                                                                      │
╰────

```

For example, which takes all of the excel files in `./Data/Final_Annotated_Data/Spanish` and writes all the rows from all the files into one CSV file `Data/spanish.csv`

``` bash
uv run scripts/medical_wikipedia_excel_to_csv.py ./Data/Final_Annotated_Data/Spanish --output Data/alt_spanish.csv --punct-to-z9
```


### NAACL 2015 Annotated Excel data to CSV

**Note** to use this script you need to instal the `excel-conversion` extra: `uv pip install ".[excel-conversion]"`

If you want to convert the [NAACL 2015](https://aclanthology.org/N15-1137.pdf) Excel files into a single CSV script the following script will do so;


``` bash
uv run scripts/naacl_2015_excel_to_csv.py --help
                                                                                                                                                                                                                                                                                                                                                                                                                                        
  Usage: naacl_2015_excel_to_csv.py [OPTIONS] FOLDER                                                                                                                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                                                                                                                                        
 Build a sentence-structured CSV from a folder of annotated Excel files.                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                                        
 Each Excel file must contain columns: token, predicted_usas, corrected_usas.                                                                                                                                                                                                                                                                                                                                                           
 For Italian data, a 'mwe' column is also required.                                                                                                                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                                                                                                                                        
 The output CSV contains: id, token, corrected_usas[, mwe], where id is                                                                                                                                                                                                                                                                                                                                                                 
 FILE_NAME|SENTENCE_COUNT|TOKEN_COUNT. A blank row is inserted after each                                                                                                                                                                                                                                                                                                                                                               
 sentence-ending token to delimit sentences.                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                        
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    folder      DIRECTORY  Folder containing Excel files to process. [required]                                                                                                                                                                                                                                                                                                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --output              -o      PATH               Path for the output CSV file. [default: output.csv]                                                                                                                                                                                                                                                                                                                                 │
│ --language            -l      [chinese|italian]  Language of the data (chinese or italian). [default: chinese]                                                                                                                                                                                                                                                                                                                       │
│ --install-completion                             Install completion for the current shell.                                                                                                                                                                                                                                                                                                                                           │
│ --show-completion                                Show completion for the current shell, to copy it or customize the installation.                                                                                                                                                                                                                                                                                                    │
│ --help                                           Show this message and exit.                                                                                                                                                                                                                                                                                                                                                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

An example that takes the original Excel files from NAACL 2015 Chinese data and outputs a combined CSV file:

``` bash
uv run scripts/naacl_2015_excel_to_csv.py ./Data/naacl2015_chinese_data -o ./tests/data/parsers/naacl_2015_chinese/naacl_2015_chinese_corpus.csv
```

An example that takes the original Excel files from NAACL 2015 Italian data and outputs a combined CSV file:

``` bash
uv run scripts/naacl_2015_excel_to_csv.py ./Data/naacl2015_italian_data -l italian -o ./tests/data/parsers/naacl_2015_italian/naacl_2015_italian_corpus.csv
```

## License

The code is licensed under [Apache License Version 2.0](./LICENSE).

The following data files, that we use for testing, are licensed under [Creative Commons Attribution Non Commercial Share Alike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en); 
* [./tests/data/parsers/benedict/english/benedict_english_corpus.txt](./tests/data/parsers/benedict/english/benedict_english_corpus.txt)
* [./tests/data/parsers/benedict/finnish/benedict_finnish_corpus.txt](./tests/data/parsers/benedict/finnish/benedict_finnish_corpus.txt)
* [./tests/data/parsers/torch/torch_corpus.csv](./tests/data/parsers/torch/torch_corpus.csv)
* [./tests/data/parsers/corcencc/corcencc_corpus.txt](./tests/data/parsers/corcencc/corcencc_corpus.txt)

The following data files, that we use for testing, are licensed under [Creative Commons Attribution Share Alike 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en)
* [./tests/data/parsers/spanish_wikipedia/spanish_wikipedia_corpus.csv](./tests/data/parsers/spanish_wikipedia/spanish_wikipedia_corpus.csv)
* [./tests/data/parsers/english_wikipedia/english_wikipedia_corpus.csv](./tests/data/parsers/english_wikipedia/english_wikipedia_corpus.csv)
* [./tests/data/parsers/dutch_wikipedia/dutch_wikipedia_corpus.csv](./tests/data/parsers/dutch_wikipedia/dutch_wikipedia_corpus.csv)
* [./tests/data/parsers/danish_wikipedia/danish_wikipedia_corpus.csv](./tests/data/parsers/danish_wikipedia/danish_wikipedia_corpus.csv)
* [./tests/data/parsers/hindi_wikipedia/hindi_wikipedia_corpus.csv](./tests/data/parsers/hindi_wikipedia/hindi_wikipedia_corpus.csv)

The following data files currently have no or more specifically unknown license:
* [./tests/data/parsers/naacl_2015_chinese/naacl_2015_chinese_corpus.csv](./tests/data/parsers/naacl_2015_chinese/naacl_2015_chinese_corpus.csv) 
* [./tests/data/parsers/naacl_2015_italian/naacl_2015_italian_corpus.csv](./tests/data/parsers/naacl_2015_italian/naacl_2015_italian_corpus.csv) 
