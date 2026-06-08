"""Script to parse all Excel files in a folder into a single CSV file.

Expected input columns (Excel files)
--------------------------------------
All columns are passed through unchanged. The following column names carry
special meaning and must be present for the relevant options to take effect:

``id``
    Token identifier used to derive sentence and token counts in the summary
    statistics.  Expected format: ``<sentence_key>|<token_index>`` (e.g.
    ``doc1|3``).  Rows whose ``id`` cell is empty are treated as non-token rows
    (e.g. blank separator rows) and excluded from counts.

``POS``
    Part-of-speech tag for each token.  Used by ``--punct-to-z9``: rows where
    ``POS == 'PUNCT'`` may trigger a Z9 substitution in ``corrected USAS``.

``predicted USAS``
    USAS semantic tag produced by the tagger.  Used by ``--punct-to-z9``: rows
    where ``predicted USAS == 'PUNCT'`` may trigger a Z9 substitution in
    ``corrected USAS``.

``corrected USAS``
    Human-corrected USAS semantic tag.  Reported in summary statistics and
    modified in-place by ``--punct-to-z9``.

Output CSV columns
------------------
``source_file`` *(optional)*
    Inserted as the first column when ``--add-source`` is supplied.  Contains
    the filename (not the full path) of the Excel file the row originated from.

All remaining columns are the union of columns from every processed Excel file,
concatenated in the order the files are read.  Column values are unchanged
except that ``corrected USAS`` may be set to ``Z9`` when ``--punct-to-z9`` is
active (see ``--help`` for the exact substitution rules).
"""

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

app = typer.Typer(help="Parse all Excel files in a folder into a single CSV file.")


def _fix_punct_to_z9(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace PUNCT-derived tags in the corrected USAS column with Z9.

    Conditions (per token row):
    - ``corrected USAS`` equals ``'PUNCT'``, or
    - ``corrected USAS`` is empty AND (``predicted USAS`` or ``POS`` equals ``'PUNCT'``).

    Returns the updated DataFrame and the count of cells changed.
    """
    if "corrected USAS" not in df.columns:
        return df, 0

    corrected = df["corrected USAS"].fillna("").astype(str).str.strip()
    predicted = (
        df["predicted USAS"].fillna("").astype(str).str.strip()
        if "predicted USAS" in df.columns
        else pd.Series("", index=df.index)
    )
    pos = (
        df["POS"].fillna("").astype(str).str.strip()
        if "POS" in df.columns
        else pd.Series("", index=df.index)
    )

    mask = (corrected == "PUNCT") | (
        (corrected == "") & ((predicted == "PUNCT") | (pos == "PUNCT"))
    )

    df = df.copy()
    df.loc[mask, "corrected USAS"] = "Z9"
    return df, int(mask.sum())


def _print_stats(df: pd.DataFrame) -> None:
    """Echo sentence, token, and corrected-USAS-tag counts from the combined DataFrame."""
    typer.echo("\nDataset summary:")

    if "id" in df.columns:
        token_mask = df["id"].notna() & (df["id"].astype(str).str.strip() != "")
        n_tokens = int(token_mask.sum())
        sentence_keys = (
            df.loc[token_mask, "id"]
            .astype(str)
            .str.rsplit("|", n=1)
            .str[0]
        )
        n_sentences = int(sentence_keys.nunique())
        typer.echo(f"  Sentences:           {n_sentences:,}")
        typer.echo(f"  Tokens:              {n_tokens:,}")
    else:
        typer.echo(f"  Tokens:              {len(df):,}")

    if "corrected USAS" in df.columns:
        n_corrected = int(
            (df["corrected USAS"].fillna("").astype(str).str.strip() != "").sum()
        )
        typer.echo(f"  Corrected USAS tags: {n_corrected:,}")


def _find_excel_files(folder: Path, recursive: bool) -> list[Path]:
    """Find all Excel files in a folder.

    Args:
        folder: Path to the folder to search.
        recursive: Whether to search subdirectories recursively.

    Returns:
        list[Path]: Sorted list of Excel file paths found.
    """
    patterns = ("*.xlsx", "*.xls", "*.xlsm", "*.xlsb")
    if recursive:
        files = [f for pattern in patterns for f in folder.rglob(pattern)]
    else:
        files = [f for pattern in patterns for f in folder.glob(pattern)]
    return sorted(set(files))


@app.command()
def main(
    folder: Annotated[
        Path,
        typer.Argument(
            help="Folder containing Excel files to parse.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path for the output CSV file."),
    ] = Path("output.csv"),
    sheet: Annotated[
        str | None,
        typer.Option(
            "--sheet",
            "-s",
            help="Sheet name or 0-based index to read from each file. Defaults to the first sheet.",
        ),
    ] = None,
    add_source: Annotated[
        bool,
        typer.Option(
            "--add-source/--no-add-source",
            help="Add a 'source_file' column with the originating filename.",
        ),
    ] = False,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive/--no-recursive",
            "-r",
            help="Search for Excel files recursively in subdirectories.",
        ),
    ] = False,
    punct_to_z9: Annotated[
        bool,
        typer.Option(
            "--punct-to-z9/--no-punct-to-z9",
            help=(
                "Replace PUNCT tags in 'corrected USAS' with Z9. "
                "Also fills empty 'corrected USAS' cells with Z9 when "
                "'predicted USAS' or 'POS' is PUNCT."
            ),
        ),
    ] = False,
) -> None:
    """Parse all Excel files in FOLDER into a single CSV file.

    Args:
        folder: Folder containing Excel files to parse.
        output: Path for the output CSV file. Defaults to output.csv.
        sheet: Sheet name or 0-based index to read. Defaults to the first sheet.
        add_source: Add a 'source_file' column with the originating filename.
        recursive: Search subdirectories recursively.
        punct_to_z9: Replace PUNCT tags with Z9 in the corrected USAS column.

    Raises:
        typer.Exit: If no Excel files are found or a file cannot be read.
    """
    excel_files = _find_excel_files(folder, recursive)

    if not excel_files:
        typer.echo(f"No Excel files found in '{folder}'.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(excel_files)} Excel file(s).")

    sheet_arg = int(sheet) if sheet is not None and sheet.isdigit() else sheet
    sheet_to_read = 0 if sheet_arg is None else sheet_arg

    frames: list[pd.DataFrame] = []
    for path in excel_files:
        try:
            df = pd.read_excel(path, sheet_name=sheet_to_read)
        except Exception as exc:
            typer.echo(f"  [SKIP] {path.name}: {exc}", err=True)
            continue

        if add_source:
            df.insert(0, "source_file", path.name)

        frames.append(df)
        typer.echo(f"  [OK]   {path.name}  ({len(df)} rows)")

    if not frames:
        typer.echo("No data could be read from any Excel file.", err=True)
        raise typer.Exit(code=1)

    combined = pd.concat(frames, ignore_index=True)

    if punct_to_z9:
        combined, n_changed = _fix_punct_to_z9(combined)
        typer.echo(f"\nReplaced PUNCT tags with Z9 in 'corrected USAS' ({n_changed} cell(s) updated).")

    combined.to_csv(output, index=False)

    typer.echo(f"\nWrote {len(combined)} rows to '{output}'.")
    _print_stats(combined)


if __name__ == "__main__":
    app()
