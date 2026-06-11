"""Build a sentence-structured CSV from a folder of tab-separated text files."""

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

app = typer.Typer(help="Build a sentence-structured CSV from tab-separated text files.")

_SENTENCE_ENDING = "."


def _process_file(text_file: Path) -> list[dict]:
    file_name = text_file.stem
    rows: list[dict] = []
    sentence_count = 1
    token_count = 1

    with text_file.open(encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) > 3:
                typer.echo(
                    f"  [ERROR] {text_file.name} line {line_num}: "
                    f"{len(parts)} fields found (max 3)",
                    err=True,
                )
                raise typer.Exit(code=1)

            token = parts[0]
            tag = parts[1] if len(parts) >= 2 else ""
            mwe = parts[2] if len(parts) == 3 else ""

            # Skip the header row
            if line_num == 1 and token == "TOKEN" and tag == "TAG" and mwe == "MWE":
                continue

            rows.append(
                {
                    "id": f"{file_name}|{sentence_count}|{token_count}",
                    "token": token,
                    "corrected_usas": tag,
                    "mwe": mwe,
                }
            )

            if token == _SENTENCE_ENDING:
                rows.append({"id": "", "token": "", "corrected_usas": "", "mwe": ""})
                sentence_count += 1
                token_count = 1
            else:
                token_count += 1

    typer.echo(f"  [OK]   {text_file.name}  ({sentence_count - 1} sentences)")
    return rows


@app.command()
def main(
    folder: Annotated[
        Path,
        typer.Argument(
            help="Folder containing .txt files to process.",
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
) -> None:
    """Build a sentence-structured CSV from a folder of tab-separated text files.

    Each line in a text file may have 1–3 tab-separated values:

    \b
      TOKEN[<TAB>TAG[<TAB>MWE]]

    TOKEN is required. TAG is the USAS semantic tag. MWE is an index that groups
    tokens into the same multi-word expression when they share the same value.
    Lines with more than 3 fields cause an error.

    A token consisting of a single full stop (.) marks a sentence boundary; a
    blank row is inserted into the output after it to delimit sentences.

    The output CSV contains columns: id, token, corrected_usas, mwe, where id is
    FILE_STEM|SENTENCE_COUNT|TOKEN_COUNT.
    """
    text_files = sorted(folder.glob("*.txt"))

    if not text_files:
        typer.echo(f"No .txt files found in '{folder}'.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(text_files)} file(s).")

    all_rows: list[dict] = []
    for text_file in text_files:
        all_rows.extend(_process_file(text_file))

    if not all_rows:
        typer.echo("No data could be read from any text file.", err=True)
        raise typer.Exit(code=1)

    output_df = pd.DataFrame(all_rows, columns=["id", "token", "corrected_usas", "mwe"])
    output_df.to_csv(output, index=False)
    typer.echo(f"\nWrote {len(output_df)} rows to '{output}'")


if __name__ == "__main__":
    app()
