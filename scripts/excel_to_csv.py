"""Script to parse all Excel files in a folder into a single CSV file."""

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

app = typer.Typer(help="Parse all Excel files in a folder into a single CSV file.")


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
) -> None:
    """Parse all Excel files in FOLDER into a single CSV file.

    Args:
        folder: Folder containing Excel files to parse.
        output: Path for the output CSV file. Defaults to output.csv.
        sheet: Sheet name or 0-based index to read. Defaults to the first sheet.
        add_source: Add a 'source_file' column with the originating filename.
        recursive: Search subdirectories recursively.

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
    combined.to_csv(output, index=False)

    typer.echo(f"\nWrote {len(combined)} rows to '{output}'.")


if __name__ == "__main__":
    app()
