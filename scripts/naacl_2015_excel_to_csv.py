"""Build a sentence-structured CSV from a folder of annotated Excel files."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

REQUIRED_COLUMNS = {"token", "predicted_usas", "corrected_usas"}

_LANGUAGE_CONFIG: dict[str, dict] = {
    "chinese": {
        "sentence_endings": {"。"},
        "extra_columns": [],
    },
    "italian": {
        "sentence_endings": {"."},
        "extra_columns": ["mwe"],
    },
}

app = typer.Typer(help="Build a sentence-structured CSV from annotated Excel files.")


class Language(str, Enum):
    chinese = "chinese"
    italian = "italian"


def _process_file(
    excel_file: Path,
    sentence_endings: set[str],
    extra_columns: list[str],
) -> list[dict]:
    needed = REQUIRED_COLUMNS | set(extra_columns)
    try:
        df = pd.read_excel(excel_file)
    except Exception as exc:
        typer.echo(f"  [SKIP] {excel_file.name}: {exc}", err=True)
        return []

    missing = needed - set(df.columns)
    if missing:
        typer.echo(f"  [SKIP] {excel_file.name}: missing columns {missing}", err=True)
        return []

    output_columns = ["token", "corrected_usas"] + extra_columns
    file_name = excel_file.stem
    rows: list[dict] = []
    sentence_count = 1
    token_count = 1

    for _, row in df.iterrows():
        token = row["token"]
        entry: dict = {"id": f"{file_name}|{sentence_count}|{token_count}"}
        for col in output_columns:
            entry[col] = row[col]
        rows.append(entry)

        if str(token) in sentence_endings:
            rows.append({"id": "", **{col: "" for col in output_columns}})
            sentence_count += 1
            token_count = 1
        else:
            token_count += 1

    typer.echo(f"  [OK]   {excel_file.name}  ({sentence_count - 1} sentences)")
    return rows


@app.command()
def main(
    folder: Annotated[
        Path,
        typer.Argument(
            help="Folder containing Excel files to process.",
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
    language: Annotated[
        Language,
        typer.Option("--language", "-l", help="Language of the data (chinese or italian)."),
    ] = Language.chinese,
) -> None:
    """Build a sentence-structured CSV from a folder of annotated Excel files.

    Each Excel file must contain columns: token, predicted_usas, corrected_usas.
    For Italian data, a 'mwe' column is also required.

    The output CSV contains: id, token, corrected_usas[, mwe], where id is
    FILE_NAME|SENTENCE_COUNT|TOKEN_COUNT. A blank row is inserted after each
    sentence-ending token to delimit sentences.
    """
    excel_files = sorted(
        {f for pattern in ("*.xlsx", "*.xls") for f in folder.glob(pattern)}
    )

    if not excel_files:
        typer.echo(f"No Excel files found in '{folder}'.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(excel_files)} file(s).")

    config = _LANGUAGE_CONFIG[language.value]
    all_rows: list[dict] = []
    for excel_file in excel_files:
        all_rows.extend(
            _process_file(excel_file, config["sentence_endings"], config["extra_columns"])
        )

    if not all_rows:
        typer.echo("No data could be read from any Excel file.", err=True)
        raise typer.Exit(code=1)

    output_columns = ["id", "token", "corrected_usas"] + config["extra_columns"]
    output_df = pd.DataFrame(all_rows, columns=output_columns)
    output_df.to_csv(output, index=False)
    typer.echo(f"\nWrote {len(output_df)} rows to '{output}'")


if __name__ == "__main__":
    app()
