"""Build a sentence-structured CSV from a folder of annotated Excel files."""

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

CHINESE_FULL_STOP = "。"
REQUIRED_COLUMNS = {"token", "predicted_usas", "corrected_usas"}

app = typer.Typer(help="Build a sentence-structured CSV from annotated Excel files.")


def _process_file(excel_file: Path) -> list[dict]:
    try:
        df = pd.read_excel(excel_file)
    except Exception as exc:
        typer.echo(f"  [SKIP] {excel_file.name}: {exc}", err=True)
        return []

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        typer.echo(f"  [SKIP] {excel_file.name}: missing columns {missing}", err=True)
        return []

    file_name = excel_file.stem
    rows: list[dict] = []
    sentence_count = 1
    token_count = 1

    for _, row in df.iterrows():
        token = row["token"]
        rows.append({
            "id": f"{file_name}|{sentence_count}|{token_count}",
            "token": token,
            "corrected_usas": row["corrected_usas"],
        })

        if str(token) == CHINESE_FULL_STOP:
            rows.append({"id": "", "token": "", "corrected_usas": ""})
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
) -> None:
    """Build a sentence-structured CSV from a folder of annotated Excel files.

    Each Excel file must contain columns: token, predicted_usas, corrected_usas.
    The output CSV contains: id, token, corrected_usas, where id is
    FILE_NAME|SENTENCE_COUNT|TOKEN_COUNT. A blank row is inserted after each
    Chinese full stop (。) to delimit sentences.
    """
    excel_files = sorted(
        {f for pattern in ("*.xlsx", "*.xls") for f in folder.glob(pattern)}
    )

    if not excel_files:
        typer.echo(f"No Excel files found in '{folder}'.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(excel_files)} file(s).")

    all_rows: list[dict] = []
    for excel_file in excel_files:
        all_rows.extend(_process_file(excel_file))

    if not all_rows:
        typer.echo("No data could be read from any Excel file.", err=True)
        raise typer.Exit(code=1)

    output_df = pd.DataFrame(all_rows, columns=["id", "token", "corrected_usas"])
    output_df.to_csv(output, index=False)
    typer.echo(f"\nWrote {len(output_df)} rows to '{output}'")


if __name__ == "__main__":
    app()
