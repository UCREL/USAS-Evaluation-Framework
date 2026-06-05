"""
Build a sentence-structured CSV from a folder of annotated Excel files.

Each Excel file must contain columns: token, predicted_usas, corrected_usas

The output CSV contains: id, token, corrected_usas
  - id format: FILE_NAME|SENTENCE_COUNT|TOKEN_COUNT
  - A blank separator row is inserted after each Chinese full stop (。)
  - Sentence and token counts reset per file

Usage:
    python scripts/build_sentence_csv.py <input_folder> <output_csv>
"""

import sys
from pathlib import Path

import pandas as pd

CHINESE_FULL_STOP = "。"
REQUIRED_COLUMNS = {"token", "predicted_usas", "corrected_usas"}


def process_folder(input_folder: str, output_path: str) -> None:
    input_dir = Path(input_folder)
    excel_files = sorted(
        {f for pattern in ("*.xlsx", "*.xls") for f in input_dir.glob(pattern)}
    )

    if not excel_files:
        print(f"No Excel files found in '{input_folder}'")
        sys.exit(1)

    print(f"Found {len(excel_files)} file(s).")
    all_rows: list[dict] = []

    for excel_file in excel_files:
        file_name = excel_file.stem

        try:
            df = pd.read_excel(excel_file)
        except Exception as exc:
            print(f"  [SKIP] {excel_file.name}: {exc}")
            continue

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            print(f"  [SKIP] {excel_file.name}: missing columns {missing}")
            continue

        sentence_count = 1
        token_count = 1

        for _, row in df.iterrows():
            token = row["token"]
            corrected_usas = row["corrected_usas"]

            all_rows.append({
                "id": f"{file_name}|{sentence_count}|{token_count}",
                "token": token,
                "corrected_usas": corrected_usas,
            })

            if str(token) == CHINESE_FULL_STOP:
                all_rows.append({"id": "", "token": "", "corrected_usas": ""})
                sentence_count += 1
                token_count = 1
            else:
                token_count += 1

        print(f"  [OK]   {excel_file.name}  ({sentence_count - 1} sentences)")

    output_df = pd.DataFrame(all_rows, columns=["id", "token", "corrected_usas"])
    output_df.to_csv(output_path, index=False)
    print(f"\nWrote {len(output_df)} rows to '{output_path}'")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/build_sentence_csv.py <input_folder> <output_csv>")
        sys.exit(1)

    process_folder(sys.argv[1], sys.argv[2])
