from __future__ import annotations

import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_NAME = "grid_results_summary.csv"
PATTERN = "grid_results_*.csv"


def extract_metadata(path: Path) -> tuple[str, str]:
    """Return (resolution, target) parsed from file name."""
    parts = path.stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected grid result file name: {path.name}")
    resolution = parts[2]
    target = "_".join(parts[3:])
    return resolution, target


def main() -> None:
    csv_paths = sorted(
        p for p in BASE_DIR.glob(PATTERN) if p.name != OUTPUT_NAME
    )
    if not csv_paths:
        raise SystemExit("No grid_results CSV files found.")

    combined_rows: list[dict[str, str]] = []
    base_fieldnames: list[str] | None = None

    for path in csv_paths:
        resolution, target = extract_metadata(path)
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                continue

            if base_fieldnames is None:
                base_fieldnames = reader.fieldnames
            elif reader.fieldnames != base_fieldnames:
                raise ValueError(
                    "All grid_results files must share the same columns; "
                    f"{path.name} differs from {csv_paths[0].name}."
                )

            for row in reader:
                combined_row = {
                    "grid_resolution": resolution,
                    "target_label": target,
                }
                combined_row.update(row)
                combined_rows.append(combined_row)

    if not combined_rows:
        raise SystemExit("No rows found across grid_results CSV files.")

    def sort_key(item: dict[str, str]) -> tuple[str, str, float]:
        rank_str = item.get("rank_test_score", "")
        try:
            rank = float(rank_str)
        except (TypeError, ValueError):
            rank = float("inf")
        return (item["grid_resolution"], item["target_label"], rank)

    combined_rows.sort(key=sort_key)

    if base_fieldnames is None:
        raise SystemExit("Unable to determine column headers for grid results.")

    fieldnames = ["grid_resolution", "target_label"] + base_fieldnames

    output_path = BASE_DIR / OUTPUT_NAME
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"Wrote {len(combined_rows)} rows to {output_path.name}")


if __name__ == "__main__":
    main()
