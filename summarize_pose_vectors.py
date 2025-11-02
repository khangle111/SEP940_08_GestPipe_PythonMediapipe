import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FINGER_LEFT_COLS = [f"left_finger_state_{i}" for i in range(5)]
FINGER_RIGHT_COLS = [f"right_finger_state_{i}" for i in range(5)]
FINGER_COLS = FINGER_LEFT_COLS + FINGER_RIGHT_COLS
DELTA_SMALL_THRESHOLD = 0.001
DELTA_FALLBACK_MAG = 0.0005
DEFAULT_OUTPUT_DIR = Path("training_results")
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "gesture_data_compact.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize dominant finger vectors per pose label and keep the most "
            "frequent combinations until a coverage threshold is met."
        )
    )
    parser.add_argument(
        "--input",
        default="gesture_data_09_10_2025.csv",
        help="CSV file produced by collect_data_hybrid.py (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help=(
            "Destination CSV for the condensed dataset "
            "matching the original schema "
            "(default: training_results/gesture_data_compact.csv)."
        ),
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.90,
        help="Minimum cumulative coverage (0-1) to keep combos (default: %(default)s).",
    )
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=14,
        help="Maximum number of combos to retain (default: %(default)s).",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help=(
            "Optional path to write an auxiliary summary CSV. "
            "If omitted, only the compact dataset is produced."
        ),
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Could not find dataset: {path}")
    df = pd.read_csv(path)
    missing = [
        col
        for col in [
            "pose_label",
            *FINGER_COLS,
            "delta_x",
            "delta_y",
            "main_axis_x",
            "main_axis_y",
            "motion_x_start",
            "motion_y_start",
            "motion_x_mid",
            "motion_y_mid",
            "motion_x_end",
            "motion_y_end",
        ]
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def dominant_delta(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0

    large_mask = np.abs(arr) >= DELTA_SMALL_THRESHOLD
    large_vals = arr[large_mask]

    def _sign_counts(data: np.ndarray) -> tuple[int, int]:
        pos = int(np.sum(data > 0))
        neg = int(np.sum(data < 0))
        return pos, neg

    pos, neg = _sign_counts(large_vals if large_vals.size else arr)
    if pos == neg == 0:
        return 0.0
    sign = 1.0 if pos >= neg else -1.0

    same_sign = arr[(arr * sign) > 0]
    same_sign_large = same_sign[np.abs(same_sign) >= DELTA_SMALL_THRESHOLD]
    if same_sign_large.size:
        magnitude = float(np.median(np.abs(same_sign_large)))
    else:
        magnitude = DELTA_FALLBACK_MAG
    magnitude = max(magnitude, DELTA_FALLBACK_MAG)
    return sign * magnitude


def summarize(df: pd.DataFrame, coverage: float, max_vectors: int) -> pd.DataFrame:
    group_cols = ["pose_label"] + FINGER_COLS
    motion_cols = [
        "motion_x_start",
        "motion_y_start",
        "motion_x_mid",
        "motion_y_mid",
        "motion_x_end",
        "motion_y_end",
    ]

    grouped = (
        df.groupby(group_cols)
        .agg(
            count=("pose_label", "size"),
            axis_x_sum=("main_axis_x", "sum"),
            axis_y_sum=("main_axis_y", "sum"),
            delta_x_values=("delta_x", list),
            delta_y_values=("delta_y", list),
            **{f"{col}_values": (col, list) for col in motion_cols},
        )
        .reset_index()
    )

    grouped = grouped.sort_values("count", ascending=False).reset_index(drop=True)
    total_samples = grouped["count"].sum()
    if total_samples == 0:
        raise ValueError("Dataset appears to be empty after grouping.")

    selected_rows = []
    cumulative = 0.0
    for row in grouped.itertuples(index=False):
        cumulative += row.count / total_samples
        selected_rows.append(row)
        if cumulative >= coverage or len(selected_rows) >= max_vectors:
            break

    records = []
    summary_rows = []
    cumulative_running = 0.0
    for idx, row in enumerate(selected_rows, start=1):
        left_vec = [getattr(row, col) for col in FINGER_LEFT_COLS]
        right_vec = [getattr(row, col) for col in FINGER_RIGHT_COLS]
        dominant_axis = "x" if row.axis_x_sum >= row.axis_y_sum else "y"
        dom_delta_x = dominant_delta(row.delta_x_values)
        dom_delta_y = dominant_delta(row.delta_y_values)
        frac = row.count / total_samples
        cumulative_running += frac

        main_axis_x = 1 if dominant_axis == "x" else 0
        main_axis_y = 1 if dominant_axis == "y" else 0

        def median_from(values: Iterable[float]) -> float:
            arr = np.asarray(list(values), dtype=float)
            if arr.size == 0:
                return 0.0
            return float(np.median(arr))

        record = {
            "instance_id": idx,
            "pose_label": row.pose_label,
        }
        record.update({col: int(value) for col, value in zip(FINGER_LEFT_COLS, left_vec)})
        record.update({col: int(value) for col, value in zip(FINGER_RIGHT_COLS, right_vec)})

        for col in motion_cols:
            values = getattr(row, f"{col}_values")
            record[col] = round(median_from(values), 6)

        record["main_axis_x"] = main_axis_x
        record["main_axis_y"] = main_axis_y
        record["delta_x"] = round(dom_delta_x, 6)
        record["delta_y"] = round(dom_delta_y, 6)
        records.append(record)

        summary_rows.append(
            {
                "pose_label": row.pose_label,
                "left_state": "".join(str(int(v)) for v in left_vec),
                "right_state": "".join(str(int(v)) for v in right_vec),
                "dominant_axis": dominant_axis,
                "dominant_delta_x": round(dom_delta_x, 6),
                "dominant_delta_y": round(dom_delta_y, 6),
                "count": int(row.count),
                "fraction": round(frac, 4),
                "cumulative_fraction": round(min(cumulative_running, 1.0), 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    compact_df = pd.DataFrame(records)

    if not compact_df.empty:
        # ensure column order matches original dataset
        ordered_cols = [
            "instance_id",
            "pose_label",
            *FINGER_LEFT_COLS,
            *FINGER_RIGHT_COLS,
            *motion_cols,
            "main_axis_x",
            "main_axis_y",
            "delta_x",
            "delta_y",
        ]
        compact_df = compact_df[ordered_cols]

    return compact_df, summary_df


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = load_dataset(input_path)
    compact_df, summary_df = summarize(df, coverage=args.coverage, max_vectors=args.max_vectors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compact_df.to_csv(output_path, index=False)
    print(f"Wrote compact dataset ({len(compact_df)} rows) to {output_path}")

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"Wrote summary ({len(summary_df)} rows) to {summary_path}")


if __name__ == "__main__":
    main()
