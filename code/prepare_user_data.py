#!/usr/bin/env python3
"""
Chuẩn bị dữ liệu huấn luyện tùy biến cho từng admin/user.

Luồng mới:
    python prepare_user_data.py --user-id 123 --custom-csv path/to.csv

Luồng cũ (tương thích):
    python prepare_user_data.py user_Khang
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_COMPACT = SCRIPT_DIR / "training_results" / "gesture_data_compact.csv"
DEFAULT_ORIGINAL_DATA = SCRIPT_DIR / "gesture_data_09_10_2025.csv"

CUSTOM_SAMPLES = 100
CUSTOM_ERROR_RATIO = 0.3
DEFAULT_SYNTH_SAMPLES = 80
RANDOM_SEED = 42


def resolve_user_path(args: argparse.Namespace) -> Path:
    """Xác định thư mục user sẽ chứa kết quả."""
    if args.user_dir:
        return Path(args.user_dir).resolve()
    if args.user_folder:
        return Path(args.user_folder).resolve()
    if args.user_id:
        base = Path(args.output_root).resolve() if args.output_root else SCRIPT_DIR
        return (base / f"user_{args.user_id}").resolve()
    raise ValueError("Cần cung cấp --user-dir, --user-id hoặc đối số user_folder (legacy).")


def ensure_custom_csv(user_path: Path, custom_csv: str | None) -> Path:
    """Đảm bảo có file dữ liệu custom và copy vào folder user nếu cần."""
    if custom_csv:
        src = Path(custom_csv).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Không tìm thấy file custom CSV: {src}")
        user_path.mkdir(parents=True, exist_ok=True)
        dest = user_path / src.name
        if dest != src:
            shutil.copy2(src, dest)
            print(f"[INFO] Đã copy file custom vào {dest}")
        else:
            print(f"[INFO] Sử dụng file custom có sẵn: {dest}")
        return dest

    candidates = sorted(user_path.glob("gesture_data_custom_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy file custom trong {user_path}. Cần file theo mẫu gesture_data_custom_*.csv"
        )
    print(f"[INFO] Phát hiện file custom: {candidates[0]}")
    return candidates[0]


def load_dataframe(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} không tồn tại: {path}")
    print(f"[LOAD] {label}: {path}")
    return pd.read_csv(path)


def create_compact_dataset(base_df: pd.DataFrame, user_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Ghép base + custom thành compact dataset (mỗi gesture 1 sample)."""
    user_gestures = set(user_df["pose_label"].unique())
    samples: list[pd.Series] = []

    print("\n[STEP] Tạo compact dataset...")
    for _, base_row in base_df.iterrows():
        gesture = base_row["pose_label"]
        if gesture in user_gestures:
            chosen = user_df[user_df["pose_label"] == gesture].iloc[0].copy()
            source = "CUSTOM"
        else:
            chosen = base_row.copy()
            source = "BASE"
        chosen["instance_id"] = len(samples) + 1
        samples.append(chosen)
        print(f"   [{source}] {gesture}")

    compact_df = pd.DataFrame(samples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compact_df.to_csv(out_path, index=False)
    print(f"[SAVED] Compact dataset -> {out_path}")
    return compact_df


def add_noise(row: pd.Series, error_mode: bool) -> pd.Series:
    """Thêm nhiễu để mô phỏng lỗi khi người dùng thực hiện gesture."""
    noisy = row.copy()
    if error_mode:
        if np.random.random() < 0.5:
            finger_cols = [f"right_finger_state_{i}" for i in range(5)]
            for _ in range(np.random.choice([1, 2])):
                col = np.random.choice(finger_cols)
                noisy[col] = 1 - noisy[col]

        for col in ("delta_x", "delta_y"):
            if col in noisy:
                noisy[col] += np.random.normal(0, 0.05)

        if np.random.random() < 0.2:
            if "main_axis_x" in noisy:
                noisy["main_axis_x"] = 1 - noisy["main_axis_x"]
            if "main_axis_y" in noisy:
                noisy["main_axis_y"] = 1 - noisy["main_axis_y"]
    else:
        motion_cols = [
            "motion_x_start",
            "motion_y_start",
            "motion_x_mid",
            "motion_y_mid",
            "motion_x_end",
            "motion_y_end",
            "delta_x",
            "delta_y",
        ]
        for col in motion_cols:
            if col in noisy:
                noisy[col] += np.random.normal(0, 0.008)
    return noisy


def original_samples_for_gesture(original_df: pd.DataFrame | None, gesture: str) -> Iterable[pd.Series]:
    if original_df is None:
        return []
    subset = original_df[original_df["pose_label"] == gesture]
    return subset.itertuples(index=False, name=None) if not subset.empty else []


def create_balanced_dataset(
    compact_df: pd.DataFrame,
    user_gestures: set[str],
    original_df: pd.DataFrame | None,
    out_path: Path,
) -> pd.DataFrame:
    print("\n[STEP] Tạo balanced dataset...")
    np.random.seed(RANDOM_SEED)
    rows: list[pd.Series] = []

    for gesture in compact_df["pose_label"].unique():
        template = compact_df[compact_df["pose_label"] == gesture].iloc[0]
        is_custom = gesture in user_gestures

        if not is_custom:
            originals = original_df[original_df["pose_label"] == gesture] if original_df is not None else None
            if originals is not None and not originals.empty:
                print(f"   [DEFAULT] {gesture}: dùng {len(originals)} mẫu gốc")
                for _, sample in originals.iterrows():
                    sample_copy = sample.copy()
                    sample_copy["instance_id"] = len(rows) + 1
                    rows.append(sample_copy)
                continue

        if is_custom:
            print(f"   [CUSTOM] {gesture}: sinh {CUSTOM_SAMPLES} mẫu (30% lỗi)")
            error_count = int(CUSTOM_SAMPLES * CUSTOM_ERROR_RATIO)
            for idx in range(CUSTOM_SAMPLES):
                has_error = idx < error_count
                new_row = add_noise(template, has_error)
                new_row["instance_id"] = len(rows) + 1
                rows.append(new_row)
        else:
            print(f"   [DEFAULT] {gesture}: không có dữ liệu gốc, sinh {DEFAULT_SYNTH_SAMPLES} mẫu")
            for _ in range(DEFAULT_SYNTH_SAMPLES):
                new_row = add_noise(template, False)
                new_row["instance_id"] = len(rows) + 1
                rows.append(new_row)

    balanced_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(out_path, index=False)
    print(f"[SAVED] Balanced dataset -> {out_path} ({len(balanced_df)} mẫu)")
    return balanced_df


def run_training(balanced_file: Path, skip_training: bool) -> None:
    if skip_training:
        print("\n[TRAINING] Bỏ qua bước train (do dùng --skip-training).")
        return

    cmd = [sys.executable, str(SCRIPT_DIR / "train_user_models.py"), "--dataset", str(balanced_file)]
    print("\n[TRAINING] Chạy:", " ".join(cmd))
    print("=" * 60)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(SCRIPT_DIR),
        bufsize=1,
    )

    logs: list[str] = []
    while True:
        line = process.stdout.readline()
        if line == "" and process.poll() is not None:
            break
        if line:
            clean = line.rstrip()
            print(clean)
            logs.append(clean)

    code = process.poll()
    print("=" * 60)
    if code != 0:
        print(f"[ERROR] Train thất bại, exit code {code}")
        print(f"[HINT] Tự chạy lại: python train_user_models.py --dataset \"{balanced_file}\"")
        return

    summary = [l for l in logs if "F1-score" in l or "accuracy" in l]
    if summary:
        print("\n[SUMMARY]")
        for item in summary:
            print("   " + item)
    print("[SUCCESS] Train hoàn tất.")


def prepare_user_training(args: argparse.Namespace) -> bool:
    try:
        user_path = resolve_user_path(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return False

    user_path.mkdir(parents=True, exist_ok=True)
    try:
        custom_csv = ensure_custom_csv(user_path, args.custom_csv)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return False

    base_path = Path(args.base_compact).resolve() if args.base_compact else DEFAULT_BASE_COMPACT
    original_path = Path(args.original_data).resolve() if args.original_data else DEFAULT_ORIGINAL_DATA

    try:
        base_df = load_dataframe(base_path, "Base compact dataset")
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return False

    try:
        user_df = load_dataframe(custom_csv, "Custom dataset")
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return False

    if original_path.exists():
        original_df = pd.read_csv(original_path)
        print(f"[INFO] Original dataset: {len(original_df)} mẫu từ {original_path}")
    else:
        print(f"[WARN] Không tìm thấy original dataset ({original_path}). Sẽ sinh dữ liệu bằng noise.")
        original_df = None

    compact_file = user_path / "training_results" / "gesture_data_compact.csv"
    balanced_file = user_path / "gesture_data_1000_balanced.csv"

    compact_df = create_compact_dataset(base_df, user_df, compact_file)
    balanced_df = create_balanced_dataset(compact_df, set(user_df["pose_label"].unique()), original_df, balanced_file)

    run_training(balanced_file, args.skip_training)

    print("\n[DONE]")
    print(f"   Compact : {compact_file} ({len(compact_df)} dòng)")
    print(f"   Balanced: {balanced_file} ({len(balanced_df)} dòng)")
    print(f"   Models  : {user_path / 'models'}")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu và train model cho user gesture.")
    parser.add_argument(
        "user_folder",
        nargs="?",
        help="(Legacy) thư mục user sẵn có (ví dụ user_Bi).",
    )
    parser.add_argument("--user-id", help="ID user/admin (sẽ tạo folder user_<id>).")
    parser.add_argument("--user-dir", help="Đường dẫn tuyệt đối tới thư mục user.")
    parser.add_argument("--output-root", help="Thư mục cha để tạo user_<id> nếu dùng --user-id.")
    parser.add_argument("--custom-csv", help="Đường dẫn file CSV custom vừa thu thập.")
    parser.add_argument("--base-compact", help="Đường dẫn file compact gốc.")
    parser.add_argument("--original-data", help="Đường dẫn dataset mặc định đầy đủ.")
    parser.add_argument("--skip-training", action="store_true", help="Chỉ tạo dữ liệu, không chạy train.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    success = prepare_user_training(args)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
