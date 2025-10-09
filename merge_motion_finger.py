import argparse
import os
from pathlib import Path

import pandas as pd

DEFAULT_MOTION = 'gesture_motion_delta_main_axis_realistic.csv'
DEFAULT_FINGER = 'gesture_motion_finger_state_realistic.csv'
DEFAULT_OUTPUT = 'gesture_motion_dataset_merged.csv'


def merge_datasets(motion_path: Path, finger_path: Path, output_path: Path) -> None:
    if not motion_path.is_file():
        raise FileNotFoundError(f'Missing motion csv: {motion_path}')
    if not finger_path.is_file():
        raise FileNotFoundError(f'Missing finger csv: {finger_path}')

    motion_df = pd.read_csv(motion_path)
    finger_df = pd.read_csv(finger_path)

    common_cols = [col for col in ['instance_id', 'pose_label', 'base_instance_id', 'augmentation_index'] if col in motion_df.columns and col in finger_df.columns]
    if 'instance_id' not in common_cols:
        raise ValueError('Both CSV files must contain at least the column "instance_id" to merge.')

    merged = motion_df.merge(finger_df, on=common_cols, how='inner', suffixes=('', '_finger'))

    finger_cols = [col for col in merged.columns if col.startswith('left_finger_state_') or col.startswith('right_finger_state_')]
    if not finger_cols:
        raise ValueError('Merged dataset does not contain finger state columns.')

    motion_cols = [
        'main_axis_x', 'main_axis_y', 'delta_x', 'delta_y',
        'motion_x_start', 'motion_y_start', 'motion_x_mid', 'motion_y_mid',
        'motion_x_end', 'motion_y_end'
    ]
    motion_cols = [col for col in motion_cols if col in merged.columns]

    base_columns = [
        *(col for col in ['instance_id', 'base_instance_id', 'augmentation_index'] if col in merged.columns),
        'pose_label'
    ]

    ordered_cols = base_columns + sorted(finger_cols) + motion_cols
    remaining_cols = [col for col in merged.columns if col not in ordered_cols]
    ordered_cols += remaining_cols

    merged = merged[ordered_cols]
    merged.to_csv(output_path, index=False)

    print(f'Merged dataset saved to {output_path} ({len(merged)} rows).')


def parse_args():
    parser = argparse.ArgumentParser(description='Merge motion and finger CSVs into a single dataset file.')
    parser.add_argument('--motion', default=DEFAULT_MOTION, help='Path to motion CSV file.')
    parser.add_argument('--finger', default=DEFAULT_FINGER, help='Path to finger-state CSV file.')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Output CSV path.')
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    motion_path = (base_dir / args.motion).resolve()
    finger_path = (base_dir / args.finger).resolve()
    output_path = (base_dir / args.output).resolve()

    merge_datasets(motion_path, finger_path, output_path)


if __name__ == '__main__':
    main()
