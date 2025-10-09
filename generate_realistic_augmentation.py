import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CAPTURE_SRC = BASE_DIR / "gesture_motion_capture.csv"
DATASET_OUT = BASE_DIR / "gesture_motion_dataset_realistic.csv"

FINGER_COLS = [
    "left_finger_state_0", "left_finger_state_1", "left_finger_state_2", 
    "left_finger_state_3", "left_finger_state_4",
    "right_finger_state_0", "right_finger_state_1", "right_finger_state_2", 
    "right_finger_state_3", "right_finger_state_4",
]
MOTION_POINT_COLS = [
    "motion_x_start", "motion_y_start", "motion_x_mid", 
    "motion_y_mid", "motion_x_end", "motion_y_end",
]
MOTION_COLS = ["main_axis_x", "main_axis_y", "delta_x", "delta_y"]
REQUIRED_COLS = ["instance_id", "pose_label", *FINGER_COLS, *MOTION_POINT_COLS, *MOTION_COLS]

# Realistic augmentation settings
EXACT_FINGER_RATIO = 0.9  # 90% samples with exact finger states
ONE_FINGER_ERROR_RATIO = 0.1  # 10% samples with 1-finger error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate realistic augmented dataset with proper finger state distribution."
    )
    parser.add_argument("--capture", type=Path, default=CAPTURE_SRC,
                      help="Path to the gesture_motion_capture.csv file.")
    parser.add_argument("--aug-per-sample", type=int, default=20,
                      help="Number of augmented samples per original sample.")
    parser.add_argument("--motion-noise", type=float, default=0.02,
                      help="Noise factor for motion coordinates.")
    parser.add_argument("--delta-scale-range", type=float, default=0.2,
                      help="Scale variation range for delta values (±).")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility.")
    return parser.parse_args()


def load_and_validate_capture(path: Path) -> pd.DataFrame:
    """Load and validate capture CSV"""
    if not path.is_file():
        raise FileNotFoundError(f"Capture CSV not found: {path}")
    
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Ensure finger columns are binary
    for col in FINGER_COLS:
        df[col] = df[col].fillna(0).astype(int)
        df[col] = df[col].clip(0, 1)  # Force binary
    
    return df


def extract_canonical_finger_states(df: pd.DataFrame):
    """Extract the most common finger state pattern for each pose"""
    canonical_states = {}
    
    for pose, pose_df in df.groupby("pose_label"):
        # Get all finger state patterns for this pose
        patterns = []
        for _, row in pose_df.iterrows():
            pattern = tuple(int(row[col]) for col in FINGER_COLS)
            patterns.append(pattern)
        
        # Find most common pattern
        counter = Counter(patterns)
        canonical_pattern, count = counter.most_common(1)[0]
        
        canonical_states[pose] = {
            'pattern': np.array(canonical_pattern, dtype=int),
            'frequency': count / len(patterns)
        }
        
        print(f"[INFO] Pose '{pose}': canonical pattern {canonical_pattern} "
              f"(appears {count}/{len(patterns)} = {count/len(patterns):.1%})")
    
    return canonical_states


def generate_realistic_finger_states(canonical_pattern: np.ndarray, 
                                   num_samples: int, 
                                   rng: np.random.Generator):
    """Generate realistic finger states with mostly exact patterns"""
    
    num_exact = int(num_samples * EXACT_FINGER_RATIO)
    num_one_error = num_samples - num_exact
    
    samples = []
    
    # 90% exact canonical patterns
    for _ in range(num_exact):
        samples.append(canonical_pattern.copy())
    
    # 10% with exactly 1 finger different
    for _ in range(num_one_error):
        pattern = canonical_pattern.copy()
        
        # Randomly flip exactly 1 finger
        finger_idx = rng.integers(0, len(pattern))
        pattern[finger_idx] = 1 - pattern[finger_idx]  # Flip 0->1 or 1->0
        
        samples.append(pattern)
    
    # Shuffle to mix exact and error samples
    rng.shuffle(samples)
    return samples


def augment_motion_realistically(original_motion: dict, 
                                motion_noise: float,
                                delta_scale_range: float,
                                rng: np.random.Generator):
    """Add realistic noise to motion data"""
    
    # Original coordinates
    start_x, start_y = original_motion['start']
    mid_x, mid_y = original_motion['mid'] 
    end_x, end_y = original_motion['end']
    delta_x, delta_y = original_motion['delta_x'], original_motion['delta_y']
    
    # Add small coordinate noise
    start_x += rng.normal(0, motion_noise)
    start_y += rng.normal(0, motion_noise)
    
    # Scale delta with variation
    scale_factor = 1.0 + rng.uniform(-delta_scale_range, delta_scale_range)
    delta_x *= scale_factor
    delta_y *= scale_factor
    
    # Recompute end point based on scaled delta
    end_x = start_x + delta_x
    end_y = start_y + delta_y
    mid_x = start_x + 0.5 * delta_x
    mid_y = start_y + 0.5 * delta_y
    
    # Clamp to [0, 1] range
    start_x = np.clip(start_x, 0.0, 1.0)
    start_y = np.clip(start_y, 0.0, 1.0)
    mid_x = np.clip(mid_x, 0.0, 1.0)
    mid_y = np.clip(mid_y, 0.0, 1.0)
    end_x = np.clip(end_x, 0.0, 1.0)
    end_y = np.clip(end_y, 0.0, 1.0)
    
    # Recalculate delta after clamping
    actual_delta_x = end_x - start_x
    actual_delta_y = end_y - start_y
    
    # Determine main axis
    if abs(actual_delta_x) >= abs(actual_delta_y):
        main_axis_x, main_axis_y = 1.0, 0.0
        final_delta_x = actual_delta_x
        final_delta_y = 0.0
    else:
        main_axis_x, main_axis_y = 0.0, 1.0
        final_delta_x = 0.0
        final_delta_y = actual_delta_y
    
    return {
        'motion_x_start': start_x,
        'motion_y_start': start_y,
        'motion_x_mid': mid_x,
        'motion_y_mid': mid_y,
        'motion_x_end': end_x,
        'motion_y_end': end_y,
        'main_axis_x': main_axis_x,
        'main_axis_y': main_axis_y,
        'delta_x': final_delta_x,
        'delta_y': final_delta_y,
    }


def generate_augmented_dataset(df: pd.DataFrame,
                             canonical_states: dict,
                             aug_per_sample: int,
                             motion_noise: float,
                             delta_scale_range: float,
                             seed: int):
    """Generate augmented dataset with realistic variations"""
    
    rng = np.random.default_rng(seed)
    augmented_rows = []
    next_instance_id = 1
    
    print(f"\n[INFO] Generating {aug_per_sample} samples per original...")
    print(f"[INFO] Finger state distribution: {EXACT_FINGER_RATIO:.0%} exact, {ONE_FINGER_ERROR_RATIO:.0%} 1-finger error")
    
    for _, row in df.iterrows():
        pose = row['pose_label']
        base_instance_id = int(row['instance_id'])
        canonical_pattern = canonical_states[pose]['pattern']
        
        # Original motion data
        original_motion = {
            'start': (row['motion_x_start'], row['motion_y_start']),
            'mid': (row['motion_x_mid'], row['motion_y_mid']),
            'end': (row['motion_x_end'], row['motion_y_end']),
            'delta_x': row['delta_x'],
            'delta_y': row['delta_y']
        }
        
        # Generate realistic finger states for this pose
        finger_samples = generate_realistic_finger_states(
            canonical_pattern, aug_per_sample, rng
        )
        
        # Create augmented samples
        for aug_idx, finger_states in enumerate(finger_samples):
            # Augment motion data
            motion_data = augment_motion_realistically(
                original_motion, motion_noise, delta_scale_range, rng
            )
            
            # Create row
            row_dict = {
                'instance_id': next_instance_id,
                'base_instance_id': base_instance_id,
                'augmentation_index': aug_idx + 1,
                'pose_label': pose,
                **{FINGER_COLS[i]: int(finger_states[i]) for i in range(len(FINGER_COLS))},
                **motion_data
            }
            
            augmented_rows.append(row_dict)
            next_instance_id += 1
    
    return augmented_rows


def main():
    args = parse_args()
    
    print("=== REALISTIC DATASET GENERATION ===")
    print(f"Input: {args.capture}")
    print(f"Output: {DATASET_OUT}")
    print(f"Augmentation: {args.aug_per_sample} samples per original")
    
    # Load and validate data
    df = load_and_validate_capture(args.capture)
    print(f"\n[INFO] Loaded {len(df)} original samples")
    
    # Extract canonical finger states for each pose
    canonical_states = extract_canonical_finger_states(df)
    
    # Generate augmented dataset
    augmented_rows = generate_augmented_dataset(
        df, canonical_states, args.aug_per_sample,
        args.motion_noise, args.delta_scale_range, args.seed
    )
    
    # Save results
    output_df = pd.DataFrame(augmented_rows)
    column_order = [
        "instance_id", "base_instance_id", "augmentation_index", "pose_label",
        *FINGER_COLS, *MOTION_POINT_COLS, *MOTION_COLS,
    ]
    output_df = output_df[column_order]
    output_df.to_csv(DATASET_OUT, index=False)
    
    # Summary
    print(f"\n=== GENERATION COMPLETE ===")
    print(f"Generated {len(output_df)} total samples")
    
    for pose in output_df['pose_label'].unique():
        pose_count = len(output_df[output_df['pose_label'] == pose])
        print(f"  {pose}: {pose_count} samples")
    
    print(f"\nSaved to: {DATASET_OUT}")
    
    # Finger state distribution analysis
    print(f"\n=== FINGER STATE VALIDATION ===")
    for pose in canonical_states.keys():
        pose_df = output_df[output_df['pose_label'] == pose]
        canonical = canonical_states[pose]['pattern']
        
        exact_matches = 0
        one_diff_matches = 0
        
        for _, row in pose_df.iterrows():
            current = np.array([int(row[col]) for col in FINGER_COLS])
            diff_count = np.sum(current != canonical)
            
            if diff_count == 0:
                exact_matches += 1
            elif diff_count == 1:
                one_diff_matches += 1
        
        total = len(pose_df)
        print(f"  {pose}: {exact_matches/total:.1%} exact, {one_diff_matches/total:.1%} 1-diff, "
              f"{(total-exact_matches-one_diff_matches)/total:.1%} other")


if __name__ == "__main__":
    main()