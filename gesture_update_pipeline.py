import os
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys

# === CONFIGURATION ===
BASE_DIR = Path(__file__).resolve().parent
CODE_DIR = BASE_DIR / "code"
STANDARD_CSV = BASE_DIR / "training_results/gesture_data_compact.csv"
TRAIN_SCRIPT = CODE_DIR / "train_motion_svm_all_models.py"

AUGMENT_PER_SAMPLE = 20  # 5 samples -> 100 samples
MOTION_NOISE = 0.02
DELTA_SCALE_RANGE = 0.2

def augment_user_data(user_csv_path, target_samples=100):
    """
    Augment user's 5 samples to target_samples (default 100)
    
    Args:
        user_csv_path: Path to user's custom CSV file
        target_samples: Target number of samples after augmentation
    
    Returns:
        augmented_df: DataFrame with augmented samples
    """
    print(f"Augmenting {user_csv_path}...")
    
    # Load user data
    user_df = pd.read_csv(user_csv_path)
    original_count = len(user_df)
    
    if original_count == 0:
        raise ValueError("No data found in user CSV")
    
    # Calculate augmentation needed
    aug_per_sample = target_samples // original_count
    print(f"   Original samples: {original_count}")
    print(f"   Target samples: {target_samples}")
    print(f"   Augmentation per sample: {aug_per_sample}")
    
    # Prepare augmented data
    augmented_samples = []
    
    for idx, row in user_df.iterrows():
        # Keep original sample
        augmented_samples.append(row.to_dict())
        
        # Generate augmented samples
        for aug_idx in range(aug_per_sample - 1):
            augmented_sample = augment_single_sample(row, aug_idx + 1)
            augmented_samples.append(augmented_sample)
    
    # Create augmented DataFrame
    augmented_df = pd.DataFrame(augmented_samples)
    
    # Update instance_id
    augmented_df['instance_id'] = range(1, len(augmented_df) + 1)
    
    print(f"Augmentation complete: {len(augmented_df)} samples")
    return augmented_df


def augment_single_sample(original_row, aug_idx):
    """
    Create one augmented sample from original sample
    
    Args:
        original_row: Original pandas Series
        aug_idx: Augmentation index for variation
    
    Returns:
        dict: Augmented sample data
    """
    np.random.seed(42 + aug_idx)  # Reproducible but varied
    
    augmented = original_row.to_dict()
    
    # 1. Finger states: 90% exact, 10% with 1-finger error
    finger_error_chance = 0.1
    if np.random.random() < finger_error_chance:
        # Add 1-finger error
        finger_indices = list(range(5))  # right hand fingers 0-4
        error_finger = np.random.choice(finger_indices)
        finger_col = f'right_finger_state_{error_finger}'
        
        if finger_col in augmented:
            # Flip finger state
            augmented[finger_col] = 1 - augmented[finger_col]
    
    # 2. Motion coordinates: Add small noise
    motion_cols = [
        'motion_x_start', 'motion_y_start',
        'motion_x_mid', 'motion_y_mid', 
        'motion_x_end', 'motion_y_end'
    ]
    
    for col in motion_cols:
        if col in augmented:
            noise = np.random.normal(0, MOTION_NOISE)
            augmented[col] += noise
            # Keep in valid range [0, 1]
            augmented[col] = np.clip(augmented[col], 0, 1)
    
    # 3. Delta values: Add scale variation
    delta_cols = ['delta_x', 'delta_y']
    for col in delta_cols:
        if col in augmented:
            scale_factor = 1 + np.random.uniform(-DELTA_SCALE_RANGE, DELTA_SCALE_RANGE)
            augmented[col] *= scale_factor
    
    # 4. Main axis: Slight rotation
    if 'main_axis_x' in augmented and 'main_axis_y' in augmented:
        angle_noise = np.random.uniform(-0.1, 0.1)  # Small rotation
        cos_noise, sin_noise = np.cos(angle_noise), np.sin(angle_noise)
        
        old_x = augmented['main_axis_x']
        old_y = augmented['main_axis_y']
        
        augmented['main_axis_x'] = old_x * cos_noise - old_y * sin_noise
        augmented['main_axis_y'] = old_x * sin_noise + old_y * cos_noise
    
    return augmented


def combine_datasets(standard_csv, augmented_user_df, output_csv):
    """
    Combine standard dataset with augmented user data
    
    Args:
        standard_csv: Path to standard gesture dataset
        augmented_user_df: Augmented user DataFrame
        output_csv: Output path for combined dataset
    
    Returns:
        combined_df: Combined DataFrame
    """
    print(f"Combining datasets...")
    
    # Load standard dataset
    standard_df = pd.read_csv(standard_csv)
    print(f"   Standard dataset: {len(standard_df)} samples")
    print(f"   User dataset: {len(augmented_user_df)} samples")
    
    # Combine datasets
    combined_df = pd.concat([standard_df, augmented_user_df], ignore_index=True)
    
    # Update instance_id
    combined_df['instance_id'] = range(1, len(combined_df) + 1)
    
    # Save combined dataset
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined dataset saved: {output_csv}")
    print(f"   Total samples: {len(combined_df)}")
    
    # Show gesture distribution
    gesture_counts = combined_df['pose_label'].value_counts()
    print("\nGesture distribution:")
    for gesture, count in gesture_counts.items():
        print(f"   {gesture}: {count} samples")
    
    return combined_df


def retrain_models(combined_csv, output_dir="models"):
    """
    Retrain all models with combined dataset using existing train script
    
    Args:
        combined_csv: Path to combined dataset
        output_dir: Output directory for trained models (will use script's default)
    """
    print(f"Retraining models with combined dataset...")
    
    # Use the existing train script with --dataset parameter
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--dataset", str(combined_csv)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Model training completed successfully!")
        print("📁 New models saved in:", output_dir)
        
        # List generated model files
        model_files = list(Path(output_dir).glob("*.pkl"))
        if model_files:
            print("\n🎯 Generated model files:")
            for model_file in model_files:
                print(f"   {model_file.name}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ Model training failed!")
        print("Error output:", e.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ Training script not found: {TRAIN_SCRIPT}")
        return False


def process_user_gesture_update(user_csv_path, gesture_name=None):
    """
    Complete pipeline: Augment user data -> Combine -> Retrain models
    
    Args:
        user_csv_path: Path to user's custom CSV file
        gesture_name: Name of updated gesture (for logging)
    
    Returns:
        bool: Success status
    """
    try:
        user_csv_path = Path(user_csv_path)
        
        if not user_csv_path.exists():
            raise FileNotFoundError(f"User CSV not found: {user_csv_path}")
        
        # Extract user name from filename
        user_name = user_csv_path.stem.replace('gesture_data_custom_', '')
        
        print(f"🚀 Processing gesture update for user: {user_name}")
        if gesture_name:
            print(f"   Updated gesture: {gesture_name}")
        print(f"   User data: {user_csv_path}")
        
        # Step 1: Augment user data (5 -> 100 samples)
        augmented_df = augment_user_data(user_csv_path, target_samples=100)
        
        # Step 2: Combine with standard dataset
        combined_csv = BASE_DIR / f"gesture_data_combined_{user_name}.csv"
        combined_df = combine_datasets(STANDARD_CSV, augmented_df, combined_csv)
        
        # Step 3: Retrain models
        model_output_dir = BASE_DIR / f"models_{user_name}"
        success = retrain_models(combined_csv, model_output_dir)
        
        if success:
            print(f"\n🎉 Gesture update pipeline completed successfully!")
            print(f"📂 User-specific models: {model_output_dir}")
            print(f"📊 Combined dataset: {combined_csv}")
            
            # Create summary
            create_update_summary(user_name, gesture_name, combined_df, model_output_dir)
            
            return True
        else:
            print(f"\n❌ Pipeline failed during model training")
            return False
            
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        return False


def create_update_summary(user_name, gesture_name, combined_df, model_dir):
    """Create summary report of the update process"""
    
    summary_file = BASE_DIR / f"update_summary_{user_name}.txt"
    
    with open(summary_file, 'w') as f:
        f.write(f"=== GESTURE UPDATE SUMMARY ===\n")
        f.write(f"User: {user_name}\n")
        f.write(f"Updated Gesture: {gesture_name or 'Unknown'}\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n\n")
        
        f.write(f"Dataset Information:\n")
        f.write(f"- Total samples: {len(combined_df)}\n")
        f.write(f"- Unique gestures: {combined_df['pose_label'].nunique()}\n\n")
        
        f.write(f"Gesture Distribution:\n")
        gesture_counts = combined_df['pose_label'].value_counts()
        for gesture, count in gesture_counts.items():
            f.write(f"- {gesture}: {count} samples\n")
        
        f.write(f"\nModel Files:\n")
        model_files = list(Path(model_dir).glob("*.pkl"))
        for model_file in model_files:
            f.write(f"- {model_file.name}\n")
    
    print(f"📋 Summary saved: {summary_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process gesture update: Augment -> Combine -> Retrain")
    parser.add_argument("--user-csv", required=True, help="Path to user's custom CSV file")
    parser.add_argument("--gesture", help="Name of updated gesture")
    
    args = parser.parse_args()
    
    success = process_user_gesture_update(args.user_csv, args.gesture)
    exit(0 if success else 1)