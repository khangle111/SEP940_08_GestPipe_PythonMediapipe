import os
import pandas as pd
import numpy as np
import subprocess
import sys
from pathlib import Path
import shutil

# === CONFIGURATION ===
STANDARD_CSV = "../training_results/gesture_data_compact.csv"
AUGMENT_PER_SAMPLE = 20  # 5 samples -> 100 samples
MOTION_NOISE = 0.02
DELTA_SCALE_RANGE = 0.2

def augment_user_data(user_csv_path, target_samples=100):
    """
    Augment user's 5 samples to target_samples (default 100)
    """
    print(f"Step 1: Augmenting {user_csv_path}...")
    
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
    
    print(f"   Augmentation complete: {len(augmented_df)} samples")
    return augmented_df

def augment_single_sample(original_row, aug_idx):
    """Create one augmented sample from original sample"""
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

def augment_standard_gesture(gesture_df, target_samples):
    """Augment standard gesture data using existing augmentation logic"""
    original_count = len(gesture_df)
    if original_count == 0:
        return gesture_df
        
    aug_per_sample = target_samples // original_count
    augmented_samples = []
    
    for idx, row in gesture_df.iterrows():
        # Keep original sample
        augmented_samples.append(row.to_dict())
        
        # Generate augmented samples  
        for aug_idx in range(aug_per_sample - 1):
            augmented_sample = augment_single_sample(row, aug_idx + 1)
            augmented_samples.append(augmented_sample)
    
    # Add extra samples if needed
    while len(augmented_samples) < target_samples:
        idx = len(augmented_samples) % original_count
        row = gesture_df.iloc[idx]
        augmented_sample = augment_single_sample(row, len(augmented_samples) + 1)
        augmented_samples.append(augmented_sample)
    
    # Create DataFrame
    result_df = pd.DataFrame(augmented_samples)
    return result_df

def create_balanced_dataset_with_user_data(user_csv_path, all_user_gestures, output_csv):
    """
    Create balanced dataset with exactly 1000 samples total (100 per gesture):
    - User gestures: Augment to 100 samples each (e.g., 2 gestures = 200 samples)
    - Standard gestures: 100 samples each for remaining gestures (e.g., 8 gestures = 800 samples)
    - Total: 10 gestures × 100 samples = 1000 samples
    """
    print(f"Step 2: Creating balanced dataset with user gestures...")
    print(f"   User gestures: {all_user_gestures}")
    
    # Load base dataset 
    base_df = pd.read_csv("gesture_data_09_10_2025.csv")
    print(f"   Base dataset: {len(base_df)} samples")
    
    # Load user data
    user_df = pd.read_csv(user_csv_path)
    user_sample_count = len(user_df)
    print(f"   User data: {user_sample_count} samples")
    
    # Get all 10 standard gesture types
    all_standard_gestures = ['home', 'end', 'next_slide', 'previous_slide', 'zoom_in', 'zoom_out', 
                           'rotate_left', 'rotate_up', 'rotate_down', 'rotate_right']
    
    # Augment user gestures to 100 samples each
    augmented_user_data = []
    for gesture in all_user_gestures:
        gesture_df = user_df[user_df['pose_label'] == gesture].copy()
        samples_count = len(gesture_df)
        print(f"   Augmenting user '{gesture}': {samples_count} -> 100 samples")
        
        # Save temp file for augmentation
        temp_csv = f"temp_{gesture}.csv" 
        gesture_df.to_csv(temp_csv, index=False)
        
        # Augment using existing function
        augmented_gesture_df = augment_user_data(temp_csv, target_samples=100)
        augmented_user_data.append(augmented_gesture_df)
        
        # Clean up temp file
        Path(temp_csv).unlink()
    
    # Get non-user gestures for standard data (100 samples each)
    non_user_gestures = [g for g in all_standard_gestures if g not in all_user_gestures]
    print(f"   Non-user gestures: {non_user_gestures} ({len(non_user_gestures)} types)")
    
    standard_samples = []
    for gesture in non_user_gestures:
        gesture_df = base_df[base_df['pose_label'] == gesture].copy()
        current_count = len(gesture_df)
        
        if current_count >= 100:
            # Sample down to 100
            sampled_df = gesture_df.sample(n=100, random_state=42)
        else:
            # Augment up to 100
            augmented_df = augment_standard_gesture(gesture_df, 100)
            sampled_df = augmented_df
            
        standard_samples.append(sampled_df)
        print(f"   Standard '{gesture}': {current_count} -> 100 samples")
    
    # Combine user data + standard data (each gesture = 100 samples)
    all_data = augmented_user_data + standard_samples
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Update instance_id
    combined_df['instance_id'] = range(1, len(combined_df) + 1)
    
    # Save combined dataset
    combined_df.to_csv(output_csv, index=False)
    print(f"   Final balanced dataset: {len(combined_df)} samples")
    
    # Show final distribution
    gesture_counts = combined_df['pose_label'].value_counts()
    print(f"\n   Final distribution:")
    for gesture, count in gesture_counts.items():
        marker = "🔄" if gesture in all_user_gestures else "📋"
        print(f"     {marker} {gesture}: {count} samples")
    
    return combined_df
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=".")
        print("   Balanced dataset created successfully!")
        
        # Load and show final distribution
        df = pd.read_csv(output_csv)
        gesture_counts = df['pose_label'].value_counts()
        print(f"\n   Final balanced dataset: {len(df)} samples")
        print("   Distribution:")
        for gesture, count in gesture_counts.items():
            print(f"     {gesture}: {count} samples")
        
        return df
        
    except subprocess.CalledProcessError as e:
        print("   Failed to create balanced dataset!")
        print(f"   Error: {e.stderr}")
        raise Exception("Balanced dataset creation failed")
    except Exception as e:
        print(f"   Error: {str(e)}")
        raise

def train_models(combined_csv, user_dir):
    """
    Train models using train_motion_svm_all_models.py
    """
    print(f"Step 3: Training models...")
    
    # Run training script with combined dataset
    cmd = [
        sys.executable, "train_user_models.py",
        "--dataset", str(combined_csv)
    ]
    
    try:
        print(f"   Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=".", timeout=300)
        print("   Model training completed successfully!")
        
        # Move generated models to user directory
        models_src = Path("models")
        results_src = Path("training_results")
        
        if models_src.exists():
            models_dst = user_dir / "models"
            if models_dst.exists():
                shutil.rmtree(models_dst)
            shutil.copytree(models_src, models_dst)
            print(f"   Models copied to: {models_dst}")
            
            # List model files
            model_files = list(models_dst.glob("*.pkl"))
            if model_files:
                print("   Generated model files:")
                for model_file in model_files:
                    print(f"     {model_file.name}")
        
        if results_src.exists():
            results_dst = user_dir / "training_results"
            if results_dst.exists():
                shutil.rmtree(results_dst)
            shutil.copytree(results_src, results_dst)
            print(f"   Training results copied to: {results_dst}")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("   Model training timed out (300s limit exceeded)")
        return False
    except subprocess.CalledProcessError as e:
        print("   Model training failed!")
        print(f"   Return code: {e.returncode}")
        if e.stderr:
            print(f"   Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"   Training error: {str(e)}")
        return False

def create_compact_dataset(balanced_df, user_dir):
    """
    Create compact dataset with one representative sample per gesture
    Distinguishes between static and dynamic gestures for optimal selection
    """
    print(f"Step 4: Creating compact dataset...")
    
    # Define gesture types - you can expand this list
    static_gestures = {'home', 'end'}  # Add more static gestures here
    dynamic_gestures = {
        'next_slide', 'previous_slide', 'zoom_in', 'zoom_out',
        'rotate_left', 'rotate_right', 'rotate_up', 'rotate_down'
    }
    
    # Motion thresholds for classification
    STATIC_THRESHOLD = 0.05      # < 0.05 = static
    DYNAMIC_THRESHOLD = 0.1      # > 0.1 = dynamic  
    MIN_MOTION = 0.001           # Minimum motion to avoid 0.000
    
    compact_samples = []
    
    # Get one representative sample per gesture
    for gesture in balanced_df['pose_label'].unique():
        gesture_df = balanced_df[balanced_df['pose_label'] == gesture]
        
        # Calculate motion magnitudes
        magnitudes = np.sqrt(gesture_df['delta_x']**2 + gesture_df['delta_y']**2)
        
        # Classify gesture as static or dynamic based on median motion
        median_motion = magnitudes.median()
        is_static = gesture in static_gestures or median_motion < STATIC_THRESHOLD
        
        print(f"   Processing {gesture:15}: {'STATIC' if is_static else 'DYNAMIC'} (median motion: {median_motion:.3f})")
        
        # Find the most common finger pattern first
        right_cols = [f'right_finger_state_{i}' for i in range(5)]
        finger_patterns = []
        for idx, row in gesture_df.iterrows():
            pattern = tuple(int(row[col]) for col in right_cols)
            finger_patterns.append(pattern)
        
        from collections import Counter
        pattern_counts = Counter(finger_patterns)
        most_common_pattern = pattern_counts.most_common(1)[0][0]
        
        # Filter to samples with most common pattern
        pattern_mask = []
        for idx, row in gesture_df.iterrows():
            current_pattern = tuple(int(row[col]) for col in right_cols)
            pattern_mask.append(current_pattern == most_common_pattern)
        
        pattern_samples = gesture_df[pattern_mask].copy()
        pattern_magnitudes = np.sqrt(pattern_samples['delta_x']**2 + pattern_samples['delta_y']**2)
        
        if is_static:
            # For STATIC gestures: Select sample with motion in range [MIN_MOTION, STATIC_THRESHOLD)
            # Priority: closest to center of static range (around 0.02-0.03)
            target_static_motion = 0.025  # Sweet spot for static gestures
            
            # Filter samples in static range
            static_mask = (pattern_magnitudes >= MIN_MOTION) & (pattern_magnitudes < STATIC_THRESHOLD)
            
            if static_mask.sum() > 0:
                static_samples = pattern_samples[static_mask]
                static_magnitudes = pattern_magnitudes[static_mask]
                
                # Select sample closest to target static motion
                distances = abs(static_magnitudes - target_static_motion)
                best_idx = distances.idxmin()
                representative = static_samples.loc[best_idx].copy()
                print(f"     Selected STATIC sample with motion: {static_magnitudes.loc[best_idx]:.3f} (target: {target_static_motion:.3f})")
            else:
                # Fallback: select minimum non-zero motion
                non_zero_mask = pattern_magnitudes > MIN_MOTION
                if non_zero_mask.sum() > 0:
                    min_idx = pattern_magnitudes[non_zero_mask].idxmin()
                    representative = pattern_samples.loc[min_idx].copy()
                    print(f"     Selected STATIC fallback with motion: {pattern_magnitudes.loc[min_idx]:.3f}")
                else:
                    # Last resort: take any sample and adjust motion manually
                    representative = pattern_samples.iloc[0].copy()
                    representative['delta_x'] = target_static_motion if representative['delta_x'] >= 0 else -target_static_motion
                    representative['delta_y'] = 0.0
                    print(f"     Created STATIC sample with adjusted motion: {target_static_motion:.3f}")
        else:
            # For DYNAMIC gestures: Select sample with motion > DYNAMIC_THRESHOLD closest to median
            dynamic_mask = pattern_magnitudes >= DYNAMIC_THRESHOLD
            
            if dynamic_mask.sum() > 0:
                dynamic_samples = pattern_samples[dynamic_mask]
                dynamic_magnitudes = pattern_magnitudes[dynamic_mask]
                
                # Select sample closest to median of dynamic samples
                dynamic_median = dynamic_magnitudes.median()
                distances = abs(dynamic_magnitudes - dynamic_median)
                median_idx = distances.idxmin()
                representative = dynamic_samples.loc[median_idx].copy()
                print(f"     Selected DYNAMIC sample with motion: {dynamic_magnitudes.loc[median_idx]:.3f} (dynamic median: {dynamic_median:.3f})")
            else:
                # Fallback: select sample with highest motion
                max_idx = pattern_magnitudes.idxmax()
                representative = pattern_samples.loc[max_idx].copy()
                
                # Scale up motion if it's too small for dynamic gesture
                current_motion = pattern_magnitudes.loc[max_idx]
                if current_motion < DYNAMIC_THRESHOLD:
                    scale_factor = DYNAMIC_THRESHOLD * 1.2 / current_motion  # Make it 20% above threshold
                    representative['delta_x'] *= scale_factor
                    representative['delta_y'] *= scale_factor
                    new_motion = np.sqrt(representative['delta_x']**2 + representative['delta_y']**2)
                    print(f"     Scaled DYNAMIC motion from {current_motion:.3f} to {new_motion:.3f}")
                else:
                    print(f"     Selected DYNAMIC fallback with motion: {current_motion:.3f}")
        
        # Update instance_id for compact format
        representative['instance_id'] = len(compact_samples) + 1
        compact_samples.append(representative)
    
    # Create compact DataFrame
    compact_df = pd.DataFrame(compact_samples)
    
    # Save compact dataset
    compact_file = user_dir / "gesture_data_compact.csv"
    compact_df.to_csv(compact_file, index=False)
    
    print(f"   Compact dataset created: {compact_file}")
    print(f"   Contains {len(compact_df)} representative samples")
    
    # Show compact summary
    print("   Compact dataset contents:")
    for idx, row in compact_df.iterrows():
        gesture = row['pose_label']
        right_fingers = [f"{int(row[f'right_finger_state_{i}'])}" for i in range(5)]
        finger_pattern = "".join(right_fingers)
        delta_x = row['delta_x']
        delta_y = row['delta_y']
        print(f"     {gesture:15} | Fingers: {finger_pattern} | Motion: ({delta_x:6.3f}, {delta_y:6.3f})")
    
    return compact_df

def create_user_summary(user_dir, user_name, gesture_name, combined_df):
    """Create summary report for user"""
    
    summary_file = user_dir / "update_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write(f"=== GESTURE UPDATE SUMMARY ===\n")
        f.write(f"User: {user_name}\n")
        f.write(f"Updated Gesture: {gesture_name}\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n\n")
        
        f.write(f"Dataset Information:\n")
        f.write(f"- Total samples: {len(combined_df)}\n")
        f.write(f"- Unique gestures: {combined_df['pose_label'].nunique()}\n\n")
        
        f.write(f"Gesture Distribution:\n")
        gesture_counts = combined_df['pose_label'].value_counts()
        for gesture, count in gesture_counts.items():
            f.write(f"- {gesture}: {count} samples\n")
        
        f.write(f"Generated Files:\n")
        f.write(f"- models/: Trained .pkl files\n")
        f.write(f"- training_results/: Training metrics and reports\n")
        f.write(f"- balanced_dataset.csv: Final balanced training dataset\n")
        f.write(f"- gesture_data_compact.csv: Compact reference dataset (1 sample per gesture)\n")
    
    print(f"   Summary saved: {summary_file}")

def auto_detect_csv_files():
    """Auto-detect CSV files in current directory and user folders"""
    csv_files = []
    
    # Check current directory
    for file in Path('.').glob('*.csv'):
        if file.name.startswith('gesture_data_custom_'):
            csv_files.append(file)
    
    # Check user folders for CSV files
    for user_folder in Path('.').glob('user_*'):
        if user_folder.is_dir():
            for csv_file in user_folder.glob('gesture_data_custom_*.csv'):
                csv_files.append(csv_file)
    
    return csv_files

def create_user_folder_and_collect(user_name, gesture_name):
    """
    Create user folder structure for data collection
    Returns the path where CSV should be saved
    """
    user_dir = Path(f"user_{user_name}")
    user_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (user_dir / "raw_data").mkdir(exist_ok=True)
    (user_dir / "models").mkdir(exist_ok=True, parents=True)
    (user_dir / "training_results").mkdir(exist_ok=True, parents=True)
    
    # Return path for CSV file
    csv_filename = f"gesture_data_custom_{user_name}_{gesture_name}.csv"
    csv_path = user_dir / "raw_data" / csv_filename
    
    print(f"=== USER FOLDER SETUP ===")
    print(f"Created: {user_dir}")
    print(f"CSV will be saved to: {csv_path}")
    print(f"Structure:")
    print(f"  {user_dir}/")
    print(f"  |-- raw_data/           -- CSV files from data collection")
    print(f"  |-- models/             -- Trained .pkl files") 
    print(f"  |-- training_results/   -- Training metrics")
    print(f"  |-- balanced_dataset.csv -- Final training dataset")
    print(f"  \\-- gesture_data_compact.csv -- Reference dataset")
    
    return csv_path

def extract_user_info(csv_path):
    """Extract user name and gesture from CSV file"""
    csv_path = Path(csv_path)
    
    # Extract user from filename: gesture_data_custom_USERNAME_GESTURE_TIMESTAMP.csv
    filename_parts = csv_path.stem.split('_')
    if len(filename_parts) >= 4 and csv_path.name.startswith('gesture_data_custom_'):
        user_name = filename_parts[3]  # Nam from gesture_data_custom_Nam_rotate_right_20251109_014715
        gesture_name_from_filename = filename_parts[4] if len(filename_parts) > 4 else "unknown"
    else:
        user_name = csv_path.stem
        gesture_name_from_filename = "unknown"
    
    # Read CSV to detect format and extract info
    try:
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            return user_name, "unknown", {}
        
        # Check for new format (from collect_data_update.py)
        if 'gesture' in df.columns and 'user' in df.columns:
            # New format: user,gesture,finger_thumb,finger_index,finger_middle,finger_ring,finger_pinky,motion_x,motion_y,timestamp
            gesture_name = df['gesture'].iloc[0]
            gesture_counts = df['gesture'].value_counts().to_dict()
            user_name = df['user'].iloc[0]  # Get user from CSV data
            
        # Check for old format (legacy)
        elif 'pose_label' in df.columns:
            # Old format: pose_label,right_finger_state_0,right_finger_state_1,...
            gesture_name = df['pose_label'].iloc[0]
            gesture_counts = df['pose_label'].value_counts().to_dict()
            
        else:
            # Unknown format
            gesture_name = gesture_name_from_filename
            gesture_counts = {gesture_name: len(df)} if gesture_name != "unknown" else {}
            
    except Exception as e:
        print(f"   Warning: Error reading CSV: {e}")
        gesture_name = gesture_name_from_filename
        gesture_counts = {}
    
    return user_name, gesture_name, gesture_counts

def check_existing_user_gestures(user_name):
    """Check if user already has gestures and what they are"""
    user_dir = Path(f"user_{user_name}")
    existing_gestures = []
    
    if user_dir.exists():
        # Check raw_data folder for CSV files
        raw_data_dir = user_dir / "raw_data"
        if raw_data_dir.exists():
            for csv_file in raw_data_dir.glob('gesture_data_custom_*.csv'):
                try:
                    df = pd.read_csv(csv_file)
                    if 'pose_label' in df.columns and len(df) > 0:
                        gesture = df['pose_label'].iloc[0]
                        if gesture not in existing_gestures:
                            existing_gestures.append(gesture)
                except:
                    pass
        
        # Also check for existing balanced dataset
        balanced_file = user_dir / "balanced_dataset.csv"
        if balanced_file.exists():
            try:
                df = pd.read_csv(balanced_file)
                if 'pose_label' in df.columns:
                    # Filter out standard gestures, only show user's custom ones
                    all_gestures = df['pose_label'].unique()
                    standard_gestures = {
                        'home', 'end', 'next_slide', 'previous_slide',
                        'zoom_in', 'zoom_out', 'rotate_left', 'rotate_right',
                        'rotate_up', 'rotate_down'
                    }
                    for gesture in all_gestures:
                        if gesture not in standard_gestures and gesture not in existing_gestures:
                            existing_gestures.append(gesture)
            except:
                pass
    
    return existing_gestures

def process_user_folder(user_name):
    """
    Process all CSV files in a user's folder
    Auto-detect and process each gesture
    """
    user_dir = Path(f"user_{user_name}")
    
    if not user_dir.exists():
        print(f"ERROR: User folder not found: {user_dir}")
        return False
    
    raw_data_dir = user_dir / "raw_data"
    if not raw_data_dir.exists():
        print(f"ERROR: Raw data folder not found: {raw_data_dir}")
        return False
    
    # Find all CSV files
    csv_files = list(raw_data_dir.glob('gesture_data_custom_*.csv'))
    
    if not csv_files:
        print(f"ERROR: No CSV files found in {raw_data_dir}")
        return False
    
    print(f"=== PROCESSING USER FOLDER: {user_name} ===")
    print(f"Found {len(csv_files)} CSV file(s)")
    
    success_count = 0
    
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")
        
        # Extract user name and gesture info
        user_name_from_file, gesture_name, gesture_counts = extract_user_info(csv_file)
        
        print(f"   User: {user_name_from_file}")
        print(f"   Gesture: {gesture_name}")
        print(f"   Samples: {sum(gesture_counts.values()) if gesture_counts else 0}")
        
        # Process this CSV
        success = process_user_gesture_update(csv_file, gesture_name)
        if success:
            success_count += 1
            print(f"   SUCCESS!")
        else:
            print(f"   FAILED!")
    
    print(f"\n=== FOLDER PROCESSING SUMMARY ===")
    print(f"Processed: {success_count}/{len(csv_files)} files")
    print(f"User folder: {user_dir.absolute()}")
    
    return success_count == len(csv_files)

def process_user_gesture_update(user_csv_path, gesture_name=None):
    """
    Complete pipeline: Record 5 -> Augment to 100 -> Combine -> Train -> User folder
    Supports multiple gestures per user (accumulative)
    """
    try:
        user_csv_path = Path(user_csv_path)
        
        if not user_csv_path.exists():
            raise FileNotFoundError(f"User CSV not found: {user_csv_path}")
        
        # Auto-extract user name and ALL gesture info
        user_name, detected_gesture, gesture_counts = extract_user_info(user_csv_path)
        
        # Get ALL unique gestures in the file
        df = pd.read_csv(user_csv_path)
        if 'pose_label' in df.columns:
            all_gestures = df['pose_label'].unique().tolist()
        else:
            all_gestures = [detected_gesture]
        
        # Check existing gestures for this user
        existing_gestures = check_existing_user_gestures(user_name)
        
        print(f"=== GESTURE UPDATE PIPELINE ===")
        print(f"User: {user_name}")
        print(f"Detected gestures: {all_gestures}")
        print(f"Input: {user_csv_path}")
        if len(gesture_counts) > 0:
            print(f"Sample distribution: {dict(gesture_counts)}")
        
        if existing_gestures:
            print(f"Existing custom gestures: {existing_gestures}")
        
        # Show warnings for EACH gesture that will be updated
        for gesture in all_gestures:
            if gesture in existing_gestures:
                print(f"WARNING: '{gesture}' will be updated (replaced)")
            else:
                print(f"INFO: Adding new gesture '{gesture}' to user's collection")
        
        # Use first gesture as main for compatibility
        main_gesture = all_gestures[0] if all_gestures else detected_gesture
        print()
        
        # Create user directory
        user_dir = Path(f"user_{user_name}")
        user_dir.mkdir(exist_ok=True)
        
        # Step 1: Create balanced dataset - REPLACE all user gestures completely
        combined_csv = user_dir / "balanced_dataset.csv"
        combined_df = create_balanced_dataset_with_user_data(user_csv_path, all_gestures, combined_csv)
        
        # Step 3: Train models
        success = train_models(combined_csv, user_dir)
        
        if success:
            # Step 4: Create compact dataset for reference
            compact_df = create_compact_dataset(combined_df, user_dir)
            
            # Create summary (list all gestures processed)
            summary_gestures = ", ".join(all_gestures)
            create_user_summary(user_dir, user_name, summary_gestures, combined_df)
            
            print(f"\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
            print(f"User folder: {user_dir.absolute()}")
            print(f"- models/: Contains .pkl files")
            print(f"- training_results/: Contains training metrics")
            print(f"- balanced_dataset.csv: Final balanced training dataset")
            print(f"- gesture_data_compact.csv: Compact reference dataset")
            print(f"- update_summary.txt: Process summary")
            
            return True
        else:
            print(f"\n=== PIPELINE FAILED ===")
            return False
            
    except Exception as e:
        print(f"\n=== PIPELINE ERROR ===")
        print(f"Error: {str(e)}")
        return False

def interactive_mode():
    """Interactive mode to select CSV file and options"""
    print("=== INTERACTIVE GESTURE UPDATE MODE ===")
    
    # Auto-detect CSV files
    csv_files = auto_detect_csv_files()
    
    if not csv_files:
        print("ERROR: No user CSV files found!")
        print("Expected format: gesture_data_custom_USERNAME.csv")
        return False
    
    print(f"[FILES] Found {len(csv_files)} user CSV file(s):")
    
    # Show available files with details
    for i, csv_file in enumerate(csv_files, 1):
        user_name, gesture_name, gesture_counts = extract_user_info(csv_file)
        samples = sum(gesture_counts.values()) if gesture_counts else 0
        print(f"  {i}. {csv_file.name}")
        print(f"     User: {user_name}")
        print(f"     Gesture: {gesture_name}")
        print(f"     Samples: {samples}")
        print()
    
    # Let user choose
    try:
        if len(csv_files) == 1:
            choice = 1
            print(f"[AUTO] Auto-selecting: {csv_files[0].name}")
        else:
            choice = int(input(f"Choose file (1-{len(csv_files)}): "))
            if choice < 1 or choice > len(csv_files):
                print("[ERROR] Invalid choice!")
                return False
        
        selected_file = csv_files[choice - 1]
        
        # Process the selected file
        print(f"Processing {selected_file.name}...")
        return process_user_gesture_update(selected_file)
        
    except (ValueError, KeyboardInterrupt):
        print("Operation cancelled!")
        return False

def list_mode():
    """List all available CSV files"""
    print("=== AVAILABLE USER CSV FILES ===")
    
    csv_files = auto_detect_csv_files()
    
    if not csv_files:
        print("ERROR: No user CSV files found!")
        print("Expected format: gesture_data_custom_USERNAME.csv")
        return
    
    for csv_file in csv_files:
        user_name, gesture_name, gesture_counts = extract_user_info(csv_file)
        samples = sum(gesture_counts.values()) if gesture_counts else 0
        
        print(f"\n[FILE] File: {csv_file.name}")
        print(f"   User: {user_name}")
        print(f"   Gesture: {gesture_name}")
        print(f"   Samples: {samples}")
        print(f"   Command: python user_gesture_pipeline.py --user-csv {csv_file.name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="User Gesture Update Pipeline - Flexible CSV Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (auto-detect and choose)
  python user_gesture_pipeline.py
  
  # Create user folder structure for data collection
  python user_gesture_pipeline.py --create-folder Nam zoom_in
  
  # Process single CSV file (auto-detect gesture)  
  python user_gesture_pipeline.py --user-csv gesture_data_custom_Khang.csv
  
  # Process all CSVs in user folder
  python user_gesture_pipeline.py --user-folder Nam
  
  # List all available files
  python user_gesture_pipeline.py --list
        """
    )
    
    parser.add_argument("--user-csv", help="Path to user's custom CSV file (optional)")
    parser.add_argument("--gesture", help="Name of updated gesture (optional, auto-detected)")
    parser.add_argument("--user-folder", help="Process all CSV files in user's folder")
    parser.add_argument("--create-folder", nargs=2, metavar=('USER', 'GESTURE'), help="Create user folder structure for data collection")
    parser.add_argument("--list", action="store_true", help="List all available CSV files")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode (default if no args)")
    
    args = parser.parse_args()
    
    # Determine mode
    if args.list:
        list_mode()
        success = True
    elif args.create_folder:
        # Create user folder mode
        user_name, gesture_name = args.create_folder
        csv_path = create_user_folder_and_collect(user_name, gesture_name)
        print(f"SUCCESS: Folder created successfully!")
        print(f"Next steps:")
        print(f"1. Collect data and save to: {csv_path}")
        print(f"2. Run: python user_gesture_pipeline.py --user-folder {user_name}")
        success = True
    elif args.user_folder:
        # Process user folder mode
        success = process_user_folder(args.user_folder)
    elif args.user_csv:
        # Direct processing mode
        success = process_user_gesture_update(args.user_csv, args.gesture)
    else:
        # Interactive mode (default)
        success = interactive_mode()
    
    exit(0 if success else 1)