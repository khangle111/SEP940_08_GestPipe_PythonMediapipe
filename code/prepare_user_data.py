#!/usr/bin/env python3
"""
Simple script to prepare user training data
Usage: python prepare_user_data.py user_folder
Example: python prepare_user_data.py user_Bi
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def prepare_user_training_data(user_folder):
    """Prepare complete training data for user"""
    
    user_path = Path(user_folder)
    if not user_path.exists():
        print(f"❌ User folder not found: {user_folder}")
        return False
    
    # 1. Find user's custom data file
    custom_files = list(user_path.glob("gesture_data_custom_*.csv"))
    if not custom_files:
        print(f"❌ No custom data file found in {user_folder}")
        print("   Expected: gesture_data_custom_*.csv")
        return False
    
    user_data_file = custom_files[0]
    print(f"📂 Found user data: {user_data_file}")
    
    # 2. Load base reference data
    base_file = Path("training_results/gesture_data_compact.csv")
    if not base_file.exists():
        print(f"❌ Base reference file not found: {base_file}")
        return False
    
    print(f"📂 Loading base data: {base_file}")
    base_df = pd.read_csv(base_file)
    user_df = pd.read_csv(user_data_file)
    
    user_gestures = user_df['pose_label'].unique()
    print(f"✅ User has {len(user_gestures)} custom gestures: {list(user_gestures)}")
    
    # 3. Create compact dataset (10 samples)
    print(f"\n🔄 Creating compact dataset...")
    compact_samples = []
    
    for _, base_row in base_df.iterrows():
        gesture = base_row['pose_label']
        
        if gesture in user_gestures:
            # Use user's custom data
            user_samples = user_df[user_df['pose_label'] == gesture]
            selected = user_samples.iloc[0].copy()  # Use first sample
            selected['instance_id'] = len(compact_samples) + 1
            print(f"   ✅ {gesture}: Using CUSTOM data")
        else:
            # Use base data
            selected = base_row.copy()
            selected['instance_id'] = len(compact_samples) + 1
            print(f"   📋 {gesture}: Using BASE data")
        
        compact_samples.append(selected)
    
    compact_df = pd.DataFrame(compact_samples)
    
    # 4. Create training_results directory
    training_dir = user_path / "training_results"
    training_dir.mkdir(exist_ok=True)
    
    # Save compact dataset
    compact_file = training_dir / "gesture_data_compact.csv"
    compact_df.to_csv(compact_file, index=False)
    print(f"✅ Compact dataset saved: {compact_file}")
    
    # 5. Generate balanced dataset (1000 samples)
    print(f"\n🔄 Generating balanced dataset (1000 samples)...")
    balanced_samples = []
    
    np.random.seed(42)  # For reproducibility
    
    for gesture in compact_df['pose_label'].unique():
        base_row = compact_df[compact_df['pose_label'] == gesture].iloc[0]
        
        for i in range(100):  # 100 samples per gesture
            new_row = base_row.copy()
            new_row['instance_id'] = len(balanced_samples) + 1
            
            # Add small noise to motion features
            motion_cols = ['motion_x_start', 'motion_y_start', 'motion_x_mid', 
                          'motion_y_mid', 'motion_x_end', 'motion_y_end', 
                          'main_axis_x', 'main_axis_y', 'delta_x', 'delta_y']
            
            for col in motion_cols:
                if col in new_row:
                    new_row[col] += np.random.normal(0, 0.01)
            
            balanced_samples.append(new_row)
    
    balanced_df = pd.DataFrame(balanced_samples)
    
    # Save balanced dataset
    balanced_file = user_path / "gesture_data_1000_balanced.csv"
    balanced_df.to_csv(balanced_file, index=False)
    print(f"✅ Balanced dataset saved: {balanced_file}")
    
    # 6. Auto-train user models
    print(f"\n🚀 Auto-training user models...")
    
    # Import training functions
    import subprocess
    import os
    
    try:
        # Run training command
        cmd = f"python train_user_models.py --dataset {balanced_file}"
        print(f"   Running: {cmd}")
        
        result = subprocess.run(cmd.split(), 
                              capture_output=True, 
                              text=True, 
                              cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"✅ Training completed successfully!")
            print(f"📊 Model saved to: {user_path}/models/")
        else:
            print(f"❌ Training failed:")
            print(result.stderr)
    
    except Exception as e:
        print(f"❌ Training error: {e}")
        print(f"🔧 Manual train: python train_user_models.py --dataset {balanced_file}")
    
    # Summary
    print(f"\n🎉 User setup completed!")
    print(f"   📊 Templates: {len(compact_df)} gestures → {compact_file}")
    print(f"   📊 Training: {len(balanced_df)} samples → {balanced_file}")
    print(f"   🤖 Models: {user_path}/models/")
    
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python prepare_user_data.py <user_folder>")
        print("Example: python prepare_user_data.py user_Bi")
        sys.exit(1)
    
    user_folder = sys.argv[1]
    success = prepare_user_training_data(user_folder)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()