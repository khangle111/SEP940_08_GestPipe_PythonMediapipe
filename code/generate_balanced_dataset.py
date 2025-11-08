#!/usr/bin/env python3
"""
Generate balanced dataset (1000 samples) from compact dataset (10 samples)
Usage: python generate_balanced_dataset.py input_compact.csv output_balanced.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def generate_balanced_dataset(input_file, output_file, samples_per_gesture=100):
    """Generate balanced dataset from compact dataset"""
    
    # Load compact dataset
    print(f"📂 Loading compact dataset: {input_file}")
    compact_df = pd.read_csv(input_file)
    
    gestures = compact_df['pose_label'].unique()
    print(f"✅ Found {len(compact_df)} samples, {len(gestures)} gestures")
    print(f"   Gestures: {list(gestures)}")
    
    # Generate balanced samples
    print(f"🔄 Generating {samples_per_gesture} samples per gesture...")
    balanced_samples = []
    
    for gesture in gestures:
        base_row = compact_df[compact_df['pose_label'] == gesture].iloc[0].copy()
        print(f"   Processing {gesture}...")
        
        for i in range(samples_per_gesture):
            new_row = base_row.copy()
            new_row['instance_id'] = len(balanced_samples) + 1
            
            # Add small random variations to motion features
            noise_scale = 0.01  # Small noise to avoid overfitting
            motion_cols = ['motion_x_start', 'motion_y_start', 'motion_x_mid', 
                          'motion_y_mid', 'motion_x_end', 'motion_y_end', 
                          'main_axis_x', 'main_axis_y', 'delta_x', 'delta_y']
            
            for col in motion_cols:
                if col in new_row:
                    # Add Gaussian noise
                    new_row[col] += np.random.normal(0, noise_scale)
            
            balanced_samples.append(new_row)
    
    # Create balanced dataframe
    balanced_df = pd.DataFrame(balanced_samples)
    
    # Save to file
    print(f"💾 Saving balanced dataset: {output_file}")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n✅ Balanced dataset created successfully!")
    print(f"   📊 Total samples: {len(balanced_df)}")
    print(f"   📊 Samples per gesture: {samples_per_gesture}")
    print(f"   📊 Gestures: {len(gestures)}")
    
    # Verify distribution
    gesture_counts = balanced_df['pose_label'].value_counts()
    print(f"\n📈 Sample distribution:")
    for gesture, count in gesture_counts.items():
        print(f"   {gesture}: {count} samples")
    
    return output_file

def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_balanced_dataset.py <input_compact.csv> <output_balanced.csv>")
        print("Example: python generate_balanced_dataset.py user_Bi/gesture_data_compact_balanced.csv user_Bi/gesture_data_1000_balanced.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate balanced dataset
    generate_balanced_dataset(input_file, output_file)

if __name__ == "__main__":
    main()