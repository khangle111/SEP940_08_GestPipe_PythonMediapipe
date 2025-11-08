#!/usr/bin/env python3
"""
Create compact dataset from user's gesture data
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def create_compact_with_user_gestures(user_data_file, output_compact_file, base_compact_file="general_training_results/gesture_data_compact.csv"):
    """
    Create compact dataset by replacing user's custom gestures in base compact dataset
    """
    print(f"Loading base compact data from: {base_compact_file}")
    
    # Load base compact dataset (default 10 gestures)
    base_df = pd.read_csv(base_compact_file)
    print(f"Base compact gestures: {base_df['pose_label'].unique().tolist()}")
    
    print(f"\nLoading user data from: {user_data_file}")
    
    # Load user data
    user_df = pd.read_csv(user_data_file)
    user_gestures = user_df['pose_label'].unique()
    
    print(f"User custom gestures: {user_gestures.tolist()}")
    
    # Show distribution
    gesture_counts = user_df['pose_label'].value_counts()
    print("\nUser gesture distribution:")
    for gesture, count in gesture_counts.items():
        print(f"  {gesture}: {count} samples")
    
    # Start with base compact data
    final_data = []
    instance_id = 1
    
    # Process each gesture from base compact
    for _, base_row in base_df.iterrows():
        gesture = base_row['pose_label']
        
        if gesture in user_gestures:
            # Replace with user's custom data
            print(f"\nReplacing {gesture} with user data...")
            
            # Get user samples for this gesture
            user_samples = user_df[user_df['pose_label'] == gesture]
            
            # Calculate delta magnitude for each user sample
            deltas = []
            for _, row in user_samples.iterrows():
                delta_mag = np.sqrt(row['delta_x']**2 + row['delta_y']**2)
                deltas.append(delta_mag)
            
            # Find median delta magnitude
            median_delta = np.median(deltas)
            print(f"  User median delta magnitude: {median_delta:.4f}")
            
            # Select sample closest to median
            best_idx = np.argmin([abs(d - median_delta) for d in deltas])
            selected_sample = user_samples.iloc[best_idx].copy()
            
            # Update instance_id
            selected_sample['instance_id'] = instance_id
            
            # Show finger states
            left_fingers = [int(selected_sample[f'left_finger_state_{i}']) for i in range(5)]
            right_fingers = [int(selected_sample[f'right_finger_state_{i}']) for i in range(5)]
            print(f"  USER CUSTOM - L:{left_fingers} R:{right_fingers} delta_mag:{deltas[best_idx]:.4f}")
            
            final_data.append(selected_sample)
        else:
            # Keep base data
            print(f"\nKeeping base {gesture}...")
            base_sample = base_row.copy()
            base_sample['instance_id'] = instance_id
            
            # Show finger states
            left_fingers = [int(base_sample[f'left_finger_state_{i}']) for i in range(5)]
            right_fingers = [int(base_sample[f'right_finger_state_{i}']) for i in range(5)]
            print(f"  BASE DEFAULT - L:{left_fingers} R:{right_fingers}")
            
            final_data.append(base_sample)
        
        instance_id += 1
    
    # Convert to DataFrame
    compact_df = pd.DataFrame(final_data)
    
    print(f"\nFinal compact dataset: {len(compact_df)} samples")
    
    # Save compact file
    compact_df.to_csv(output_compact_file, index=False)
    print(f"\nCompact dataset saved to: {output_compact_file}")
    
    # Show summary
    print("\nFinal compact dataset contents:")
    for _, row in compact_df.iterrows():
        left_fingers = ''.join([str(int(row[f'left_finger_state_{i}'])) for i in range(5)])
        right_fingers = ''.join([str(int(row[f'right_finger_state_{i}'])) for i in range(5)])
        is_custom = row['pose_label'] in user_gestures
        marker = "CUSTOM" if is_custom else "DEFAULT"
        print(f"  {row['pose_label']:15} | L:{left_fingers} R:{right_fingers} | Motion: ({row['delta_x']:6.3f}, {row['delta_y']:6.3f}) [{marker}]")
    
    return compact_df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create compact dataset with user custom gestures')
    parser.add_argument('--user-data', required=True, help='Path to user combined data file (e.g., gesture_data_custom_A.csv)')
    parser.add_argument('--output', help='Output compact file path (optional)')
    parser.add_argument('--base-compact', default='general_training_results/gesture_data_compact.csv', 
                       help='Base compact dataset with all default gestures')
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Auto-generate output path in user's training_results directory
        input_path = Path(args.user_data)
        user_folder = input_path.parent
        training_results_dir = user_folder / 'training_results'
        training_results_dir.mkdir(exist_ok=True)
        output_path = training_results_dir / 'gesture_data_compact.csv'
    
    # Create compact dataset with user customizations
    create_compact_with_user_gestures(args.user_data, output_path, args.base_compact)

if __name__ == "__main__":
    main()