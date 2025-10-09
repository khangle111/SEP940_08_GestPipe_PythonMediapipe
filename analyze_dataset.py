import pandas as pd
import numpy as np
from collections import Counter

# Load training dataset
df = pd.read_csv('gesture_motion_dataset_realistic.csv')

print("=== GESTURE DISTRIBUTION ANALYSIS ===")
print(f"Total samples: {len(df)}")
print(f"Gestures: {df['pose_label'].value_counts()}")

print("\n=== FINGER STATE PATTERNS BY GESTURE ===")
finger_cols = [f'left_finger_state_{i}' for i in range(5)] + [f'right_finger_state_{i}' for i in range(5)]

for gesture in df['pose_label'].unique():
    gesture_df = df[df['pose_label'] == gesture]
    print(f"\n--- {gesture.upper()} ---")
    
    # Get finger patterns
    patterns = []
    for _, row in gesture_df.iterrows():
        left_pattern = tuple(int(row[f'left_finger_state_{i}']) for i in range(5))
        right_pattern = tuple(int(row[f'right_finger_state_{i}']) for i in range(5))
        patterns.append((left_pattern, right_pattern))
    
    # Count most common patterns
    counter = Counter(patterns)
    print(f"Total samples: {len(gesture_df)}")
    print("Top 3 finger patterns:")
    for i, (pattern, count) in enumerate(counter.most_common(3)):
        left, right = pattern
        print(f"  {i+1}. Left: {left}, Right: {right} - {count} samples ({count/len(gesture_df)*100:.1f}%)")
    
    # Motion analysis
    motion_left = len(gesture_df[gesture_df['delta_x'] < 0])
    motion_right = len(gesture_df[gesture_df['delta_x'] > 0])
    motion_up = len(gesture_df[gesture_df['delta_y'] < 0])
    motion_down = len(gesture_df[gesture_df['delta_y'] > 0])
    
    print(f"Motion distribution:")
    print(f"  Left: {motion_left} ({motion_left/len(gesture_df)*100:.1f}%)")
    print(f"  Right: {motion_right} ({motion_right/len(gesture_df)*100:.1f}%)")
    print(f"  Up: {motion_up} ({motion_up/len(gesture_df)*100:.1f}%)")
    print(f"  Down: {motion_down} ({motion_down/len(gesture_df)*100:.1f}%)")

print("\n=== GESTURE SIMILARITY ANALYSIS ===")
# Check overlap between gestures
gestures = df['pose_label'].unique()
for i, g1 in enumerate(gestures):
    for g2 in gestures[i+1:]:
        g1_df = df[df['pose_label'] == g1]
        g2_df = df[df['pose_label'] == g2]
        
        # Check finger pattern overlap
        g1_patterns = set()
        g2_patterns = set()
        
        for _, row in g1_df.iterrows():
            pattern = tuple(int(row[col]) for col in finger_cols)
            g1_patterns.add(pattern)
            
        for _, row in g2_df.iterrows():
            pattern = tuple(int(row[col]) for col in finger_cols)
            g2_patterns.add(pattern)
        
        overlap = g1_patterns.intersection(g2_patterns)
        if overlap:
            print(f"{g1} vs {g2}: {len(overlap)} overlapping finger patterns")
            for pattern in list(overlap)[:3]:  # Show first 3
                left = pattern[:5]
                right = pattern[5:]
                print(f"  Overlap: Left={left}, Right={right}")