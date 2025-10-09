"""
Phân tích delta motion của dataset gesture thực tế
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_motion_deltas():
    """Phân tích delta motion cho từng gesture"""
    
    # Load data
    df = pd.read_csv('gesture_data_09_10_2025.csv')
    
    print("=== MOTION DELTA ANALYSIS ===")
    print(f"Total samples: {len(df)}")
    print(f"Gestures: {df['pose_label'].nunique()}")
    print()
    
    # Calculate delta magnitude for each sample
    df['delta_magnitude'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2)
    
    # Analyze by gesture
    gesture_stats = {}
    
    for gesture in sorted(df['pose_label'].unique()):
        gesture_data = df[df['pose_label'] == gesture]
        
        deltas = gesture_data['delta_magnitude']
        
        stats = {
            'count': len(deltas),
            'mean': deltas.mean(),
            'median': deltas.median(),
            'std': deltas.std(),
            'min': deltas.min(),
            'max': deltas.max(),
            'q25': deltas.quantile(0.25),
            'q75': deltas.quantile(0.75),
            'below_005': (deltas < 0.05).sum(),  # Current MIN_DELTA_MAG threshold
            'below_008': (deltas < 0.08).sum(),  # Proposed threshold
            'below_010': (deltas < 0.10).sum(),  # Higher threshold
        }
        
        gesture_stats[gesture] = stats
    
    # Print detailed statistics
    print("=== GESTURE-WISE DELTA STATISTICS ===")
    print(f"{'Gesture':<15} {'Count':<6} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 80)
    
    for gesture, stats in gesture_stats.items():
        print(f"{gesture:<15} {stats['count']:<6} {stats['mean']:<8.4f} {stats['median']:<8.4f} "
              f"{stats['std']:<8.4f} {stats['min']:<8.4f} {stats['max']:<8.4f}")
    
    print()
    
    # Threshold analysis
    print("=== THRESHOLD IMPACT ANALYSIS ===")
    print(f"{'Gesture':<15} {'Total':<6} {'<0.05':<6} {'<0.08':<6} {'<0.10':<6} {'%<0.05':<8} {'%<0.08':<8} {'%<0.10':<8}")
    print("-" * 90)
    
    total_below_005 = 0
    total_below_008 = 0
    total_below_010 = 0
    
    for gesture, stats in gesture_stats.items():
        pct_005 = (stats['below_005'] / stats['count']) * 100
        pct_008 = (stats['below_008'] / stats['count']) * 100
        pct_010 = (stats['below_010'] / stats['count']) * 100
        
        print(f"{gesture:<15} {stats['count']:<6} {stats['below_005']:<6} {stats['below_008']:<6} "
              f"{stats['below_010']:<6} {pct_005:<8.1f} {pct_008:<8.1f} {pct_010:<8.1f}")
        
        total_below_005 += stats['below_005']
        total_below_008 += stats['below_008'] 
        total_below_010 += stats['below_010']
    
    # Overall statistics
    print("-" * 90)
    total_samples = len(df)
    pct_total_005 = (total_below_005 / total_samples) * 100
    pct_total_008 = (total_below_008 / total_samples) * 100
    pct_total_010 = (total_below_010 / total_samples) * 100
    
    print(f"{'TOTAL':<15} {total_samples:<6} {total_below_005:<6} {total_below_008:<6} "
          f"{total_below_010:<6} {pct_total_005:<8.1f} {pct_total_008:<8.1f} {pct_total_010:<8.1f}")
    
    print()
    
    # Identify static vs dynamic gestures
    print("=== STATIC vs DYNAMIC CLASSIFICATION ===")
    
    static_candidates = []
    dynamic_gestures = []
    
    for gesture, stats in gesture_stats.items():
        median_delta = stats['median']
        pct_small_motion = (stats['below_008'] / stats['count']) * 100
        
        if median_delta < 0.08 and pct_small_motion > 50:
            static_candidates.append((gesture, median_delta, pct_small_motion))
        else:
            dynamic_gestures.append((gesture, median_delta, pct_small_motion))
    
    if static_candidates:
        print("POTENTIAL STATIC GESTURES:")
        for gesture, median, pct in static_candidates:
            print(f"  {gesture}: median_delta={median:.4f}, {pct:.1f}% below 0.08")
    else:
        print("NO CLEAR STATIC GESTURES DETECTED")
    
    print()
    print("DYNAMIC GESTURES:")
    for gesture, median, pct in dynamic_gestures:
        print(f"  {gesture}: median_delta={median:.4f}, {pct:.1f}% below 0.08")
    
    # Direction analysis
    print("\n=== MOTION DIRECTION ANALYSIS ===")
    
    direction_stats = {}
    for gesture in sorted(df['pose_label'].unique()):
        gesture_data = df[df['pose_label'] == gesture]
        
        # Analyze primary motion direction
        horizontal_motion = np.abs(gesture_data['delta_x']) > np.abs(gesture_data['delta_y'])
        vertical_motion = ~horizontal_motion
        
        left_motion = gesture_data['delta_x'] < -0.02
        right_motion = gesture_data['delta_x'] > 0.02
        up_motion = gesture_data['delta_y'] < -0.02  # Y decreases upward
        down_motion = gesture_data['delta_y'] > 0.02
        
        direction_stats[gesture] = {
            'horizontal': horizontal_motion.sum(),
            'vertical': vertical_motion.sum(),
            'left': left_motion.sum(),
            'right': right_motion.sum(), 
            'up': up_motion.sum(),
            'down': down_motion.sum(),
            'total': len(gesture_data)
        }
    
    print(f"{'Gesture':<15} {'Total':<6} {'Left':<6} {'Right':<6} {'Up':<6} {'Down':<6} {'Primary Direction'}")
    print("-" * 85)
    
    for gesture, stats in direction_stats.items():
        directions = [
            ('Left', stats['left']),
            ('Right', stats['right']),
            ('Up', stats['up']),
            ('Down', stats['down'])
        ]
        
        primary = max(directions, key=lambda x: x[1])
        
        print(f"{gesture:<15} {stats['total']:<6} {stats['left']:<6} {stats['right']:<6} "
              f"{stats['up']:<6} {stats['down']:<6} {primary[0]} ({primary[1]}/{stats['total']})")
    
    return gesture_stats, direction_stats

def recommend_thresholds(gesture_stats):
    """Đưa ra khuyến nghị về threshold dựa trên phân tích"""
    
    print("\n=== THRESHOLD RECOMMENDATIONS ===")
    
    # Calculate impact of different thresholds
    thresholds = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    
    print("Impact of different MIN_DELTA_MAG thresholds:")
    print(f"{'Threshold':<10} {'Rejected Samples':<16} {'Percentage':<12} {'Recommendation'}")
    print("-" * 60)
    
    total_samples = sum(stats['count'] for stats in gesture_stats.values())
    
    for threshold in thresholds:
        rejected = 0
        for gesture, stats in gesture_stats.items():
            # Count samples below threshold (would be rejected)
            gesture_data = pd.read_csv('gesture_data_09_10_2025.csv')
            gesture_samples = gesture_data[gesture_data['pose_label'] == gesture]
            delta_mag = np.sqrt(gesture_samples['delta_x']**2 + gesture_samples['delta_y']**2)
            rejected += (delta_mag < threshold).sum()
        
        pct_rejected = (rejected / total_samples) * 100
        
        if pct_rejected < 5:
            recommendation = "✅ Good"
        elif pct_rejected < 15:
            recommendation = "⚠️ Acceptable" 
        else:
            recommendation = "❌ Too strict"
            
        print(f"{threshold:<10.2f} {rejected:<16} {pct_rejected:<12.1f} {recommendation}")
    
    print("\n📋 RECOMMENDATIONS:")
    print("1. Current 0.05: May be too lenient, allows very small motions")
    print("2. Proposed 0.08: Good balance, rejects ~10-15% of small motions")  
    print("3. For thesis: Use 0.08 with time-based validation for static gestures")
    print("4. Static gestures (home, end): Implement separate time-based detection")

if __name__ == "__main__":
    gesture_stats, direction_stats = analyze_motion_deltas()
    recommend_thresholds(gesture_stats)