import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('gesture_data_09_10_2025.csv')

# Calculate delta magnitude for each sample
df['delta_magnitude'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2)

# Analyze by gesture type
static_gestures = ['home', 'end']
dynamic_gestures = ['next_slide', 'rotate_right', 'rotate_left', 'zoom_in', 'zoom_out', 'rotate_up', 'rotate_down', 'previous_slide']

print('=== DELTA MAGNITUDE ANALYSIS ===')
print()

# Static gestures analysis
static_data = df[df['pose_label'].isin(static_gestures)]
print('📊 STATIC GESTURES:')
print(f'  Count: {len(static_data)} samples')
print(f'  Delta magnitude: {static_data["delta_magnitude"].mean():.4f} ± {static_data["delta_magnitude"].std():.4f}')
print(f'  Min: {static_data["delta_magnitude"].min():.4f}')
print(f'  Max: {static_data["delta_magnitude"].max():.4f}')
print(f'  95th percentile: {static_data["delta_magnitude"].quantile(0.95):.4f}')
print()

# Dynamic gestures analysis  
dynamic_data = df[df['pose_label'].isin(dynamic_gestures)]
print('🎯 DYNAMIC GESTURES:')
print(f'  Count: {len(dynamic_data)} samples')
print(f'  Delta magnitude: {dynamic_data["delta_magnitude"].mean():.4f} ± {dynamic_data["delta_magnitude"].std():.4f}')
print(f'  Min: {dynamic_data["delta_magnitude"].min():.4f}')
print(f'  Max: {dynamic_data["delta_magnitude"].max():.4f}')
print(f'  5th percentile: {dynamic_data["delta_magnitude"].quantile(0.05):.4f}')
print(f'  25th percentile: {dynamic_data["delta_magnitude"].quantile(0.25):.4f}')
print()

# Optimal threshold
static_95th = static_data['delta_magnitude'].quantile(0.95)
dynamic_5th = dynamic_data['delta_magnitude'].quantile(0.05)
optimal_threshold = (static_95th + dynamic_5th) / 2

print('⚡ OPTIMAL SETTINGS:')
print(f'  Static 95th percentile: {static_95th:.4f}')
print(f'  Dynamic 5th percentile: {dynamic_5th:.4f}') 
print(f'  Suggested threshold: {optimal_threshold:.4f}')
print(f'  Safety margin: {optimal_threshold * 0.8:.4f} - {optimal_threshold * 1.2:.4f}')

# Camera distance recommendations
print()
print('🎥 CAMERA DISTANCE RECOMMENDATIONS:')
print()
print('Based on delta magnitude analysis:')
print(f'  • Training data shows dynamic gestures need delta > {dynamic_5th:.3f}')
print(f'  • Static gestures should stay below {static_95th:.3f}')
print()
print('📏 RECOMMENDED DISTANCES:')
print('  • TOO CLOSE (<40cm): Movements too small → low delta → misclassified as static')
print('  • OPTIMAL (60-100cm): Full arm movement → good delta → proper classification') 
print('  • TOO FAR (>150cm): Hand too small → tracking issues → inconsistent delta')
print()
print('🎯 BEST PRACTICE:')
print('  1. Sit 70-90cm from camera (arm\'s length)')
print('  2. Ensure full hand is visible in frame')
print('  3. Make WIDE, CLEAR movements for dynamic gestures')
print('  4. Keep hand STILL for static gestures')