import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('gesture_data_09_10_2025.csv')

# Focus on zoom gestures with correct finger pattern
zoom_in_data = df[(df['pose_label'] == 'zoom_in') & 
                  (df['right_finger_state_0'] == 1) &
                  (df['right_finger_state_1'] == 1) &
                  (df['right_finger_state_2'] == 1) &
                  (df['right_finger_state_3'] == 0) &
                  (df['right_finger_state_4'] == 0)]

zoom_out_data = df[(df['pose_label'] == 'zoom_out') & 
                   (df['right_finger_state_0'] == 1) &
                   (df['right_finger_state_1'] == 1) &
                   (df['right_finger_state_2'] == 1) &
                   (df['right_finger_state_3'] == 0) &
                   (df['right_finger_state_4'] == 0)]

print('🎯 MOTION DIRECTION ANALYSIS (with correct finger pattern):')
print()

print('📈 ZOOM_IN motion patterns:')
print(f'  Count: {len(zoom_in_data)} samples')
print(f'  Delta X: {zoom_in_data["delta_x"].mean():.4f} ± {zoom_in_data["delta_x"].std():.4f}')
print(f'  Delta Y: {zoom_in_data["delta_y"].mean():.4f} ± {zoom_in_data["delta_y"].std():.4f}')
print(f'  Y < 0 (UP motion): {(zoom_in_data["delta_y"] < 0).sum()} samples ({(zoom_in_data["delta_y"] < 0).mean()*100:.1f}%)')
print(f'  Y > 0 (DOWN motion): {(zoom_in_data["delta_y"] > 0).sum()} samples ({(zoom_in_data["delta_y"] > 0).mean()*100:.1f}%)')

print()
print('📉 ZOOM_OUT motion patterns:')
print(f'  Count: {len(zoom_out_data)} samples')
print(f'  Delta X: {zoom_out_data["delta_x"].mean():.4f} ± {zoom_out_data["delta_x"].std():.4f}')
print(f'  Delta Y: {zoom_out_data["delta_y"].mean():.4f} ± {zoom_out_data["delta_y"].std():.4f}')
print(f'  Y < 0 (UP motion): {(zoom_out_data["delta_y"] < 0).sum()} samples ({(zoom_out_data["delta_y"] < 0).mean()*100:.1f}%)')
print(f'  Y > 0 (DOWN motion): {(zoom_out_data["delta_y"] > 0).sum()} samples ({(zoom_out_data["delta_y"] > 0).mean()*100:.1f}%)')

print()
print('🔍 YOUR TEST MOTIONS vs TRAINING:')
print('  Test zoom_in: delta_y = -0.346 (UP motion)')
print('  Test zoom_out: delta_y = +0.408 (DOWN motion)')
print()

# Check main axis patterns
zoom_in_axis_y = zoom_in_data[zoom_in_data['main_axis_y'] == 1]
zoom_in_axis_x = zoom_in_data[zoom_in_data['main_axis_x'] == 1]
zoom_out_axis_y = zoom_out_data[zoom_out_data['main_axis_y'] == 1]
zoom_out_axis_x = zoom_out_data[zoom_out_data['main_axis_x'] == 1]

print('📊 MAIN AXIS PATTERNS:')
print(f'  zoom_in - Y axis: {len(zoom_in_axis_y)} samples ({len(zoom_in_axis_y)/len(zoom_in_data)*100:.1f}%)')
print(f'  zoom_in - X axis: {len(zoom_in_axis_x)} samples ({len(zoom_in_axis_x)/len(zoom_in_data)*100:.1f}%)')
print(f'  zoom_out - Y axis: {len(zoom_out_axis_y)} samples ({len(zoom_out_axis_y)/len(zoom_out_data)*100:.1f}%)')
print(f'  zoom_out - X axis: {len(zoom_out_axis_x)} samples ({len(zoom_out_axis_x)/len(zoom_out_data)*100:.1f}%)')

print()
print('💡 CONFIDENCE ISSUE ANALYSIS:')
print('Possible reasons for low confidence (43-48%):')
print('1. Motion magnitude too large compared to training')
print('2. Feature normalization issues')
print('3. Other features (left hand) affecting prediction')
print('4. Model needs more similar training samples')