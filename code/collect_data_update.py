import os
import csv
import collections
import msvcrt
import sys

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# === CONFIG ===
DEFAULT_CSV = 'training_results/gesture_data_compact.csv'  # Base dataset (read-only)
BUFFER_SIZE = 60
SMOOTHING_WINDOW = 3
MIN_FRAMES = 12
MIN_CONFIDENCE = 0.7

# Available gestures for update
AVAILABLE_GESTURES = [
    'home', 'end', 'next_slide', 'previous_slide',
    'rotate_left', 'rotate_right', 'rotate_up', 'rotate_down',
    'zoom_in', 'zoom_out'
]

# User-specific data collection
COLLECTED_SAMPLES = []  # Store samples for real-time analysis
USER_NAME = ""  # Will be set during initialization
CUSTOM_CSV = ""  # Will be set based on user name

# Conflict detection - load reference data once
REFERENCE_DATA = None  # Will be loaded from gesture_data_compact.csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

LEFT_COLUMNS = [f'left_finger_state_{i}' for i in range(5)]
RIGHT_COLUMNS = [f'right_finger_state_{i}' for i in range(5)]
MOTION_COLUMNS = [
    'motion_x_start', 'motion_y_start',
    'motion_x_mid', 'motion_y_mid',
    'motion_x_end', 'motion_y_end',
]
FEATURE_COLUMNS = ['main_axis_x', 'main_axis_y', 'delta_x', 'delta_y']

def load_reference_data():
    """Load reference data for conflict detection"""
    global REFERENCE_DATA
    if REFERENCE_DATA is None:
        try:
            REFERENCE_DATA = pd.read_csv('training_results/gesture_data_compact.csv')
            print(f"✅ Loaded reference data: {len(REFERENCE_DATA)} samples")
        except Exception as e:
            print(f"❌ Failed to load reference data: {e}")
            REFERENCE_DATA = pd.DataFrame()  # Empty fallback
    return REFERENCE_DATA

def get_motion_direction(delta_x, delta_y, threshold=0.01):
    """Get primary motion direction from delta values"""
    abs_x = abs(delta_x)
    abs_y = abs(delta_y)
    
    if abs_x < threshold and abs_y < threshold:
        return "static"
    
    if abs_x > abs_y:
        return "right" if delta_x > 0 else "left"
    else:
        return "down" if delta_y > 0 else "up"

def check_gesture_conflict(left_states, right_states, delta_x, delta_y, target_gesture):
    """
    Check if gesture conflicts with existing reference data
    Returns: (has_conflict, conflict_message)
    """
    ref_data = load_reference_data()
    if ref_data.empty:
        return False, "No reference data"
    
    # Get motion direction of current gesture
    current_direction = get_motion_direction(delta_x, delta_y)
    
    # Check against all reference samples
    for _, row in ref_data.iterrows():
        # Check finger pattern match (both hands)
        ref_left = [int(row[f'left_finger_state_{i}']) for i in range(5)]
        ref_right = [int(row[f'right_finger_state_{i}']) for i in range(5)]
        
        if ref_left == left_states and ref_right == right_states:
            # Same finger pattern! Check motion direction
            ref_direction = get_motion_direction(row['delta_x'], row['delta_y'])
            ref_gesture = row['pose_label']
            
            if current_direction == ref_direction:
                # Same finger + same direction = CONFLICT!
                return True, f"❌ CONFLICT: Same fingers L{left_states}R{right_states} + {ref_direction} direction as existing '{ref_gesture}'"
            # If different direction, continue checking (no conflict with this sample)
    
    return False, f"✅ OK: No conflicts found (direction: {current_direction})"

def get_finger_states(hand_landmarks, handedness_label):
    states = [0, 0, 0, 0, 0]
    if hand_landmarks is None:
        return states

    wrist = hand_landmarks.landmark[0]
    mcp_middle = hand_landmarks.landmark[9]
    mcp_pinky = hand_landmarks.landmark[17]
    v1 = [mcp_middle.x - wrist.x, mcp_middle.y - wrist.y]
    v2 = [mcp_pinky.x - wrist.x, mcp_pinky.y - wrist.y]
    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
    palm_facing = 1 if cross_z > 0 else -1

    if handedness_label == 'Right':
        if palm_facing > 0:
            states[0] = 1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0
        else:
            states[0] = 1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0
    else:
        if palm_facing > 0:
            states[0] = 1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0
        else:
            states[0] = 1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0

    states[1] = 1 if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y else 0
    states[2] = 1 if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y else 0
    states[3] = 1 if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y else 0
    states[4] = 1 if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y else 0
    return states


def is_fist(hand_landmarks):
    if hand_landmarks is None:
        return False
    bent = 0
    if hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y:
        bent += 1
    if hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y:
        bent += 1
    if hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y:
        bent += 1
    if hand_landmarks.landmark[20].y > hand_landmarks.landmark[17].y:
        bent += 1
    return bent >= 3


def ensure_capture_csv_exists(csv_path):
    if os.path.isfile(csv_path):
        return
    columns = ['instance_id', 'pose_label'] + LEFT_COLUMNS + RIGHT_COLUMNS + MOTION_COLUMNS + FEATURE_COLUMNS
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)


def next_instance_id(csv_path):
    if not os.path.isfile(csv_path):
        return 1
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return 1
        return int(df['instance_id'].max()) + 1
    except Exception:
        return 1


def smooth_points(buffer):
    if not buffer:
        return []
    window = SMOOTHING_WINDOW
    right_points = [entry for entry in buffer]
    smoothed = []
    for idx in range(len(right_points)):
        start = max(0, idx - window // 2)
        end = min(len(right_points), idx + window // 2 + 1)
        segment = right_points[start:end]
        smoothed.append(np.mean(segment, axis=0))
    return smoothed


def compute_motion_features(smoothed):
    if len(smoothed) < 2:
        return None
    start = smoothed[0]
    mid = smoothed[len(smoothed) // 2]
    end = smoothed[-1]
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    if abs(dx) >= abs(dy):
        main_x, main_y = 1, 0
        delta_x, delta_y = dx, 0.0
    else:
        main_x, main_y = 0, 1
        delta_x, delta_y = 0.0, dy
    return {
        'start': start,
        'mid': mid,
        'end': end,
        'main_axis_x': main_x,
        'main_axis_y': main_y,
        'delta_x': float(delta_x),
        'delta_y': float(delta_y),
    }


# === REAL-TIME CONFLICT DETECTION ===
def extract_finger_pattern_from_states(right_states):
    """Extract active fingers from finger states"""
    active_fingers = []
    for i, state in enumerate(right_states):
        if state == 1:  # Finger extended
            active_fingers.append(i)
    return set(active_fingers)


def extract_direction_from_features(features):
    """Extract normalized direction from motion features"""
    dx = features['delta_x']
    dy = features['delta_y']
    magnitude = np.sqrt(dx**2 + dy**2)
    
    if magnitude > 0:
        return (dx / magnitude, dy / magnitude), magnitude
    return (0, 0), 0


def check_realtime_conflict(new_sample, pose_label, existing_csv='training_results/gesture_data_compact.csv'):
    """
    Check if new sample conflicts with existing gestures using strict finger+direction matching
    Returns: (has_conflict, conflict_message, conflicting_gestures)
    """
    try:
        # Extract finger states and motion from new sample
        left_states = new_sample['left_states']
        right_states = new_sample['right_states']
        features = new_sample['features']
        delta_x = features['delta_x']
        delta_y = features['delta_y']
        
        # Use the new conflict detection logic
        has_conflict, conflict_msg = check_gesture_conflict(left_states, right_states, delta_x, delta_y, pose_label)
        
        if has_conflict:
            return True, conflict_msg, []
        else:
            return False, conflict_msg, []
            
    except Exception as e:
        return False, f"Error checking conflicts: {e}", []


# Global variable to store samples for current session
SESSION_SAMPLES = []

def arrow_menu(title, options, descriptions=None):
    """
    Interactive menu với arrow keys và Enter
    """
    if not options:
        return None
        
    current_index = 0
    
    def print_menu():
        os.system('cls' if os.name == 'nt' else 'clear')
        print(title)
        print("=" * len(title))
        print()
        
        for i, option in enumerate(options):
            prefix = "► " if i == current_index else "  "
            desc = f" - {descriptions[i]}" if descriptions and i < len(descriptions) else ""
            
            if i == current_index:
                print(f"\033[92m{prefix}{option}{desc}\033[0m")  # Green highlight
            else:
                print(f"{prefix}{option}{desc}")
        
        print()
        print("🔼🔽 Mũi tên lên/xuống | ⏎ Enter để chọn | ESC để thoát")
    
    while True:
        print_menu()
        
        # Get key input
        key = msvcrt.getch()
        
        if key == b'\xe0':  # Arrow key prefix
            key = msvcrt.getch()
            if key == b'H':  # Up arrow
                current_index = (current_index - 1) % len(options)
            elif key == b'P':  # Down arrow
                current_index = (current_index + 1) % len(options)
        elif key == b'\r':  # Enter
            return current_index
        elif key == b'\x1b':  # ESC
            return None
        elif key == b'q' or key == b'Q':  # Q to quit
            return None

def yes_no_menu(question):
    """
    Yes/No menu với arrow keys
    """
    options = ["Có (Yes)", "Không (No)"]
    result = arrow_menu(question, options)
    
    if result is None:
        return None
    return result == 0  # True for Yes, False for No

def save_capture(instance_id, pose_label, left_states, right_states, features, csv_path):
    ensure_capture_csv_exists(csv_path)
    
    # Store sample for real-time analysis
    sample = {
        'right_states': right_states,
        'features': features,
        'pose_label': pose_label
    }
    COLLECTED_SAMPLES.append(sample)
    
    # Store sample for session (will be saved in standard format)
    user_row = {
        'finger_thumb': right_states[0],
        'finger_index': right_states[1],
        'finger_middle': right_states[2], 
        'finger_ring': right_states[3],
        'finger_pinky': right_states[4],
        'start': features['start'],
        'mid': features['mid'],
        'end': features['end'],
        'main_axis_x': features['main_axis_x'],
        'main_axis_y': features['main_axis_y'],
        'motion_x': features['delta_x'],
        'motion_y': features['delta_y']
    }
    SESSION_SAMPLES.append(user_row)
    
    # Save to original format (for compatibility)
    row = [instance_id, pose_label]
    row += [int(v) for v in left_states]
    row += [int(v) for v in right_states]
    start, mid, end = features['start'], features['mid'], features['end']
    row += [float(start[0]), float(start[1]), float(mid[0]), float(mid[1]), float(end[0]), float(end[1])]
    row += [features['main_axis_x'], features['main_axis_y'], features['delta_x'], features['delta_y']]
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(row)

def save_session_to_user_folder(pose_label):
    """Save all session samples to standard format CSV files"""
    if not SESSION_SAMPLES:
        return None
        
    # Create user folder structure
    user_folder = f"user_{USER_NAME}"
    raw_data_folder = os.path.join(user_folder, "raw_data")
    os.makedirs(raw_data_folder, exist_ok=True)
    
    # Convert session samples to standard format
    standard_rows = []
    
    for i, sample in enumerate(SESSION_SAMPLES):
        # Create standard format row with real motion coordinates
        start = sample['start']
        mid = sample['mid'] 
        end = sample['end']
        
        row = [
            i + 1,         # instance_id
            pose_label,    # pose_label
            0, 0, 0, 0, 0, # left_finger_state_0-4 (always 0 in our case)
            sample['finger_thumb'],    # right_finger_state_0
            sample['finger_index'],    # right_finger_state_1  
            sample['finger_middle'],   # right_finger_state_2
            sample['finger_ring'],     # right_finger_state_3
            sample['finger_pinky'],    # right_finger_state_4
            float(start[0]), float(start[1]),  # motion_x_start, motion_y_start (REAL coordinates)
            float(mid[0]), float(mid[1]),      # motion_x_mid, motion_y_mid (REAL coordinates)  
            float(end[0]), float(end[1]),      # motion_x_end, motion_y_end (REAL coordinates)
            sample['main_axis_x'],  # main_axis_x (from features)
            sample['main_axis_y'],  # main_axis_y (from features)
            sample['motion_x'],     # delta_x
            sample['motion_y']      # delta_y
        ]
        standard_rows.append(row)
    
    # Standard column names
    columns = [
        'instance_id', 'pose_label',
        'left_finger_state_0', 'left_finger_state_1', 'left_finger_state_2', 'left_finger_state_3', 'left_finger_state_4',
        'right_finger_state_0', 'right_finger_state_1', 'right_finger_state_2', 'right_finger_state_3', 'right_finger_state_4',
        'motion_x_start', 'motion_y_start', 'motion_x_mid', 'motion_y_mid', 'motion_x_end', 'motion_y_end',
        'main_axis_x', 'main_axis_y', 'delta_x', 'delta_y'
    ]
    
    # Save individual gesture file (raw_data folder)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    user_csv = os.path.join(raw_data_folder, f"gesture_data_custom_{USER_NAME}_{pose_label}_{timestamp}.csv")
    
    individual_df = pd.DataFrame(standard_rows, columns=columns)
    individual_df.to_csv(user_csv, index=False)
    
    print(f"\n💾 Đã lưu {len(SESSION_SAMPLES)} mẫu (format chuẩn) vào: {user_csv}")
    
    # Also update master CSV file in user folder
    update_master_csv_in_user_folder(pose_label, standard_rows, columns)
    
    return user_csv

def update_master_csv_in_user_folder(pose_label, standard_rows, columns):
    """Update master CSV file in user folder with standard format"""
    user_folder = f"user_{USER_NAME}"
    master_csv = os.path.join(user_folder, f"gesture_data_custom_{USER_NAME}.csv")
    
    # Get current max instance_id if file exists
    start_id = 1
    if os.path.exists(master_csv):
        try:
            existing_df = pd.read_csv(master_csv)
            if not existing_df.empty and 'instance_id' in existing_df.columns:
                start_id = existing_df['instance_id'].max() + 1
        except:
            start_id = 1
    
    # Update instance_id for master file (continuous numbering)
    for i, row in enumerate(standard_rows):
        row[0] = start_id + i  # Update instance_id
    
    # Create DataFrame for new data
    new_df = pd.DataFrame(standard_rows, columns=columns)
    
    # Append to existing file or create new one
    if os.path.exists(master_csv):
        # Append to existing
        new_df.to_csv(master_csv, mode='a', header=False, index=False)
        print(f"📄 Đã thêm {len(standard_rows)} mẫu vào file tổng: {master_csv}")
    else:
        # Create new file with header
        new_df.to_csv(master_csv, index=False)
        print(f"📄 Đã tạo file tổng mới: {master_csv} với {len(standard_rows)} mẫu")


def get_existing_users():
    """Get list of existing users from user folders"""
    existing_users = []
    for folder_name in os.listdir('.'):
        if os.path.isdir(folder_name) and folder_name.startswith('user_'):
            username = folder_name[5:]  # Remove 'user_' prefix
            existing_users.append(username)
    return existing_users

def check_user_gesture_exists(username, gesture_name):
    """Check if user already has data for this gesture"""
    user_folder = f"user_{username}"
    raw_data_folder = os.path.join(user_folder, "raw_data")
    
    if not os.path.exists(raw_data_folder):
        return False, []
    
    # Find existing files for this gesture
    existing_files = []
    for filename in os.listdir(raw_data_folder):
        if filename.startswith(f"gesture_data_custom_{username}_{gesture_name}") and filename.endswith('.csv'):
            existing_files.append(filename)
    
    return len(existing_files) > 0, existing_files

def get_user_info():
    """
    Get user name with arrow key selection
    """
    existing_users = get_existing_users()
    
    if existing_users:
        # Prepare options
        options = existing_users + ["Tạo user mới"]
        descriptions = ["User có sẵn"] * len(existing_users) + ["Nhập tên user mới"]
        
        choice = arrow_menu("=== CHỌN USER ===", options, descriptions)
        
        if choice is None:
            return None, None
        elif choice < len(existing_users):
            # Selected existing user
            selected_user = existing_users[choice]
            custom_csv = f'gesture_data_custom_{selected_user}.csv'
            print(f'\n✅ Đã chọn user: {selected_user}')
            return selected_user, custom_csv
        else:
            # Create new user
            return create_new_user()
    else:
        print('📂 Chưa có user nào. Tạo user mới:')
        return create_new_user()

def create_new_user():
    """Create new user"""
    while True:
        user_name = input('Nhập tên user mới (a-z, 0-9, _): ').strip()
        if not user_name:
            print('[WARN] Không có tên -> thoát.')
            return None, None
            
        # Validate user name
        if user_name.replace('_', '').replace('-', '').isalnum():
            custom_csv = f'gesture_data_custom_{user_name}.csv'
            print(f'\n✅ Tạo user mới: {user_name}')
            print(f'📁 File dữ liệu: {custom_csv}')
            return user_name, custom_csv
        else:
            print('[ERROR] Tên chỉ được chứa chữ cái, số và dấu gạch dưới.')


def select_gesture_to_update():
    """
    Arrow key selection for gestures with conflict checking
    """
    # Prepare gesture options with status
    options = []
    descriptions = []
    
    for gesture in AVAILABLE_GESTURES:
        has_data, files = check_user_gesture_exists(USER_NAME, gesture)
        options.append(gesture)
        if has_data:
            descriptions.append(f"Đã có {len(files)} file")
        else:
            descriptions.append("Gesture mới")
    
    choice = arrow_menu(f"=== CHỌN GESTURE CHO USER: {USER_NAME} ===", options, descriptions)
    
    if choice is None:
        return None
    
    selected_gesture = AVAILABLE_GESTURES[choice]
    
    # Check if user already has this gesture
    has_data, existing_files = check_user_gesture_exists(USER_NAME, selected_gesture)
    
    if has_data:
        # Show conflict and ask for confirmation
        conflict_question = f"⚠️  Gesture '{selected_gesture}' đã tồn tại ({len(existing_files)} file)\n❓ Xóa data cũ và bắt đầu lại?"
        
        continue_choice = yes_no_menu(conflict_question)
        
        if continue_choice is None:
            return None
        elif continue_choice:
            # Delete existing files
            user_folder = f"user_{USER_NAME}"
            raw_data_folder = os.path.join(user_folder, "raw_data")
            deleted_count = 0
            
            for filename in existing_files:
                filepath = os.path.join(raw_data_folder, filename)
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except Exception as e:
                    print(f'❌ Lỗi xóa {filename}: {e}')
            
            print(f'\n🗑️  Đã xóa {deleted_count} file cũ cho gesture "{selected_gesture}"')
            print(f'✅ Bắt đầu thu thập mới cho: {selected_gesture}')
            return selected_gesture
        else:
            # User chose not to overwrite, go back to selection
            print('\n🔄 Chọn gesture khác...')
            return select_gesture_to_update()
    else:
        print(f'\n✅ Đã chọn gesture mới: {selected_gesture}')
        return selected_gesture


def main():
    global USER_NAME, CUSTOM_CSV, SESSION_SAMPLES
    
    print('=== UPDATE GESTURE DEFINITION ===')
    
    # Get user info and setup custom CSV
    USER_NAME, CUSTOM_CSV = get_user_info()
    if not USER_NAME:
        return
    
    # Select gesture to update
    pose_label = select_gesture_to_update()
    if not pose_label:
        return
    
    # Clear session samples
    SESSION_SAMPLES = []

    instance_counter = next_instance_id(CUSTOM_CSV)
    print(f"\nBat dau thu thap du lieu cho gesture: '{pose_label}'")
    print("\nHuong dan:")
    print("  - Dua ca 2 tay vao khung hinh.")
    print("  - Dieu chinh tay phai theo pose moi, tay trai mo.")
    print("  - Nam tay trai de bat dau, giu chuyen dong tay phai.")
    print("  - Mo tay trai de ket thuc 1 lan ghi. Bam 'q' de thoat hoan toan.")
    print(f"  - Muc tieu: Thu thap 5 mau cho '{pose_label}'\n")

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Update Gesture Data', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Update Gesture Data', 1280, 960)

    state = 'WAIT'
    buffer = collections.deque(maxlen=BUFFER_SIZE)
    saved_count = 0
    current_left_states = None
    current_right_states = None
    conflict_message = ""
    conflict_detected = False

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print('[ERROR] Khong doc duoc frame tu camera.')
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            left_landmarks = None
            right_landmarks = None
            left_conf = 0.0
            right_conf = 0.0

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[idx].classification[0]
                    label = handedness.label
                    score = handedness.score
                    if label == 'Left':
                        left_landmarks = hand_landmarks
                        left_conf = score
                    elif label == 'Right':
                        right_landmarks = hand_landmarks
                        right_conf = score
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            left_is_fist = is_fist(left_landmarks)

            if state == 'WAIT':
                cv2.putText(frame, 'State: WAIT (Close LEFT fist to start)', (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if left_is_fist and left_conf > MIN_CONFIDENCE and right_conf > MIN_CONFIDENCE and right_landmarks:
                    current_left_states = get_finger_states(left_landmarks, 'Left')
                    current_right_states = get_finger_states(right_landmarks, 'Right')
                    buffer.clear()
                    state = 'RECORD'
                    print('\n>>> Trigger on. Bat dau ghi chuyen dong...')
                elif left_is_fist and right_landmarks is None:
                    cv2.putText(frame, 'Can tay phai trong khung!', (20, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            elif state == 'RECORD':
                cv2.putText(frame, 'State: RECORD (Open LEFT fist to stop)', (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
                if right_landmarks and right_conf > MIN_CONFIDENCE:
                    wrist = right_landmarks.landmark[0]
                    buffer.append(np.array([wrist.x, wrist.y], dtype=float))
                if not left_is_fist:
                    state = 'PROCESS'
            elif state == 'PROCESS':
                cv2.putText(frame, 'State: PROCESSING...', (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                if len(buffer) < MIN_FRAMES:
                    print(f'[WARN] Buffer qua ngan ({len(buffer)}) -> bo qua.')
                elif current_right_states is None:
                    print('[WARN] Khong co finger state tay phai -> bo qua.')
                else:
                    smoothed = smooth_points(list(buffer))
                    features = compute_motion_features(smoothed)
                    if features is None:
                        print('[WARN] Khong tinh duoc motion features.')
                    else:
                        # Real-time conflict detection
                        sample_data = {
                            'left_states': current_left_states if current_left_states else [0, 0, 0, 0, 0],
                            'right_states': current_right_states,
                            'features': features
                        }
                        
                        has_conflict, conflict_msg, conflicting_gestures = check_realtime_conflict(sample_data, pose_label)
                        
                        if has_conflict:
                            print(f"🚨 {conflict_msg}")
                            if conflicting_gestures:
                                for conflict in conflicting_gestures:
                                    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
                                    fingers_str = ', '.join([finger_names[i] for i in conflict['fingers']])
                                    print(f"   -> '{conflict['name']}': fingers({fingers_str}), similarity({conflict['similarity']:.2f})")
                            conflict_message = conflict_msg
                            conflict_detected = True
                        else:
                            # Store sample for session (will be saved to user folder only)
                            user_row = {
                                'finger_thumb': current_right_states[0],
                                'finger_index': current_right_states[1],
                                'finger_middle': current_right_states[2], 
                                'finger_ring': current_right_states[3],
                                'finger_pinky': current_right_states[4],
                                'start': features['start'],
                                'mid': features['mid'],
                                'end': features['end'],
                                'main_axis_x': features['main_axis_x'],
                                'main_axis_y': features['main_axis_y'],
                                'motion_x': features['delta_x'],
                                'motion_y': features['delta_y']
                            }
                            SESSION_SAMPLES.append(user_row)
                            
                            print(f"[INFO] Luu mau #{instance_counter} cho pose '{pose_label}' -> {USER_NAME}. {conflict_msg}")
                            instance_counter += 1
                            saved_count += 1
                            conflict_message = ""
                            conflict_detected = False
                            
                            # Check if we have enough samples (target: 5)
                            if saved_count >= 5:
                                print(f"✅ Da thu thap du 5 mau cho '{pose_label}'!")
                                print("Bam 'q' de thoat hoac tiep tuc thu thap them.")
                            
                buffer.clear()
                current_left_states = None
                current_right_states = None
                state = 'WAIT'

            # Display info
            cv2.putText(frame, f'User: {USER_NAME}', (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f'UPDATE: {pose_label}', (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f'Samples: {saved_count}/5', (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                        (0, 255, 0) if saved_count >= 5 else (255, 255, 255), 2)
            
            # Conflict warning
            if conflict_detected and conflict_message:
                # Display conflict warning
                cv2.putText(frame, '🚨 CONFLICT DETECTED!', (20, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # Show which gestures conflict
                lines = conflict_message.split(':')
                if len(lines) > 1:
                    conflict_text = lines[1].strip()
                    cv2.putText(frame, conflict_text, (20, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, 'Change fingers or direction!', (20, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Progress indicator
            elif saved_count >= 5:
                cv2.putText(frame, 'COMPLETE! Press q to exit', (20, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # Show live finger detection for feedback
                if current_right_states:
                    active_fingers = [i for i, state in enumerate(current_right_states) if state == 1]
                    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
                    if active_fingers:
                        fingers_text = ', '.join([finger_names[i] for i in active_fingers])
                        cv2.putText(frame, f'Fingers: {fingers_text}', (20, 210),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Update Gesture Data', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nĐã thoát. Tổng số lần ghi cho '{pose_label}': {saved_count}.")
        
        # Save all session samples to ONE CSV file
        if saved_count > 0:
            user_csv_path = save_session_to_user_folder(pose_label)
            if user_csv_path:
                print(f"✅ Thu thập thành công! Có thể sử dụng {saved_count} mẫu để update gesture '{pose_label}'.")
                print(f"📁 File đã lưu: {user_csv_path}")
            else:
                print("❌ Lỗi lưu file user CSV.")
        elif saved_count > 0:
            print(f"⚠️  Chỉ thu thập được {saved_count}/5 mẫu. Cần thêm {5-saved_count} mẫu nữa.")
        else:
            print("❌ Không thu thập được mẫu nào.")


if __name__ == '__main__':
    main()