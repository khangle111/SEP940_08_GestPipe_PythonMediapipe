import os
import pickle
import collections
import time

import cv2
import mediapipe as mp
import numpy as np

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Auto-detect user folder based on current path
path_parts = BASE_DIR.split(os.sep)
user_folder = None

# Check if we're in a user folder (contains 'user_' in path)
for part in path_parts:
    if part.startswith('user_'):
        user_folder = part
        break

if user_folder:
    # Running from user folder (e.g., user_Khang, user_ABC, etc.)
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    TRAINING_RESULTS_DIR = os.path.join(BASE_DIR, "training_results")
    print(f"🔄 Using {user_folder}'s personal model and training data")
else:
    # Running from main code folder
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    TRAINING_RESULTS_DIR = os.path.join(BASE_DIR, "training_results")
    print("🔄 Using main model and training data")

MODEL_PKL = os.path.join(MODELS_DIR, 'motion_svm_model.pkl')
SCALER_PKL = os.path.join(MODELS_DIR, 'motion_scaler.pkl')
STATIC_DYNAMIC_PKL = os.path.join(MODELS_DIR, 'static_dynamic_classifier.pkl')
BUFFER_SIZE = 60
SMOOTHING_WINDOW = 3
MIN_FRAMES = 12
MIN_CONFIDENCE = 0.7
MIN_DELTA_MAG = 0.0005  # Lowered for close distance (70-90cm)
DELTA_WEIGHT = 10.0  # Updated to match training script
DISPLAY_DURATION = 3.0  # Display prediction for 3 seconds
MIN_PREDICTION_CONFIDENCE = 0.60  # Increased for better confidence threshold

def load_gesture_patterns_from_training_data(compact_file=None):
    """Auto-generate gesture patterns from training data với error handling"""
    try:
        import pandas as pd

        # Default path
        if compact_file is None:
            compact_file = os.path.join(TRAINING_RESULTS_DIR, 'gesture_data_compact.csv')

        if not os.path.exists(compact_file):
            print(f"⚠️  Warning: Compact dataset not found: {compact_file}")
            print("   Using fallback patterns...")
            return get_fallback_patterns()

        df = pd.read_csv(compact_file)
        print(f"✅ Loaded compact dataset: {len(df)} samples")

        # Detect column format
        if 'pose_label' in df.columns:
            gesture_col = 'pose_label'
            finger_prefix = 'right_finger_state_'
        elif 'gesture' in df.columns:
            gesture_col = 'gesture'
            finger_prefix = 'finger_'
        else:
            # Try to detect gesture column
            possible_gesture_cols = [col for col in df.columns if 'gesture' in col.lower() or 'pose' in col.lower()]
            if possible_gesture_cols:
                gesture_col = possible_gesture_cols[0]
                print(f"   Detected gesture column: {gesture_col}")
            else:
                print("⚠️  Warning: No gesture column found, using fallback patterns")
                return get_fallback_patterns()

        # Detect finger columns
        finger_cols = [col for col in df.columns if 'finger' in col and 'state' in col]
        if not finger_cols:
            # Try alternative patterns
            finger_cols = [col for col in df.columns if 'finger' in col or 'thumb' in col or 'index' in col]

        if len(finger_cols) < 5:
            print(f"⚠️  Warning: Only found {len(finger_cols)} finger columns, expected 5")
            return get_fallback_patterns()

        print(f"   Using gesture column: {gesture_col}")
        print(f"   Using finger columns: {finger_cols[:5]}")

        patterns = {}
        unique_gestures = sorted(df[gesture_col].unique())

        for gesture in unique_gestures:
            gesture_data = df[df[gesture_col] == gesture]

            # Count finger patterns
            pattern_counts = {}
            for _, row in gesture_data.iterrows():
                try:
                    # Extract finger states (first 5 finger columns)
                    finger_states = []
                    for i in range(min(5, len(finger_cols))):
                        col = finger_cols[i]
                        state = int(float(row[col])) if pd.notna(row[col]) else 0
                        finger_states.append(state)

                    pattern_tuple = tuple(finger_states)
                    pattern_counts[pattern_tuple] = pattern_counts.get(pattern_tuple, 0) + 1

                except (ValueError, KeyError) as e:
                    continue  # Skip invalid rows

            # Sort by frequency and take top patterns
            if pattern_counts:
                sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
                top_patterns = []
                total_samples = len(gesture_data)

                for pattern, count in sorted_patterns:
                    # Take patterns that appear in at least 3% of samples
                    if count / total_samples > 0.03:
                        top_patterns.append(list(pattern))
                    if len(top_patterns) >= 3:  # Max 3 patterns per gesture
                        break

                patterns[gesture] = top_patterns
                print(f"   {gesture}: {len(top_patterns)} patterns from {len(gesture_data)} samples")
            else:
                print(f"   {gesture}: No valid patterns found")

        if not patterns:
            print("⚠️  Warning: No patterns loaded, using fallback")
            return get_fallback_patterns()

        print(f"✅ Loaded {len(patterns)} gesture patterns")
        return patterns

    except Exception as e:
        print(f"⚠️  Warning: Could not load patterns from training data: {e}")
        print("   Using fallback patterns...")
        return get_fallback_patterns()


def get_fallback_patterns():
    """Fallback patterns khi không load được từ training data"""
    return {
        'home': [[1,0,0,0,0]],
        'end': [[0,0,0,0,1]],
        'next_slide': [[0,1,1,0,0]],
        'previous_slide': [[0,1,1,0,0]],
        'rotate_right': [[1,1,0,0,0]],
        'rotate_left': [[1,1,0,0,0]],
        'rotate_up': [[1,1,0,0,0]],
        'rotate_down': [[1,1,0,0,0]],
        'zoom_in': [[1,1,1,0,0]],
        'zoom_out': [[1,1,1,0,0]],
        'zoom_in_slide': [[0,1,1,0,0]],
        'zoom_out_slide': [[0,1,1,0,0]],
        'start_present': [[1,1,1,1,1]],
        'end_present': [[1,1,1,1,1]],
    }


# Static gesture detection - Adjusted for close distance (70-90cm)
STATIC_HOLD_TIME = 1.0  # Reduced for close distance
STATIC_DELTA_THRESHOLD = 0.003  # Much lower for close distance
STATIC_DETECTION_THRESHOLD = 0.002  # Lower for close distance motion detection

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

LEFT_COLS = [f'left_finger_state_{i}' for i in range(5)]
RIGHT_COLS = [f'right_finger_state_{i}' for i in range(5)]
MOTION_COLS = ['main_axis_x', 'main_axis_y', 'delta_x', 'delta_y']


def load_model():
    """Load trained model, scaler, and static/dynamic classifier với error handling"""
    if not os.path.exists(MODEL_PKL):
        raise FileNotFoundError(f"Model file not found: {MODEL_PKL}")
    if not os.path.exists(SCALER_PKL):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PKL}")

    try:
        with open(MODEL_PKL, 'rb') as f:
            model_data = pickle.load(f)

        with open(SCALER_PKL, 'rb') as f:
            scaler = pickle.load(f)

        # Validate model format
        if 'model' not in model_data or 'label_encoder' not in model_data:
            raise ValueError("Invalid model format: missing 'model' or 'label_encoder'")

        model = model_data['model']
        label_encoder = model_data['label_encoder']

        # Check scaler
        if not hasattr(scaler, 'transform'):
            raise ValueError("Invalid scaler: missing transform method")

        # Get expected features from scaler
        expected_features = getattr(scaler, 'n_features_in_', None)
        if expected_features:
            print(f"✅ Model expects {expected_features} features")
        else:
            print("⚠️  Warning: Could not determine expected features from scaler")

        print(f"✅ Model loaded: {type(model).__name__}")
        print(f"✅ Available gestures: {len(label_encoder.classes_)} - {list(label_encoder.classes_)}")

        # Load static/dynamic classifier (optional)
        static_dynamic_data = None
        if os.path.exists(STATIC_DYNAMIC_PKL):
            try:
                with open(STATIC_DYNAMIC_PKL, 'rb') as f:
                    static_dynamic_data = pickle.load(f)
                print("✅ Static/dynamic classifier loaded")
            except Exception as e:
                print(f"⚠️  Warning: Could not load static/dynamic classifier: {e}")

        return model, label_encoder, scaler, static_dynamic_data

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


# Load gesture patterns from training data
GESTURE_PATTERNS = load_gesture_patterns_from_training_data()
print(f"Loaded {len(GESTURE_PATTERNS)} gesture patterns from training data")


def calculate_confidence_boost(predicted_gesture, finger_pattern, gesture_patterns):
    """Calculate confidence boost based on pattern matching"""
    if predicted_gesture not in gesture_patterns:
        return 0.0

    # Convert finger pattern to tuple for comparison
    current_pattern = tuple(finger_pattern)

    # Check if current pattern matches any training pattern for this gesture
    matching_patterns = gesture_patterns[predicted_gesture]

    if current_pattern in [tuple(p) for p in matching_patterns]:
        return 0.2  # Boost for exact match
    else:
        # Calculate similarity (how many fingers match)
        max_similarity = 0
        for training_pattern in matching_patterns:
            similarity = sum(a == b for a, b in zip(current_pattern, training_pattern)) / 5.0
            max_similarity = max(max_similarity, similarity)

        if max_similarity >= 0.8:  # 4 out of 5 fingers match
            return 0.1
        elif max_similarity >= 0.6:  # 3 out of 5 fingers match
            return 0.05
        else:
            return -0.1  # Penalty for poor match


def prepare_features(left_finger_states, right_finger_states, motion_features, scaler):
    """Prepare feature vector for prediction (match training preprocessing)"""
    # Finger features (10): left (5) + right (5)
    finger_features = left_finger_states + right_finger_states

    # Motion features (8): main_axis_x, main_axis_y, delta_x, delta_y, and direction features
    main_x = motion_features['main_axis_x']
    main_y = motion_features['main_axis_y']
    delta_x = motion_features['delta_x'] * 10.0  # Same DELTA_WEIGHT as training
    delta_y = motion_features['delta_y'] * 10.0

    # Direction features
    motion_left = 1.0 if motion_features['delta_x'] < 0 else 0.0
    motion_right = 1.0 if motion_features['delta_x'] > 0 else 0.0
    motion_up = 1.0 if motion_features['delta_y'] < 0 else 0.0
    motion_down = 1.0 if motion_features['delta_y'] > 0 else 0.0

    motion_vals = [main_x, main_y, delta_x, delta_y, motion_left, motion_right, motion_up, motion_down]

    # Convert to numpy arrays
    finger_array = np.array([finger_features], dtype=float)
    motion_array = np.array([motion_vals], dtype=float)

    # Scale motion features (finger states stay unscaled)
    motion_scaled = scaler.transform(motion_array)

    # Combine: finger (unscaled) + motion (scaled)
    X = np.hstack([finger_array, motion_scaled])

    return X


def get_finger_states(hand_landmarks, handedness_label):
    """Extract finger states - same as collect_data_hybrid.py"""
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


def is_trigger_closed(hand_landmarks):
    """Check if left hand is in trigger position (flexible fist detection)
    
    Accepts both patterns as trigger:
    - [0,0,0,0,0] : Complete fist
    - [1,0,0,0,0] : Fist with thumb extended (common detection error)
    """
    if hand_landmarks is None:
        return False
    
    # Get finger states for left hand
    left_states = get_finger_states(hand_landmarks, 'Left')
    
    # Accept both trigger patterns
    complete_fist = left_states == [0, 0, 0, 0, 0]
    thumb_extended_fist = left_states == [1, 0, 0, 0, 0]
    
    return complete_fist or thumb_extended_fist


def smooth_points(buffer):
    """Smooth motion points"""
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


def compute_motion_features(smoothed, is_static=False):
    """Compute motion features from smoothed points"""
    if len(smoothed) < 2:
        return None
    start = smoothed[0]
    mid = smoothed[len(smoothed) // 2]
    end = smoothed[-1]
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    
    # Calculate delta magnitude
    delta_mag = np.sqrt(dx**2 + dy**2)
    
    # For static gestures, always compute features (don't return None)
    # For dynamic gestures, require minimum motion
    if not is_static and delta_mag < MIN_DELTA_MAG:
        return None
    
    if abs(dx) >= abs(dy):
        main_x, main_y = 1, 0
        delta_x, delta_y = dx, 0.0
    else:
        main_x, main_y = 0, 1
        delta_x, delta_y = 0.0, dy
    
    # Add direction features (matching training script)
    motion_left = 1.0 if dx < 0 else 0.0
    motion_right = 1.0 if dx > 0 else 0.0
    motion_up = 1.0 if dy < 0 else 0.0
    motion_down = 1.0 if dy > 0 else 0.0
    
    return {
        'main_axis_x': main_x,
        'main_axis_y': main_y,
        'delta_x': float(delta_x),
        'delta_y': float(delta_y),
        'motion_left': motion_left,
        'motion_right': motion_right,
        'motion_up': motion_up,
        'motion_down': motion_down,
        'delta_magnitude': delta_mag,
    }


def prepare_features(left_states, right_states, motion_features, scaler):
    """Prepare features for prediction - same preprocessing as training"""
    # Combine finger states
    finger_feats = np.array(left_states + right_states, dtype=float).reshape(1, -1)

    # Calculate direction features
    motion_left = 1.0 if motion_features['delta_x'] < 0 else 0.0
    motion_right = 1.0 if motion_features['delta_x'] > 0 else 0.0
    motion_up = 1.0 if motion_features['delta_y'] < 0 else 0.0
    motion_down = 1.0 if motion_features['delta_y'] > 0 else 0.0

    # Apply delta weight and add direction features
    motion_array = np.array([[
        motion_features['main_axis_x'],
        motion_features['main_axis_y'],
        motion_features['delta_x'] * DELTA_WEIGHT,
        motion_features['delta_y'] * DELTA_WEIGHT,
        motion_left * DELTA_WEIGHT,
        motion_right * DELTA_WEIGHT,
        motion_up * DELTA_WEIGHT,
        motion_down * DELTA_WEIGHT
    ]], dtype=float)

    # Scale motion features
    motion_scaled = scaler.transform(motion_array)

    # Combine features
    X = np.hstack([finger_feats, motion_scaled])
    return X

    return X


def prepare_features(left_states, right_states, motion_features, scaler):
    """Prepare features for prediction với adaptive feature handling"""
    try:
        # Combine finger states (always 10 features) - unscaled
        finger_feats = np.array(left_states + right_states, dtype=float).reshape(1, -1)

        # Get expected features from scaler
        scaler_features = getattr(scaler, 'n_features_in_', 8)  # Scaler thường scale 8 motion features

        # Create motion features array
        motion_left = 1.0 if motion_features['delta_x'] < 0 else 0.0
        motion_right = 1.0 if motion_features['delta_x'] > 0 else 0.0
        motion_up = 1.0 if motion_features['delta_y'] < 0 else 0.0
        motion_down = 1.0 if motion_features['delta_y'] > 0 else 0.0

        if scaler_features == 8:
            # Standard case: scaler scales 8 motion features
            motion_array = np.array([[
                motion_features['main_axis_x'],
                motion_features['main_axis_y'],
                motion_features['delta_x'] * DELTA_WEIGHT,
                motion_features['delta_y'] * DELTA_WEIGHT,
                motion_left * DELTA_WEIGHT,
                motion_right * DELTA_WEIGHT,
                motion_up * DELTA_WEIGHT,
                motion_down * DELTA_WEIGHT
            ]], dtype=float)

            # Scale motion features
            motion_scaled = scaler.transform(motion_array)

            # Combine: 10 fingers (unscaled) + 8 motion (scaled) = 18 features
            X = np.hstack([finger_feats, motion_scaled])

        elif scaler_features == 4:
            # Alternative: scaler scales only 4 basic motion features
            motion_array = np.array([[
                motion_features['main_axis_x'],
                motion_features['main_axis_y'],
                motion_features['delta_x'] * DELTA_WEIGHT,
                motion_features['delta_y'] * DELTA_WEIGHT
            ]], dtype=float)

            motion_scaled = scaler.transform(motion_array)

            # Add direction features unscaled
            direction_features = np.array([[motion_left, motion_right, motion_up, motion_down]], dtype=float)
            motion_combined = np.hstack([motion_scaled, direction_features])

            X = np.hstack([finger_feats, motion_combined])

        else:
            # Unknown scaler format - try basic approach
            print(f"⚠️  Warning: Unexpected scaler features {scaler_features}, using basic adaptation")
            motion_array = np.array([[
                motion_features['main_axis_x'],
                motion_features['main_axis_y'],
                motion_features['delta_x'] * DELTA_WEIGHT,
                motion_features['delta_y'] * DELTA_WEIGHT
            ]], dtype=float)

            try:
                motion_scaled = scaler.transform(motion_array)
                X = np.hstack([finger_feats, motion_scaled])
            except:
                # If scaling fails, use unscaled
                X = np.hstack([finger_feats, motion_array])

        # Final validation
        expected_total = 10 + scaler_features  # fingers + motion
        if X.shape[1] != expected_total:
            print(f"⚠️  Warning: Feature mismatch - expected {expected_total}, got {X.shape[1]}")

        return X

    except Exception as e:
        print(f"❌ Error in prepare_features: {e}")
        # Emergency fallback - return 18 zeros
        return np.zeros((1, 18), dtype=float)


def prepare_static_features(left_states, right_states, delta_magnitude, static_scaler):
    """Prepare features for static/dynamic classification"""
    # Combine finger states + delta magnitude (matching training)
    finger_feats = np.array(left_states + right_states + [delta_magnitude], dtype=float).reshape(1, -1)
    
    # Scale features
    X_scaled = static_scaler.transform(finger_feats)
    return X_scaled

def is_static_gesture(motion_features, static_dynamic_data):
    """Determine if gesture is static or dynamic"""
    if static_dynamic_data is None:
        # Fallback: use delta magnitude threshold
        return motion_features['delta_magnitude'] < STATIC_DELTA_THRESHOLD
    
    # Use trained classifier (implement if needed)
    return motion_features['delta_magnitude'] < STATIC_DELTA_THRESHOLD


def main():
    print('=== GESTURE RECOGNITION TEST ===')
    
    # Load model
    try:
        model, label_encoder, scaler, static_dynamic_data = load_model()
        print(f"[INFO] Model loaded successfully!")
        print(f"[INFO] Available gestures: {list(label_encoder.classes_)}")
        if static_dynamic_data:
            static_gestures = static_dynamic_data.get('static_gestures', [])
            print(f"[INFO] Static gestures: {static_gestures}")
        else:
            print(f"[INFO] Using fallback static detection")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    print("Instructions:")
    print("  - Optimal distance: 70-90cm (close) or 2m+ (far)")
    print("  - Put both hands clearly in frame")
    print("  - Close LEFT fist to start recording gesture")
    print("  - For STATIC gestures: Keep RIGHT hand perfectly still for 1.5s")
    print("  - For DYNAMIC gestures: Move RIGHT hand with LARGE, CLEAR motions")
    print("  - Open LEFT fist to stop and predict")
    print("  - Press 'q' to quit")
    print("")

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Gesture Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gesture Recognition', 1280, 960)

    state = 'WAIT'
    buffer = collections.deque(maxlen=BUFFER_SIZE)
    current_left_states = None
    current_right_states = None
    
    # Static gesture tracking
    static_start_time = 0
    static_finger_states = None
    is_holding_static = False
    
    # Prediction display variables
    prediction_text = ""
    confidence_text = ""
    top_predictions_text = []
    debug_features_text = []
    prediction_start_time = 0

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print('[ERROR] Cannot read frame from camera.')
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            left_landmarks = None
            right_landmarks = None
            left_conf = 0.0
            right_conf = 0.0

            # Process hand detections
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

            left_is_trigger = is_trigger_closed(left_landmarks)

            if state == 'WAIT':
                # No text display - clean interface
                if left_is_trigger and left_conf > MIN_CONFIDENCE and right_conf > MIN_CONFIDENCE and right_landmarks:
                    current_left_states = get_finger_states(left_landmarks, 'Left')
                    current_right_states = get_finger_states(right_landmarks, 'Right') 
                    buffer.clear()
                    state = 'RECORD'
                    print('\n>>> Recording gesture...')

            elif state == 'RECORD':
                # Collect motion data
                if right_landmarks and right_conf > MIN_CONFIDENCE:
                    wrist = right_landmarks.landmark[0]
                    buffer.append(np.array([wrist.x, wrist.y], dtype=float))
                    
                    # Update right hand finger states continuously
                    current_right_states = get_finger_states(right_landmarks, 'Right')
                    
                    # Check for static gesture (consistent finger states + minimal motion)
                    if len(buffer) > 10:  # Need some buffer for motion calculation
                        recent_points = list(buffer)[-10:]  # Last 10 points
                        if len(recent_points) >= 2:
                            start_point = recent_points[0]
                            end_point = recent_points[-1]
                            recent_motion = np.sqrt((end_point[0] - start_point[0])**2 + 
                                                  (end_point[1] - start_point[1])**2)
                            
                    # Check for static gesture (consistent finger states + minimal motion)
                    if len(buffer) > 5:  # Need some buffer for stable detection
                        # Calculate motion over last few frames
                        recent_points = list(buffer)[-5:]  # Last 5 points
                        if len(recent_points) >= 2:
                            start_point = recent_points[0]
                            end_point = recent_points[-1]
                            recent_motion = np.sqrt((end_point[0] - start_point[0])**2 + 
                                                  (end_point[1] - start_point[1])**2)
                            
                            # Check if motion is minimal (static gesture)
                            if recent_motion < STATIC_DETECTION_THRESHOLD:
                                if not is_holding_static:
                                    # Start static gesture timer
                                    static_start_time = time.time()
                                    static_finger_states = current_right_states.copy()
                                    is_holding_static = True
                                    print(f'>>> Started static gesture detection...')
                                else:
                                    # Check if held long enough and finger states are consistent
                                    hold_duration = time.time() - static_start_time
                                    states_consistent = (current_right_states == static_finger_states)
                                    
                                    if hold_duration >= STATIC_HOLD_TIME and states_consistent:
                                        print(f'>>> Static gesture detected! Held for {hold_duration:.1f}s')
                                        state = 'PREDICT'
                                    elif not states_consistent:
                                        # Finger states changed, reset
                                        is_holding_static = False
                                        print(f'>>> Finger states changed, resetting static detection')
                            else:
                                # Motion detected, reset static tracking
                                if is_holding_static:
                                    print(f'>>> Motion detected ({recent_motion:.4f}), resetting static detection')
                                is_holding_static = False
                
                # Check if left trigger is opened (no longer in trigger position)
                if not left_is_trigger:
                    # Always go to unified prediction - let delta magnitude decide
                    state = 'PREDICT'

            elif state == 'PREDICT':
                # Unified prediction for both static and dynamic gestures
                if current_right_states is None:
                    print('[WARN] No right hand finger state -> skipped.')
                else:
                    try:
                        # Always compute motion features (allow small motions for static gestures)
                        smoothed = smooth_points(list(buffer)) if len(buffer) > 1 else [[0.5, 0.5], [0.5, 0.5]]
                        motion_features = compute_motion_features(smoothed, is_static=True)  # Always compute
                        
                        if motion_features is None:
                            print('[WARN] Could not compute motion features -> skipped.')
                        else:
                            # Determine gesture type based on detection method and motion
                            delta_mag = motion_features['delta_magnitude']
                            
                            if is_holding_static and delta_mag < STATIC_DELTA_THRESHOLD:
                                gesture_type = "STATIC"
                                prediction_finger_states = static_finger_states
                                print(f"  -> Detected as STATIC gesture (held still, delta={delta_mag:.4f})")
                            else:
                                gesture_type = "DYNAMIC" 
                                prediction_finger_states = current_right_states
                                print(f"  -> Detected as DYNAMIC gesture (motion detected, delta={delta_mag:.4f})")
                            # Prepare features for prediction (match training preprocessing)
                            X = prepare_features(current_left_states or [0, 0, 0, 0, 0],
                                               prediction_finger_states,
                                               motion_features,
                                               scaler)
                            
                            # Predict
                            prediction = model.predict(X)[0]
                            probabilities = model.predict_proba(X)[0]
                            confidence = np.max(probabilities)
                            
                            # Boost confidence for clear patterns
                            gesture_name = label_encoder.inverse_transform([prediction])[0]
                            finger_states = prediction_finger_states
                            
                            # Boost confidence based on training data patterns
                            boost_applied = False
                            original_confidence = confidence

                            # Use pattern matching boost function
                            pattern_boost = calculate_confidence_boost(gesture_name, finger_states, GESTURE_PATTERNS)
                            if pattern_boost != 0.0:
                                confidence = max(0.1, min(0.95, confidence + pattern_boost))
                                boost_applied = True
                            
                            # Get top-3 predictions
                            top_indices = np.argsort(probabilities)[::-1][:3]
                            top_3_predictions = []
                            for i, idx in enumerate(top_indices):
                                gesture = label_encoder.inverse_transform([idx])[0]
                                prob = probabilities[idx]
                                top_3_predictions.append(f"{i+1}. {gesture}: {prob:.3f}")
                            
                            # Debug: Display feature values
                            feature_debug = []
                            feature_debug.append(f"Gesture type: {gesture_type}")
                            feature_debug.append(f"Left fingers: {current_left_states}")
                            feature_debug.append(f"Right fingers: {prediction_finger_states}")
                            feature_debug.append(f"Delta magnitude: {delta_mag:.4f}")
                            feature_debug.append(f"Motion delta: ({motion_features['delta_x']:.3f}, {motion_features['delta_y']:.3f})")
                            feature_debug.append(f"Static detection: {is_holding_static}")
                            feature_debug.append(f"Pattern matches training data: {pattern_boost > 0}")
                            if boost_applied:
                                boost_type = "boosted" if pattern_boost > 0 else "penalized"
                                feature_debug.append(f"Pattern boost: {pattern_boost:+.2f}, Confidence: {original_confidence:.3f} → {confidence:.3f} ({boost_type})")
                            
                            # Get gesture name (already defined above)
                            # gesture_name = label_encoder.inverse_transform([prediction])[0]
                            
                            print(f"\n=== {gesture_type} GESTURE PREDICTION ===")
                            print(f"Top prediction: {gesture_name} (confidence: {confidence:.3f})")
                            print("Top 3 predictions:")
                            for pred in top_3_predictions:
                                print(f"  {pred}")
                            print("Debug info:")
                            for feat in feature_debug:
                                print(f"  {feat}")
                            
                            # Set display based on confidence threshold
                            if confidence >= MIN_PREDICTION_CONFIDENCE:
                                prediction_text = f'{gesture_type}: {gesture_name}'
                                confidence_text = f'Confidence: {confidence:.3f}'
                            else:
                                prediction_text = f'Low Confidence ({gesture_type})'
                                confidence_text = f'Max confidence: {confidence:.3f}'
                            
                            top_predictions_text = top_3_predictions
                            debug_features_text = feature_debug
                            prediction_start_time = time.time()
                            
                    except Exception as e:
                        print(f'[ERROR] Prediction failed: {e}')

                # Reset for next gesture
                buffer.clear()
                current_left_states = None
                current_right_states = None
                is_holding_static = False
                static_finger_states = None
                state = 'WAIT'

            # Display prediction if within display duration
            current_time = time.time()
            if prediction_text and (current_time - prediction_start_time) < DISPLAY_DURATION:
                # Main prediction
                color = (0, 255, 0) if "Low Confidence" not in prediction_text else (0, 0, 255)
                cv2.putText(frame, prediction_text, (20, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                cv2.putText(frame, confidence_text, (20, 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Top-3 predictions
                y_offset = 150
                cv2.putText(frame, "Top 3 Predictions:", (20, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                for i, pred_text in enumerate(top_predictions_text):
                    y_offset += 30
                    cv2.putText(frame, pred_text, (30, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                
                # Debug features (right side of screen)
                debug_x = 700
                y_offset = 50
                cv2.putText(frame, "Debug Features:", (debug_x, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                for feat_text in debug_features_text:
                    y_offset += 30
                    cv2.putText(frame, feat_text, (debug_x + 10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Add countdown timer
                remaining_time = DISPLAY_DURATION - (current_time - prediction_start_time)
                timer_text = f'Timer: {remaining_time:.1f}s'
                cv2.putText(frame, timer_text, (20, 350),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow('Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nExited gesture recognition.")


if __name__ == '__main__':
    main()
