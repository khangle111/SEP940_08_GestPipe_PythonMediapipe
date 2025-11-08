import argparse
import os
import time
import pickle
from collections import deque
from typing import Dict, List, Tuple, Optional
import cv2
import mediapipe as mp
import numpy as np

# Constants from original training_session.py
BUFFER_SIZE = 60
SMOOTHING_WINDOW = 3
MIN_FRAMES_TO_PROCESS = 12
MIN_DELTA_MAG = 0.05
RESULT_DISPLAY_SECONDS = 2.0
STATIC_HOLD_SECONDS = 1.0
INSTRUCTION_WINDOW = "Pose Instructions"

# Constants from test_gesture_recognition.py
DELTA_WEIGHT = 10.0  # Same as training
CONFIDENCE_THRESHOLD = 0.65  # 70% confidence minimum
MODELS_DIR = 'models'
MODEL_PKL = os.path.join(MODELS_DIR, 'motion_svm_model.pkl')
SCALER_PKL = os.path.join(MODELS_DIR, 'motion_scaler.pkl')
STATIC_DYNAMIC_PKL = os.path.join(MODELS_DIR, 'static_dynamic_classifier.pkl')

# Gesture templates dataset for strict validation
GESTURE_TEMPLATES_CSV = os.path.join('training_results', 'gesture_data_compact.csv')

class AttemptStats:
    def __init__(self) -> None:
        self.correct = 0
        self.wrong = 0
        self.last_result = ""
        self.last_reason = ""
        self.last_timestamp = 0.0

    def record(self, success: bool, reason: str) -> None:
        if success:
            self.correct += 1
            self.last_result = "CORRECT"
        else:
            self.wrong += 1
            self.last_result = "WRONG"
        self.last_reason = reason
        self.last_timestamp = time.time()

    def accuracy(self) -> float:
        total = self.correct + self.wrong
        return (self.correct / total) if total else 0.0

    def reset(self) -> None:
        self.correct = 0
        self.wrong = 0
        self.last_result = ""
        self.last_reason = ""
        self.last_timestamp = 0.0


def load_models():
    """Load trained SVM model, scaler, and static/dynamic classifier"""
    if not os.path.exists(MODEL_PKL) or not os.path.exists(SCALER_PKL):
        raise FileNotFoundError(f"Model files not found! Please check:\n{MODEL_PKL}\n{SCALER_PKL}")
    
    with open(MODEL_PKL, 'rb') as f:
        model_data = pickle.load(f)
    
    with open(SCALER_PKL, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load static/dynamic classifier
    static_dynamic_data = None
    if os.path.exists(STATIC_DYNAMIC_PKL):
        with open(STATIC_DYNAMIC_PKL, 'rb') as f:
            static_dynamic_data = pickle.load(f)
    
    print("✅ Models loaded successfully!")
    print(f"   - SVM Model: {len(model_data['label_encoder'].classes_)} classes")
    print(f"   - Classes: {list(model_data['label_encoder'].classes_)}")
    if static_dynamic_data:
        print(f"   - Static/Dynamic Classifier: Available")
    
    return model_data['model'], model_data['label_encoder'], scaler, static_dynamic_data


def load_gesture_templates():
    """Load gesture templates for strict validation"""
    import pandas as pd
    
    if not os.path.exists(GESTURE_TEMPLATES_CSV):
        raise FileNotFoundError(f"Gesture templates not found: {GESTURE_TEMPLATES_CSV}")
    
    df = pd.read_csv(GESTURE_TEMPLATES_CSV)
    templates = {}
    
    for _, row in df.iterrows():
        gesture = row['pose_label']
        templates[gesture] = {
            'left_fingers': [int(row[f'left_finger_state_{i}']) for i in range(5)],
            'right_fingers': [int(row[f'right_finger_state_{i}']) for i in range(5)],
            'main_axis_x': int(row['main_axis_x']),
            'main_axis_y': int(row['main_axis_y']),
            'delta_x': float(row['delta_x']),
            'delta_y': float(row['delta_y']),
            'is_static': abs(float(row['delta_x'])) < 0.02 and abs(float(row['delta_y'])) < 0.02
        }
    
    print(f"✅ Gesture templates loaded: {len(templates)} gestures")
    return templates


def get_finger_states(hand_landmarks, handedness_label: str) -> List[int]:
    """Extract finger states with improved thumb detection"""
    states = [0, 0, 0, 0, 0]
    if not hand_landmarks:
        return states

    # Get key landmarks
    wrist = hand_landmarks.landmark[0]
    thumb_tip = hand_landmarks.landmark[4]    # Thumb tip
    thumb_ip = hand_landmarks.landmark[3]     # Thumb IP joint
    thumb_mcp = hand_landmarks.landmark[2]    # Thumb MCP joint
    index_mcp = hand_landmarks.landmark[5]    # Index MCP joint
    
    mcp_middle = hand_landmarks.landmark[9]
    mcp_pinky = hand_landmarks.landmark[17]

    # Determine palm orientation
    v1 = [mcp_middle.x - wrist.x, mcp_middle.y - wrist.y]
    v2 = [mcp_pinky.x - wrist.x, mcp_pinky.y - wrist.y]
    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
    palm_facing = 1 if cross_z > 0 else -1

    # IMPROVED THUMB DETECTION
    # Method 1: Distance from thumb tip to palm center
    palm_center_x = (index_mcp.x + mcp_pinky.x) / 2
    palm_center_y = (index_mcp.y + mcp_pinky.y) / 2
    thumb_to_palm_dist = ((thumb_tip.x - palm_center_x)**2 + (thumb_tip.y - palm_center_y)**2)**0.5
    
    # Method 2: Thumb extension check (tip vs MCP joint)
    thumb_extended_x = abs(thumb_tip.x - thumb_mcp.x) > 0.04  # Extended horizontally
    thumb_extended_y = abs(thumb_tip.y - thumb_mcp.y) > 0.03  # Extended vertically
    
    # Method 3: Relative position check (original method as fallback)
    if handedness_label == "Right":
        if palm_facing > 0:
            thumb_position_open = thumb_tip.x < thumb_ip.x
        else:
            thumb_position_open = thumb_tip.x > thumb_ip.x
    else:
        if palm_facing > 0:
            thumb_position_open = thumb_tip.x < thumb_ip.x
        else:
            thumb_position_open = thumb_tip.x > thumb_ip.x
    
    # Method 4: Angle-based detection (thumb joints alignment)
    # Calculate angle between thumb MCP->IP->TIP
    import math
    def angle_between_points(p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3"""
        v1 = [p1.x - p2.x, p1.y - p2.y]
        v2 = [p3.x - p2.x, p3.y - p2.y]
        
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = (v1[0]**2 + v1[1]**2)**0.5
        mag2 = (v2[0]**2 + v2[1]**2)**0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
        return math.degrees(math.acos(cos_angle))
    
    thumb_angle = angle_between_points(thumb_mcp, thumb_ip, thumb_tip)
    thumb_straight = thumb_angle > 140  # Straight thumb (extended)
    
    # COMBINED THUMB DECISION (Multiple criteria)
    # Thumb is open if ANY of these conditions:
    # 1. Distance from palm center is large (extended away)
    # 2. Extended significantly in any direction  
    # 3. Thumb joints are nearly straight (extended)
    # 4. AND basic position check passes
    
    distance_open = thumb_to_palm_dist > 0.08
    extension_open = thumb_extended_x or thumb_extended_y
    angle_open = thumb_straight
    
    # Use OR logic - if any method detects open thumb
    thumb_is_open = (distance_open or extension_open or angle_open) and thumb_position_open
    states[0] = 1 if thumb_is_open else 0
    
    # Debug print for thumb detection
    # print(f"Thumb: dist={thumb_to_palm_dist:.3f}, ext_x={thumb_extended_x}, ext_y={thumb_extended_y}, pos={thumb_position_open}, result={states[0]}")

    # Other fingers (unchanged)
    states[1] = 1 if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y else 0
    states[2] = 1 if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y else 0
    states[3] = 1 if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y else 0
    states[4] = 1 if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y else 0
    return states


def is_fist(hand_landmarks) -> bool:
    """Check if hand is in fist position"""
    if not hand_landmarks:
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


def extract_wrist(hand_landmarks):
    """Extract wrist position"""
    if not hand_landmarks:
        return None
    wrist = hand_landmarks.landmark[0]
    return np.array([wrist.x, wrist.y], dtype=float)


def smooth_sequence(seq_xy: List[np.ndarray], window: int = 3) -> List[np.ndarray]:
    """Smooth motion sequence"""
    if not seq_xy:
        return []
    output = []
    pad = window // 2
    for idx in range(len(seq_xy)):
        start = max(0, idx - pad)
        end = min(len(seq_xy), idx + pad + 1)
        chunk = np.array(seq_xy[start:end], dtype=float)
        output.append(np.mean(chunk, axis=0))
    return output


def compute_motion_features(smoothed_xy: List[np.ndarray]) -> Optional[Dict]:
    """Compute motion features for ML prediction"""
    n = len(smoothed_xy)
    if n < 2:
        return None
    
    start = smoothed_xy[0]
    end = smoothed_xy[-1]
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    delta_mag = float(np.sqrt(dx*dx + dy*dy))
    
    # Determine main axis (matching training script logic)
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
        'raw_dx': dx,
        'raw_dy': dy,
        'delta_magnitude': delta_mag,
        'motion_left': motion_left,
        'motion_right': motion_right,
        'motion_up': motion_up,
        'motion_down': motion_down
    }


def prepare_features(left_states: List[int], right_states: List[int], motion_features: Dict, scaler, use_expected_left: bool = False, expected_left: List[int] = None) -> np.ndarray:
    """Prepare features for SVM prediction - same preprocessing as training"""
    # Use expected left states instead of actual for trigger hand
    actual_left = expected_left if (use_expected_left and expected_left) else left_states
    
    # Combine finger states
    finger_feats = np.array(actual_left + right_states, dtype=float).reshape(1, -1)
    
    # Apply delta weight and add direction features
    motion_array = np.array([[
        motion_features['main_axis_x'],
        motion_features['main_axis_y'], 
        motion_features['delta_x'] * DELTA_WEIGHT,
        motion_features['delta_y'] * DELTA_WEIGHT,
        motion_features['motion_left'] * DELTA_WEIGHT,
        motion_features['motion_right'] * DELTA_WEIGHT,
        motion_features['motion_up'] * DELTA_WEIGHT,
        motion_features['motion_down'] * DELTA_WEIGHT
    ]], dtype=float)
    
    # Scale motion features
    motion_scaled = scaler.transform(motion_array)
    
    # Combine features
    X = np.hstack([finger_feats, motion_scaled])
    return X


def prepare_static_features(left_states: List[int], right_states: List[int], delta_magnitude: float, static_scaler=None, use_expected_left: bool = False, expected_left: List[int] = None):
    """Prepare features for static/dynamic classification"""
    # Use expected left states instead of actual for trigger hand
    actual_left = expected_left if (use_expected_left and expected_left) else left_states
    
    # Combine finger states + delta magnitude (matching training)
    features = np.array([actual_left + right_states + [delta_magnitude]], dtype=float)
    
    if static_scaler:
        features = static_scaler.transform(features)
    
    return features


def evaluate_with_ml(left_states: List[int], right_states: List[int], motion_features: Dict, 
                    target_gesture: str, svm_model, label_encoder, scaler, static_dynamic_data, 
                    gesture_templates: Dict, duration: float) -> Tuple[bool, str, str]:
    """Enhanced evaluation with strict validation"""
    
    # Get expected template for target gesture
    if target_gesture not in gesture_templates:
        return False, "no_template", f"No template found for {target_gesture}"
    
    expected = gesture_templates[target_gesture]
    
    print(f"🎯 Target: {target_gesture}")
    print(f"📏 Expected fingers L:{expected['left_fingers']} R:{expected['right_fingers']}")
    print(f"📏 Recorded fingers L:{left_states} R:{right_states}")
    
    # Step 1: Finger validation (only check RIGHT hand, LEFT is trigger only)
    # LEFT hand is trigger only - don't need strict validation
    # if left_states != expected['left_fingers']:
    #     return False, "left_fingers", f"Wrong left fingers: got {left_states}, expected {expected['left_fingers']}"

    if right_states != expected['right_fingers']:
        return False, "right_fingers", f"Wrong right fingers: got {right_states}, expected {expected['right_fingers']}"
    
    print("✅ Right hand finger positions correct! (Left hand ignored as trigger)")
    
    # Step 2: Static/Dynamic classification
    is_static_expected = expected['is_static']
    
    if static_dynamic_data and 'model' in static_dynamic_data:
        try:
            static_features = prepare_static_features(
                left_states, right_states, motion_features['delta_magnitude'],
                use_expected_left=True, expected_left=expected['left_fingers']
            )
            is_static_predicted = static_dynamic_data['model'].predict(static_features)[0] == 'static'
            print(f"📊 Static/Dynamic: Expected={is_static_expected}, Predicted={is_static_predicted}")
        except:
            is_static_predicted = is_static_expected  # Fallback to template
    else:
        is_static_predicted = is_static_expected
    
    # Step 3: Static gesture validation
    if is_static_expected:
        print("🏠 Validating static gesture...")
        
        # Check duration
        if duration < STATIC_HOLD_SECONDS:
            return False, "static_duration", f"Hold longer: {duration:.1f}s < {STATIC_HOLD_SECONDS}s"
        
        # Check minimal motion
        if motion_features['delta_magnitude'] > 0.05:  # Static threshold
            return False, "static_motion", f"Too much motion: {motion_features['delta_magnitude']:.3f}"
        
        print("✅ Static gesture validated!")
        return True, "static_correct", f"Static gesture held for {duration:.1f}s"
    
    # Step 4: Dynamic gesture validation
    print("🔄 Validating dynamic gesture...")
    
    # Check if motion is sufficient
    if motion_features['delta_magnitude'] < MIN_DELTA_MAG:
        return False, "motion_small", f"Movement too small: {motion_features['delta_magnitude']:.3f}"
    
    # Step 5: Direction validation
    expected_dx = expected['delta_x']
    expected_dy = expected['delta_y']
    actual_dx = motion_features['raw_dx']
    actual_dy = motion_features['raw_dy']
    
    # Check main axis matches
    expected_main_x = expected['main_axis_x']
    actual_main_x = motion_features['main_axis_x']
    
    if expected_main_x != actual_main_x:
        axis_name = "horizontal" if expected_main_x else "vertical"
        return False, "wrong_axis", f"Wrong axis: expected {axis_name} movement"
    
    # Check direction matches
    if expected_main_x == 1:  # Horizontal movement
        if (expected_dx > 0 and actual_dx <= 0) or (expected_dx < 0 and actual_dx >= 0):
            direction = "right" if expected_dx > 0 else "left"
            return False, "wrong_direction", f"Wrong direction: expected {direction}"
    else:  # Vertical movement  
        if (expected_dy > 0 and actual_dy <= 0) or (expected_dy < 0 and actual_dy >= 0):
            direction = "down" if expected_dy > 0 else "up"
            return False, "wrong_direction", f"Wrong direction: expected {direction}"
    
    print("✅ Direction correct!")
    
    # Step 6: ML confidence validation
    try:
        X = prepare_features(
            left_states, right_states, motion_features, scaler,
            use_expected_left=True, expected_left=expected['left_fingers']
        )
        
        # Predict gesture
        prediction = svm_model.predict(X)[0]
        probabilities = svm_model.predict_proba(X)[0]
        confidence = np.max(probabilities)
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        
        print(f"🤖 ML Prediction: {predicted_label} (confidence: {confidence:.3f})")
        
        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            return False, "low_confidence", f"Too uncertain: {confidence:.1%} < {CONFIDENCE_THRESHOLD:.0%}"
        
        # Check prediction matches target
        if predicted_label != target_gesture:
            return False, "wrong_prediction", f"ML predicted: {predicted_label} ({confidence:.1%})"
        
        print("✅ ML validation passed!")
        return True, "ml_correct", f"Perfect! ({confidence:.1%} confidence)"
        
    except Exception as e:
        print(f"❌ ML Evaluation error: {e}")
        return False, "ml_error", f"Prediction failed: {str(e)}"


FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def describe_right_hand_fingers(template: Dict) -> List[Tuple[str, Tuple[int, int, int], int]]:
    """Describe right hand finger states - same as original"""
    lines: List[Tuple[str, Tuple[int, int, int], int]] = []
    for name, state in zip(FINGER_NAMES, template['right_fingers']):
        status = "Open" if state else "Closed"
        color = (80, 220, 80) if state else (70, 70, 255)  # BGR - same as original
        lines.append((f"  - {name}: {status}", color, 2))
    return lines


def describe_motion_pattern(template: Dict) -> List[Tuple[str, Tuple[int, int, int], int]]:
    """Describe motion pattern - same as original"""
    if template['is_static']:
        text = f"Static gesture: keep right hand steady for >= {STATIC_HOLD_SECONDS:.1f}s."
        return [(text, (0, 215, 255), 2)]

    dx, dy = template['delta_x'], template['delta_y']
    if template['main_axis_x'] == 1:
        if dx > 0.0:
            primary = "Primary motion: move from left to right."
        elif dx < 0.0:
            primary = "Primary motion: move from right to left."
        else:
            primary = "Primary motion: horizontal sweep."
    else:
        if dy > 0.0:
            primary = "Primary motion: move downward."
        elif dy < 0.0:
            primary = "Primary motion: move upward."
        else:
            primary = "Primary motion: vertical sweep."

    return [(primary, (0, 200, 255), 2)]


def show_gesture_instructions(gesture_name: str, gesture_template: Dict) -> bool:
    """Show detailed instructions like original training_session.py"""
    base_lines: List[Tuple[str, Tuple[int, int, int], int]] = [
        (f"Pose: {gesture_name}", (255, 255, 255), 2),
        ("", (255, 255, 255), 1),
        ("Left hand:", (255, 255, 255), 2),
        ("  - Close left fist to start recording", (255, 255, 255), 1),
        ("  - Release to finish attempt", (255, 255, 255), 1),
        ("", (255, 255, 255), 1),
        ("Right hand fingers:", (255, 255, 255), 2),
    ]

    # Add right hand finger descriptions
    base_lines.extend(describe_right_hand_fingers(gesture_template))
    base_lines.append(("", (255, 255, 255), 1))
    
    # Add motion descriptions  
    base_lines.extend(describe_motion_pattern(gesture_template))
    base_lines.append(("", (255, 255, 255), 1))
    base_lines.append(("Press Enter to begin, or Esc to cancel.", (200, 200, 200), 1))

    # Create window - same dimensions as original
    width = 900
    line_height = 36
    top_margin = 60
    height = top_margin + line_height * len(base_lines) + 40
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)  # Same background as original

    y = top_margin
    for text, color, thickness in base_lines:
        cv2.putText(
            canvas,
            text,
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,  # Same font size as original
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        y += line_height

    cv2.namedWindow(INSTRUCTION_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(INSTRUCTION_WINDOW, width, height)
    cv2.imshow(INSTRUCTION_WINDOW, canvas)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10, 32):  # Enter or Space - same as original
            cv2.destroyWindow(INSTRUCTION_WINDOW)
            return True
        if key in (27, ord("q"), ord("Q")):  # Esc or Q - same as original
            cv2.destroyWindow(INSTRUCTION_WINDOW)
            return False


def select_gesture(available_gestures: List[str]) -> Optional[str]:
    """Let user select target gesture using arrow keys + Enter (Windows compatible)"""
    import sys
    import os
    
    def get_key_windows():
        """Get keypress for Windows"""
        import msvcrt
        key = msvcrt.getch()
        if key == b'\xe0':  # Special key prefix
            key = msvcrt.getch()
            if key == b'H':  # Up arrow
                return 'UP'
            elif key == b'P':  # Down arrow
                return 'DOWN'
        elif key == b'\r':  # Enter
            return 'ENTER'
        elif key == b'\x1b':  # ESC
            return 'ESC'
        elif key == b'q':
            return 'QUIT'
        return None
    
    def get_key_unix():
        """Get keypress for Unix/Linux"""
        import termios
        import tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
            if key == '\x1b':  # ESC sequence
                key += sys.stdin.read(2)
                if key == '\x1b[A':
                    return 'UP'
                elif key == '\x1b[B':
                    return 'DOWN'
                else:
                    return 'ESC'
            elif key == '\r' or key == '\n':
                return 'ENTER'
            elif key == 'q':
                return 'QUIT'
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def clear_screen():
        """Clear screen cross-platform"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_menu(selected_idx):
        """Print the gesture selection menu"""
        clear_screen()
        
        print("🎯 Select Target Gesture:")
        print("Use ↑/↓ arrow keys to navigate, Enter to select, ESC/q to quit")
        print("=" * 50)
        
        for i, gesture in enumerate(available_gestures):
            if i == selected_idx:
                print(f"  → {gesture} ←")  # Selected item
            else:
                print(f"    {gesture}")
        
        print("=" * 50)
        print(f"Gesture {selected_idx + 1}/{len(available_gestures)}")
    
    # Choose appropriate key handler based on OS
    get_key = get_key_windows if os.name == 'nt' else get_key_unix
    
    try:
        selected_idx = 0
        
        while True:
            print_menu(selected_idx)
            
            try:
                key = get_key()
                
                if key == 'UP':
                    selected_idx = (selected_idx - 1) % len(available_gestures)
                elif key == 'DOWN':
                    selected_idx = (selected_idx + 1) % len(available_gestures)
                elif key == 'ENTER':
                    clear_screen()
                    print(f"✅ Selected: {available_gestures[selected_idx]}\n")
                    return available_gestures[selected_idx]
                elif key == 'ESC' or key == 'QUIT':
                    clear_screen()
                    print("❌ Selection cancelled\n")
                    return None
                    
            except KeyboardInterrupt:
                clear_screen()
                print("❌ Selection cancelled\n")
                return None
                
    except Exception as e:
        # Fallback for any system issues
        clear_screen()
        print(f"⚠️  Arrow key navigation not available ({str(e)}), using number selection...")
        print("\n🎯 Available Gestures:")
        for i, gesture in enumerate(available_gestures):
            print(f"  [{i}] {gesture}")
        
        while True:
            try:
                choice = input(f"\nSelect gesture (0-{len(available_gestures)-1}, Enter for 0): ").strip()
                if not choice:
                    return available_gestures[0]
                
                idx = int(choice)
                if 0 <= idx < len(available_gestures):
                    return available_gestures[idx]
                else:
                    print(f"❌ Please enter 0-{len(available_gestures)-1}")
                    
            except ValueError:
                print("❌ Please enter a number")
            except KeyboardInterrupt:
                return None


def format_stats_line(stats: AttemptStats) -> str:
    """Format statistics for display"""
    accuracy = stats.accuracy() * 100.0
    total = stats.correct + stats.wrong
    return f"Correct: {stats.correct}  Wrong: {stats.wrong}  Total: {total}  Acc: {accuracy:.1f}%"


def run_ml_training_session(camera_index: int = 0):
    """Run training session with ML models"""
    
    # Load ML models and templates
    try:
        svm_model, label_encoder, scaler, static_dynamic_data = load_models()
        gesture_templates = load_gesture_templates()
        available_gestures = list(label_encoder.classes_)
        
        print(f"📊 Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
        print(f"📏 Strict validation: Fingers + Direction + ML confidence")
        
    except Exception as e:
        print(f"❌ Failed to load models/templates: {e}")
        return
    
    # Select target gesture
    target_gesture = select_gesture(available_gestures)
    if not target_gesture:
        print("Training cancelled.")
        return
    
    # Show detailed instructions with gesture template
    target_template = gesture_templates[target_gesture]
    if not show_gesture_instructions(target_gesture, target_template):
        print("Training cancelled.")
        return
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")
    
    cv2.namedWindow("ML Training Session", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ML Training Session", 1280, 960)
    
    # Training session state
    stats = AttemptStats()
    motion_buffer: deque = deque(maxlen=BUFFER_SIZE)
    state = "IDLE"
    recorded_left_states: Optional[List[int]] = None
    recorded_right_states: Optional[List[int]] = None
    recording_start_time: Optional[float] = None
    status_text = ""
    status_timestamp = 0.0
    
    def update_status(message: str) -> None:
        nonlocal status_text, status_timestamp
        status_text = message
        status_timestamp = time.time()
    
    print(f"\n🚀 ML Training Session Started!")
    print(f"🎯 Target Gesture: {target_gesture}")
    print(f"📊 Model Classes: {len(available_gestures)}")
    print(f"💡 Press 'd' during session to toggle thumb debug mode")
    
    # Debug mode for thumb detection
    debug_thumb = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera frame not available")
                break
            
            frame = cv2.flip(frame, 1)  # Mirror effect
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            # Extract hand landmarks
            left_landmarks = None
            right_landmarks = None
            left_score = 0.0
            right_score = 0.0
            
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[i].classification[0].label
                    score = results.multi_handedness[i].classification[0].score
                    
                    if handedness == "Left":
                        left_landmarks = hand_landmarks
                        left_score = score
                    else:
                        right_landmarks = hand_landmarks
                        right_score = score
                    
                    # Draw landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get finger states
            left_states = get_finger_states(left_landmarks, "Left") if left_landmarks else [0, 0, 0, 0, 0]
            right_states = get_finger_states(right_landmarks, "Right") if right_landmarks else [0, 0, 0, 0, 0]
            
            # Debug thumb detection if enabled
            if debug_thumb and right_landmarks:
                debug_y = 200
                cv2.putText(frame, f"🔍 THUMB DEBUG MODE", (20, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
                cv2.putText(frame, f"Result: {right_states[0]} ({'OPEN' if right_states[0] else 'CLOSED'})", (20, debug_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if right_states[0] else (0, 0, 255), 2)
                cv2.putText(frame, f"All fingers: {right_states}", (20, debug_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Check confidence and fist detection
            left_confident = left_score > 0.6
            right_confident = right_score > 0.6
            left_is_fist = is_fist(left_landmarks) if left_landmarks else False
            right_is_fist = is_fist(right_landmarks) if right_landmarks else False
            
            # UI Elements
            gesture_type = "STATIC" if gesture_templates[target_gesture]['is_static'] else "DYNAMIC"
            cv2.putText(frame, f"🎯 Target: {target_gesture} ({gesture_type})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"📊 {format_stats_line(stats)} | Threshold: {CONFIDENCE_THRESHOLD:.0%}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show expected finger states
            expected = gesture_templates[target_gesture]
            finger_info = f"Expected: L{expected['left_fingers']}(trigger) R{expected['right_fingers']}(checked)"
            cv2.putText(frame, finger_info, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Status and results
            if status_text and (time.time() - status_timestamp) <= RESULT_DISPLAY_SECONDS:
                cv2.putText(frame, status_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 255, 255), 2)
            
            if stats.last_result and (time.time() - stats.last_timestamp) <= RESULT_DISPLAY_SECONDS:
                color = (0, 255, 0) if stats.last_result == "CORRECT" else (0, 0, 255)
                result_msg = f"{stats.last_result}: {stats.last_reason}" if stats.last_reason else stats.last_result
                cv2.putText(frame, result_msg, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Controls
            cv2.putText(frame, "🎮 n/p=change gesture, r=reset, q=quit", (20, 900), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, "✊ Close LEFT fist to record, release to evaluate", (20, 930), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, "🎯 MODE: RIGHT hand exact + Direction + 70% confidence (LEFT=trigger only)", (20, 960), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
            
            # State machine
            if state == "IDLE":
                cv2.putText(frame, "State: IDLE", (20, 870), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if left_confident and left_is_fist:
                    if not right_confident:
                        update_status("❌ Right hand not visible clearly")
                        continue
                    
                    # Start recording
                    recorded_left_states = left_states[:]
                    recorded_right_states = right_states[:]
                    motion_buffer.clear()
                    recording_start_time = time.time()
                    state = "RECORDING"
                    update_status("🔴 Recording gesture...")
                    
                    # Clear previous results
                    stats.last_result = ""
                    stats.last_reason = ""
                    stats.last_timestamp = 0.0
            
            elif state == "RECORDING":
                cv2.putText(frame, "State: RECORDING 🔴", (20, 870), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Record wrist motion
                if right_confident:
                    wrist_pos = extract_wrist(right_landmarks)
                    if wrist_pos is not None:
                        motion_buffer.append(wrist_pos)
                
                # Check for end of recording (release fist)
                if left_confident and not left_is_fist:
                    state = "PROCESSING"
            
            elif state == "PROCESSING":
                cv2.putText(frame, "State: PROCESSING ⚙️", (20, 870), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                duration = (time.time() - recording_start_time) if recording_start_time else 0.0
                
                # Validate recording
                if (len(motion_buffer) < MIN_FRAMES_TO_PROCESS or 
                    recorded_left_states is None or recorded_right_states is None):
                    
                    stats.record(False, "insufficient_data")
                    update_status("❌ Insufficient recording data")
                    state = "IDLE"
                    continue
                
                # Process motion
                try:
                    smoothed = smooth_sequence(list(motion_buffer), window=SMOOTHING_WINDOW)
                    motion_features = compute_motion_features(smoothed)
                    
                    if motion_features is None:
                        stats.record(False, "motion_processing_failed")
                        update_status("❌ Failed to process motion")
                    else:
                        # Enhanced ML Evaluation with strict validation
                        success, reason_code, reason_msg = evaluate_with_ml(
                            recorded_left_states, recorded_right_states, 
                            motion_features, target_gesture, 
                            svm_model, label_encoder, scaler, static_dynamic_data,
                            gesture_templates, duration
                        )
                        
                        stats.record(success, reason_msg)
                        if success:
                            update_status(f"✅ {reason_msg}")
                        else:
                            update_status(f"❌ {reason_msg}")
                            
                except Exception as e:
                    stats.record(False, f"evaluation_error: {str(e)}")
                    update_status(f"❌ Evaluation error: {str(e)}")
                
                # Reset for next attempt
                state = "IDLE"
                recorded_left_states = None
                recorded_right_states = None
                recording_start_time = None
                motion_buffer.clear()
            
            # Display frame
            cv2.imshow("ML Training Session", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):  # Toggle thumb debug
                debug_thumb = not debug_thumb
                update_status(f"👍 Thumb debug: {'ON' if debug_thumb else 'OFF'}")
            elif key == ord('n'):  # Next gesture
                current_idx = available_gestures.index(target_gesture)
                next_idx = (current_idx + 1) % len(available_gestures)
                target_gesture = available_gestures[next_idx]
                stats.reset()
                state = "IDLE"
                motion_buffer.clear()
                update_status(f"🎯 Changed to: {target_gesture}")
                target_template = gesture_templates[target_gesture]
                if not show_gesture_instructions(target_gesture, target_template):
                    break
            elif key == ord('p'):  # Previous gesture  
                current_idx = available_gestures.index(target_gesture)
                prev_idx = (current_idx - 1) % len(available_gestures)
                target_gesture = available_gestures[prev_idx] 
                stats.reset()
                state = "IDLE"
                motion_buffer.clear()
                update_status(f"🎯 Changed to: {target_gesture}")
                target_template = gesture_templates[target_gesture]
                if not show_gesture_instructions(target_gesture, target_template):
                    break
            elif key == ord('r'):  # Reset stats
                stats.reset()
                update_status("📊 Statistics reset")
    
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n📊 Final Statistics:")
        print(f"   Target: {target_gesture}")
        print(f"   {format_stats_line(stats)}")
        print("🏁 ML Training Session Complete!")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ML-based gesture training session")
    parser.add_argument('--camera-index', type=int, default=0,
                       help='Camera index for OpenCV (default: 0)')
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    print("🤖 ML-Based Gesture Training Session")
    print("=" * 40)
    
    try:
        run_ml_training_session(camera_index=args.camera_index)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())