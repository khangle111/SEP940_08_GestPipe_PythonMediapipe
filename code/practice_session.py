#!/usr/bin/env python3
"""
Practice script with model selection - can choose models from different user folders
Based on training_session_ml.py but allows selecting different model sources
"""

import argparse
import os
import time
import pickle
import joblib
from collections import deque
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

# Constants
BUFFER_SIZE = 60
SMOOTHING_WINDOW = 3
MIN_FRAMES_TO_PROCESS = 12
MIN_DELTA_MAG = 0.05
RESULT_DISPLAY_SECONDS = 2.0
STATIC_HOLD_SECONDS = 1.0
INSTRUCTION_WINDOW = "Pose Instructions"
DELTA_WEIGHT = 10.0
CONFIDENCE_THRESHOLD = 0.65

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


def find_available_models():
    """Find all available model folders (general + user folders)"""
    model_sources = []
    
    # Current script directory and parent directory
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    
    # Note: General models testing should use training_session_ml.py
    # This script focuses only on user-specific models
    
    # Add user models - check both current dir and code dir
    user_search_locations = [
        script_dir,           # code/ directory (where user_Khang likely is)
        parent_dir,           # hybrid_realtime_pipeline/ directory  
        Path('.')             # current working directory
    ]
    
    for search_dir in user_search_locations:
        for user_folder in search_dir.glob('user_*'):
            models_folder = user_folder / 'models'
            if models_folder.exists() and (models_folder / 'motion_svm_model.pkl').exists():
                user_name = user_folder.name.replace('user_', '')
                # Check if this user is already added (avoid duplicates)
                if not any(source['type'] == 'user' and source['user'] == user_name for source in model_sources):
                    model_sources.append({
                        'name': f'User: {user_name}',
                        'path': str(models_folder),
                        'type': 'user',
                        'user': user_name
                    })
    
    return model_sources


def select_model_source():
    """Let user select which models to use - simple number selection"""
    import os
    
    def clear_screen():
        """Clear screen cross-platform"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    # Find available models
    model_sources = find_available_models()
    
    if not model_sources:
        print("❌ No trained models found!")
        return None
    
    clear_screen()
    print("🤖 Select Model Source:")
    print("=" * 60)
    
    for i, source in enumerate(model_sources):
        print(f"  [{i}] {source['name']}")
        print(f"      Path: {source['path']}")
        if source['type'] == 'user':
            print(f"      💡 This will use {source['user']}'s personalized models")
        else:
            print(f"      💡 This will use the general default models")
        print()
    
    print("=" * 60)
    
    while True:
        try:
            choice = input(f"Select model (0-{len(model_sources)-1}, Enter for 0, 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                print("❌ Selection cancelled\\n")
                return None
            
            if not choice:  # Default to first option
                selected = model_sources[0]
                print(f"✅ Selected: {selected['name']}\\n")
                return selected
            
            idx = int(choice)
            if 0 <= idx < len(model_sources):
                selected = model_sources[idx]
                print(f"✅ Selected: {selected['name']}\\n")
                return selected
            else:
                print(f"❌ Please enter a number between 0-{len(model_sources)-1}")
                
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\\n❌ Selection cancelled")
            return None


def load_models_from_source(model_source):
    """Load models from selected source"""
    models_dir = Path(model_source['path'])
    
    model_pkl = models_dir / 'motion_svm_model.pkl'
    scaler_pkl = models_dir / 'motion_scaler.pkl'
    static_dynamic_pkl = models_dir / 'static_dynamic_classifier.pkl'
    
    if not model_pkl.exists() or not scaler_pkl.exists():
        raise FileNotFoundError(f"Model files not found in {models_dir}")
    
    try:
        # Try loading with joblib first (new format)
        svm_model = joblib.load(model_pkl)
        scaler = joblib.load(scaler_pkl)
        
        # Load static/dynamic classifier
        static_dynamic_model = None
        if static_dynamic_pkl.exists():
            static_dynamic_model = joblib.load(static_dynamic_pkl)
        
        # For joblib models, we need to extract the label encoder from the model
        if hasattr(svm_model, 'classes_'):
            label_encoder = type('LabelEncoder', (), {
                'classes_': svm_model.classes_,
                'inverse_transform': lambda self, y: svm_model.classes_[y]
            })()
        else:
            # Fallback - create simple label encoder
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array(['end', 'home', 'next_slide', 'previous_slide', 
                                              'rotate_down', 'rotate_left', 'rotate_right', 'rotate_up', 
                                              'zoom_in', 'zoom_out'])
        
        print("✅ Models loaded successfully (joblib format)!")
        
    except:
        # Fallback to pickle format (old format)
        try:
            with open(model_pkl, 'rb') as f:
                model_data = pickle.load(f)
            
            with open(scaler_pkl, 'rb') as f:
                scaler = pickle.load(f)
            
            # Load static/dynamic classifier
            static_dynamic_model = None
            if static_dynamic_pkl.exists():
                with open(static_dynamic_pkl, 'rb') as f:
                    static_dynamic_model = pickle.load(f)
            
            svm_model = model_data['model']
            label_encoder = model_data['label_encoder']
            
            print("✅ Models loaded successfully (pickle format)!")
            
        except Exception as e:
            raise Exception(f"Failed to load models with both joblib and pickle: {e}")
    
    print(f"   - Source: {model_source['name']}")
    print(f"   - Path: {model_source['path']}")
    print(f"   - SVM Model: {len(label_encoder.classes_)} classes")
    print(f"   - Classes: {list(label_encoder.classes_)}")
    print(f"   - SVM Model Type: {type(svm_model)}")
    if static_dynamic_model:
        print(f"   - Static/Dynamic Classifier: Available")

    return svm_model, label_encoder, scaler, static_dynamic_model, model_source
def load_gesture_templates(model_source):
    """Load gesture templates based on model source"""
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    
    template_file = None
    
    if model_source['type'] == 'user':
        # Try user-specific templates in multiple locations
        user_template_locations = [
            script_dir / f"user_{model_source['user']}" / 'training_results' / 'gesture_data_compact.csv',
            parent_dir / f"user_{model_source['user']}" / 'training_results' / 'gesture_data_compact.csv',
            Path(f"user_{model_source['user']}") / 'training_results' / 'gesture_data_compact.csv'
        ]
        
        for user_templates in user_template_locations:
            if user_templates.exists():
                template_file = user_templates
                print(f"✅ Using {model_source['user']}'s personalized gesture templates")
                break
        
        if template_file is None:
            print(f"⚠️  User templates not found, using general templates")
    
    # Fallback to general templates if user templates not found or if general model
    if template_file is None:
        general_template_locations = [
            script_dir / 'training_results' / 'gesture_data_compact.csv',
            script_dir / 'general_training_results' / 'gesture_data_compact.csv',  # fallback
            parent_dir / 'training_results' / 'gesture_data_compact.csv',
            parent_dir / 'general_training_results' / 'gesture_data_compact.csv', 
            Path('training_results') / 'gesture_data_compact.csv',
            Path('general_training_results') / 'gesture_data_compact.csv'
        ]
        
        for general_templates in general_template_locations:
            if general_templates.exists():
                template_file = general_templates
                print(f"✅ Using general gesture templates")
                break
    
    if template_file is None or not template_file.exists():
        raise FileNotFoundError(f"Gesture templates not found in any expected location")
    
    import pandas as pd
    df = pd.read_csv(template_file)
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
    
    # COMBINED THUMB DECISION
    distance_open = thumb_to_palm_dist > 0.08
    extension_open = thumb_extended_x or thumb_extended_y
    angle_open = thumb_straight
    
    # Use OR logic - if any method detects open thumb
    thumb_is_open = (distance_open or extension_open or angle_open) and thumb_position_open
    states[0] = 1 if thumb_is_open else 0

    # Other fingers
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
    """Compute motion features for ML prediction - same as training_session_ml.py"""
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
    """Prepare features for SVM prediction - auto-detect format based on scaler"""
    # Use expected left states instead of actual for trigger hand
    actual_left = expected_left if (use_expected_left and expected_left) else left_states
    
    # Auto-detect feature format based on scaler's expected input size
    try:
        # Try to determine expected feature count from scaler
        if hasattr(scaler, 'n_features_in_'):
            expected_features = scaler.n_features_in_
        elif hasattr(scaler, 'scale_'):
            expected_features = len(scaler.scale_)
        else:
            # Fallback - try both formats
            expected_features = None
    except:
        expected_features = None
    
    if expected_features == 8:
        # General models format (training_session_ml.py): 
        # Scaler only for 8 motion features, then combine with unscaled finger features
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
        
        # Scale motion features only
        motion_scaled = scaler.transform(motion_array)
        
        # Combine features
        X = np.hstack([finger_feats, motion_scaled])
        return X
        
    elif expected_features == 18:
        # User models format (train_user_models.py): 
        # Scale ALL 18 features at once
        weighted_delta_x = motion_features['delta_x'] * DELTA_WEIGHT
        weighted_delta_y = motion_features['delta_y'] * DELTA_WEIGHT
        
        # Create complete 18-feature vector matching training format
        feature_vector = np.array([
            # Finger states (10 features)
            actual_left[0], actual_left[1], actual_left[2], actual_left[3], actual_left[4],
            right_states[0], right_states[1], right_states[2], right_states[3], right_states[4],
            
            # Motion base (4 features) 
            motion_features['main_axis_x'],
            motion_features['main_axis_y'],
            weighted_delta_x,
            weighted_delta_y,
            
            # Motion directions (4 features)
            motion_features['motion_left'] * DELTA_WEIGHT,
            motion_features['motion_right'] * DELTA_WEIGHT,
            motion_features['motion_up'] * DELTA_WEIGHT,
            motion_features['motion_down'] * DELTA_WEIGHT
        ], dtype=float).reshape(1, -1)
        
        # Scale ALL 18 features at once
        X_scaled = scaler.transform(feature_vector)
        return X_scaled
    
    else:
        # Fallback: assume general format (8 motion features)
        finger_feats = np.array(actual_left + right_states, dtype=float).reshape(1, -1)
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
        motion_scaled = scaler.transform(motion_array)
        X = np.hstack([finger_feats, motion_scaled])
        return X


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
    if right_states != expected['right_fingers']:
        return False, "right_fingers", f"Wrong right fingers: got {right_states}, expected {expected['right_fingers']}"
    
    print("✅ Right hand finger positions correct! (Left hand ignored as trigger)")
    
    # Step 2: Static/Dynamic classification
    is_static_expected = expected['is_static']
    
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
        
        # Try to get confidence if available
        try:
            probabilities = svm_model.predict_proba(X)[0]
            confidence = np.max(probabilities)
        except AttributeError:
            # SVM was not trained with probability=True, use decision function instead
            try:
                decision_scores = svm_model.decision_function(X)[0]
                # Normalize decision scores to [0, 1] range as pseudo-confidence
                if len(decision_scores) > 1:  # Multi-class
                    confidence = np.max(decision_scores) / (np.max(decision_scores) - np.min(decision_scores) + 1e-6)
                else:  # Binary
                    confidence = 1.0 / (1.0 + np.exp(-decision_scores))  # Sigmoid
                confidence = min(1.0, max(0.5, confidence))  # Clamp to reasonable range
            except:
                confidence = 0.8  # Default confidence if nothing works
        
        # Get predicted label - SVM returns string labels directly
        predicted_label = prediction if isinstance(prediction, str) else str(prediction)
        
        print(f"🤖 ML Prediction: {predicted_label} (confidence: {confidence:.3f})")
        
        # Check confidence threshold (lower threshold if using decision function)
        confidence_threshold = CONFIDENCE_THRESHOLD if hasattr(svm_model, 'predict_proba') else 0.5
        if confidence < confidence_threshold:
            return False, "low_confidence", f"Too uncertain: {confidence:.1%} < {confidence_threshold:.0%}"
        
        # Check prediction matches target
        if predicted_label != target_gesture:
            return False, "wrong_prediction", f"ML predicted: {predicted_label} ({confidence:.1%})"
        
        print("✅ ML validation passed!")
        return True, "ml_correct", f"Perfect! ({confidence:.1%} confidence)"
        
    except Exception as e:
        print(f"❌ ML Evaluation error: {e}")
        return False, "ml_error", f"Prediction failed: {str(e)}"


def select_gesture(available_gestures: List[str]) -> Optional[str]:
    """Let user select target gesture - simple number selection"""
    import os
    
    def clear_screen():
        os.system('cls' if os.name == 'nt' else 'clear')
    
    clear_screen()
    print("🎯 Select Target Gesture:")
    print("=" * 50)
    
    for i, gesture in enumerate(available_gestures):
        print(f"  [{i}] {gesture}")
    
    print("=" * 50)
    
    while True:
        try:
            choice = input(f"Select gesture (0-{len(available_gestures)-1}, Enter for 0, 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                print("❌ Selection cancelled\\n")
                return None
            
            if not choice:  # Default to first option
                selected = available_gestures[0]
                print(f"✅ Selected: {selected}\\n")
                return selected
            
            idx = int(choice)
            if 0 <= idx < len(available_gestures):
                selected = available_gestures[idx]
                print(f"✅ Selected: {selected}\\n")
                return selected
            else:
                print(f"❌ Please enter a number between 0-{len(available_gestures)-1}")
                
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\\n❌ Selection cancelled")
            return None


def format_stats_line(stats: AttemptStats) -> str:
    """Format statistics for display"""
    accuracy = stats.accuracy() * 100.0
    total = stats.correct + stats.wrong
    return f"Correct: {stats.correct}  Wrong: {stats.wrong}  Total: {total}  Acc: {accuracy:.1f}%"


def run_practice_session(camera_index: int = 0):
    """Run practice session with selectable models"""
    
    # Select model source
    model_source = select_model_source()
    if not model_source:
        print("Practice cancelled.")
        return
    
    # Load models and templates from selected source
    try:
        svm_model, label_encoder, scaler, static_dynamic_data, model_info = load_models_from_source(model_source)
        gesture_templates = load_gesture_templates(model_source)
        available_gestures = list(label_encoder.classes_)
        
        print(f"📊 Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
        print(f"📏 Using models from: {model_source['name']}")
        
    except Exception as e:
        print(f"❌ Failed to load models/templates: {e}")
        return
    
    # Select target gesture
    target_gesture = select_gesture(available_gestures)
    if not target_gesture:
        print("Practice cancelled.")
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
    
    cv2.namedWindow("Practice Session", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Practice Session", 1280, 960)
    
    # Session state
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
    
    print(f"\\n🚀 Practice Session Started!")
    print(f"🎯 Target Gesture: {target_gesture}")
    print(f"🤖 Using: {model_source['name']}")
    print(f"📊 Model Classes: {len(available_gestures)}")
    
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
            
            # Check confidence and fist detection
            left_confident = left_score > 0.6
            right_confident = right_score > 0.6
            left_is_fist = is_fist(left_landmarks) if left_landmarks else False
            
            # UI Elements
            gesture_type = "STATIC" if gesture_templates[target_gesture]['is_static'] else "DYNAMIC"
            cv2.putText(frame, f"🎯 Target: {target_gesture} ({gesture_type})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"🤖 Model: {model_source['name']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
            cv2.putText(frame, f"📊 {format_stats_line(stats)} | Threshold: {CONFIDENCE_THRESHOLD:.0%}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show expected finger states
            expected = gesture_templates[target_gesture]
            finger_info = f"Expected: L{expected['left_fingers']}(trigger) R{expected['right_fingers']}(checked)"
            cv2.putText(frame, finger_info, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Status and results
            if status_text and (time.time() - status_timestamp) <= RESULT_DISPLAY_SECONDS:
                cv2.putText(frame, status_text, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 255, 255), 2)
            
            if stats.last_result and (time.time() - stats.last_timestamp) <= RESULT_DISPLAY_SECONDS:
                color = (0, 255, 0) if stats.last_result == "CORRECT" else (0, 0, 255)
                result_msg = f"{stats.last_result}: {stats.last_reason}" if stats.last_reason else stats.last_result
                cv2.putText(frame, result_msg, (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Controls
            cv2.putText(frame, "🎮 n/p=change gesture, m=change model, r=reset, q=quit", (20, 900), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, "✊ Close LEFT fist to record, release to evaluate", (20, 930), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, "🎯 MODE: RIGHT hand exact + Direction + ML confidence (LEFT=trigger only)", (20, 960), 
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
                        # ML Evaluation
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
            cv2.imshow("Practice Session", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):  # Change model
                cap.release()
                cv2.destroyAllWindows()
                run_practice_session(camera_index)  # Restart with new model selection
                return
            elif key == ord('n'):  # Next gesture
                current_idx = available_gestures.index(target_gesture)
                next_idx = (current_idx + 1) % len(available_gestures)
                target_gesture = available_gestures[next_idx]
                stats.reset()
                state = "IDLE"
                motion_buffer.clear()
                update_status(f"🎯 Changed to: {target_gesture}")
            elif key == ord('p'):  # Previous gesture  
                current_idx = available_gestures.index(target_gesture)
                prev_idx = (current_idx - 1) % len(available_gestures)
                target_gesture = available_gestures[prev_idx] 
                stats.reset()
                state = "IDLE"
                motion_buffer.clear()
                update_status(f"🎯 Changed to: {target_gesture}")
            elif key == ord('r'):  # Reset stats
                stats.reset()
                update_status("📊 Statistics reset")
    
    except KeyboardInterrupt:
        print("\\n🛑 Practice interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\\n📊 Final Statistics:")
        print(f"   Target: {target_gesture}")
        print(f"   Model: {model_source['name']}")
        print(f"   {format_stats_line(stats)}")
        print("🏁 Practice Session Complete!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Practice session with model selection")
    parser.add_argument('--camera-index', type=int, default=0,
                       help='Camera index for OpenCV (default: 0)')
    args = parser.parse_args()
    
    print("🎯 Practice Session with Model Selection")
    print("=" * 50)
    
    try:
        run_practice_session(camera_index=args.camera_index)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())