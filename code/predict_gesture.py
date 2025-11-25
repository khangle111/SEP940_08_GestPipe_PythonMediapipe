import sys
import json
import os
import pickle
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
MODEL_PKL = MODELS_DIR / "motion_svm_model.pkl"
SCALER_PKL = MODELS_DIR / "motion_scaler.pkl"

def predict_gesture(features):
    """
    Predict gesture using the trained SVM model
    """
    try:
        # Load model and scaler
        with open(MODEL_PKL, 'rb') as f:
            model_data = pickle.load(f)
        model = model_data['model']
        label_encoder = model_data['label_encoder']
        
        with open(SCALER_PKL, 'rb') as f:
            scaler = pickle.load(f)
        
        # Prepare features (same as training)
        left_fingers = features.get('left_fingers', [])
        right_fingers = features.get('right_fingers', [])
        motion_features = features.get('motion_features', {})
        
        # Finger states
        finger_states = left_fingers + right_fingers
        
        # Motion features
        main_axis_x = motion_features.get('main_axis_x', 0)
        main_axis_y = motion_features.get('main_axis_y', 0)
        delta_x = motion_features.get('delta_x', 0) * 15.0  # DELTA_WEIGHT
        delta_y = motion_features.get('delta_y', 0) * 15.0
        
        # Direction features
        motion_left = 15.0 if delta_x < 0 else 0
        motion_right = 15.0 if delta_x > 0 else 0
        motion_up = 15.0 if delta_y < 0 else 0
        motion_down = 15.0 if delta_y > 0 else 0
        
        # Combine features
        features_array = finger_states + [main_axis_x, main_axis_y, delta_x, delta_y, motion_left, motion_right, motion_up, motion_down]
        
        # Scale
        features_scaled = scaler.transform([features_array])
        
        # Predict
        pred_encoded = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0][pred_encoded]
        predicted_label = label_encoder.inverse_transform([pred_encoded])[0]
        
        return {
            'success': True,
            'predicted': predicted_label,
            'confidence': float(confidence)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            features = json.loads(sys.argv[1])
            result = predict_gesture(features)
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}))
    else:
        print(json.dumps({"success": False, "error": "No features provided"}))