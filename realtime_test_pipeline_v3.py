
import os
import time
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import deque

# ==================== CONFIG ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINGER_RULE_CSV = "gesture_motion_finger_state.csv"     # rule-based definitions
SCALER_PKL = "motion_scaler.pkl"                # from your training script
MODEL_PKL = "motion_svm_model.pkl"              # from your training script

# Camera & buffer
BUFFER_SIZE = 60
SMOOTHING_WINDOW = 3
MIN_FRAMES_TO_PROCESS = 12                      # require some frames recorded
FINGER_PRINT_INTERVAL = 5.0                     # seconds between finger-state console prints
PREDICT_DELAY_SECONDS = 1.0                     # ~1s to "predict"
PREDICT_MIN_PROB = 0.6                          # minimum probability to accept prediction

# Thresholds
MIN_CONFIDENCE = 0.6
DEFAULT_FINGER_COLS = [
    "left_finger_state_0", "left_finger_state_1", "left_finger_state_2",
    "left_finger_state_3", "left_finger_state_4",
    "right_finger_state_0", "right_finger_state_1", "right_finger_state_2",
    "right_finger_state_3", "right_finger_state_4",
]
DEFAULT_MOTION_COLS = ["main_axis_x", "main_axis_y", "delta_x", "delta_y"]
DEFAULT_DELTA_WEIGHT = 5.0
DEFAULT_MIN_DELTA_MAG = 0.05

# ==================== MediaPipe setup ====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# ==================== Utils ====================
def is_fist(hand_landmarks):
    if not hand_landmarks:
        return False
    bent = 0
    if hand_landmarks.landmark[8].y  > hand_landmarks.landmark[5].y:  bent += 1  # index
    if hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y:  bent += 1  # middle
    if hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y: bent += 1  # ring
    if hand_landmarks.landmark[20].y > hand_landmarks.landmark[17].y: bent += 1  # pinky
    return bent >= 3

def get_finger_states(hand_landmarks, handedness_label):
    """
    Return 5-bit finger states [thumb, index, middle, ring, pinky], 1=open, 0=closed.
    Thumb uses x-direction depending on palm facing; others use y relative MCP-PIP.
    """
    states = [0, 0, 0, 0, 0]
    if not hand_landmarks:
        return states

    wrist = hand_landmarks.landmark[0]
    mcp_middle = hand_landmarks.landmark[9]
    mcp_pinky = hand_landmarks.landmark[17]

    v1 = [mcp_middle.x - wrist.x, mcp_middle.y - wrist.y]
    v2 = [mcp_pinky.x - wrist.x,  mcp_pinky.y - wrist.y]
    cross_z = v1[0]*v2[1] - v1[1]*v2[0]
    palm_facing = 1 if cross_z > 0 else -1

    # Thumb
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

    # Index/Middle/Ring/Pinky (tip above PIP => open)
    states[1] = 1 if hand_landmarks.landmark[8].y  < hand_landmarks.landmark[6].y  else 0
    states[2] = 1 if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y else 0
    states[3] = 1 if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y else 0
    states[4] = 1 if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y else 0
    return states

def extract_wrist(hand_landmarks):
    if not hand_landmarks:
        return None
    w = hand_landmarks.landmark[0]
    return np.array([w.x, w.y], dtype=float)

def load_rule_csv(csv_path):
    """
    Returns: list of dict entries:
      {'pose_label': str, 'left': [0..1]*5, 'right': [0..1]*5}
    """
    rules = []
    if not os.path.isfile(csv_path):
        print(f"[WARN] Could not find finger rule CSV: {csv_path}")
        return rules
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        pose = row['pose_label']
        left = [int(row[f'left_finger_state_{i}']) for i in range(5)]
        right = [int(row[f'right_finger_state_{i}']) for i in range(5)]
        rules.append({'pose_label': pose, 'left': left, 'right': right})
    return rules

def match_rule(rules, left_vec, right_vec):
    """Returns pose_label if exact match found, else None."""
    for r in rules:
        if r['left'] == left_vec and r['right'] == right_vec:
            return r['pose_label']
    return None

def smooth_sequence(seq_xy, window=3):
    """Moving average smoothing for a list of 2D points."""
    if not seq_xy:
        return []
    arr = np.array(seq_xy, dtype=float)
    half = window // 2
    smoothed = []
    n = len(arr)
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        smoothed.append(arr[s:e].mean(axis=0))
    return smoothed

def compute_motion_features(smoothed_xy):
    """
    Input: list of 2D points (x,y), already smoothed.
    Returns: main_axis_x, main_axis_y, delta_x, delta_y using start/mid/end.
    """
    n = len(smoothed_xy)
    if n < 2:
        return None
    idx_start = 0
    idx_mid = n // 2
    idx_end = n - 1
    start = smoothed_xy[idx_start]
    end   = smoothed_xy[idx_end]
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    main_x = 1 if abs(dx) >= abs(dy) else 0
    main_y = 1 - main_x
    return [main_x, main_y, dx, dy]

def load_model_and_scaler(scaler_pkl, model_pkl):
    scaler = None
    model = None
    metadata = {}
    if os.path.isfile(scaler_pkl):
        with open(scaler_pkl, "rb") as f:
            scaler = pickle.load(f)
    else:
        print(f"[WARN] Missing scaler file: {scaler_pkl}. Running without scaling.")
    if os.path.isfile(model_pkl):
        with open(model_pkl, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict) and "model" in loaded:
            metadata = {k: v for k, v in loaded.items() if k != "model"}
            model = loaded["model"]
        else:
            model = loaded
    else:
        print(f"[ERROR] Missing model file: {model_pkl}. Cannot run motion prediction.")
    return scaler, model, metadata


# ==================== Main realtime loop ====================
def main():
    print("=== REALTIME TEST PIPELINE (Motion SVM) ===")
    print("  - Close LEFT fist (00000) to start recording.")
    print("  - Open LEFT fist to stop and run prediction.")
    print("  - Finger state snapshot prints every 5 seconds (avoid spam).")
    print("Press 'q' to quit.")
    print()

    rules = load_rule_csv(os.path.join(BASE_DIR, FINGER_RULE_CSV))
    scaler, model, model_meta = load_model_and_scaler(
        os.path.join(BASE_DIR, SCALER_PKL),
        os.path.join(BASE_DIR, MODEL_PKL),
    )

    label_encoder = model_meta.get("label_encoder") if isinstance(model_meta, dict) else None
    finger_cols = list(model_meta.get("finger_cols", DEFAULT_FINGER_COLS)) if isinstance(model_meta, dict) else list(DEFAULT_FINGER_COLS)
    motion_cols = list(model_meta.get("motion_cols", DEFAULT_MOTION_COLS)) if isinstance(model_meta, dict) else list(DEFAULT_MOTION_COLS)
    delta_weight = float(model_meta.get("delta_weight", DEFAULT_DELTA_WEIGHT)) if isinstance(model_meta, dict) else float(DEFAULT_DELTA_WEIGHT)
    min_delta_mag = float(model_meta.get("min_delta_mag", DEFAULT_MIN_DELTA_MAG)) if isinstance(model_meta, dict) else float(DEFAULT_MIN_DELTA_MAG)
    predict_min_prob = float(model_meta.get("predict_min_prob", PREDICT_MIN_PROB)) if isinstance(model_meta, dict) else float(PREDICT_MIN_PROB)

    if rules:
        print(f"[INFO] Loaded {len(rules)} finger-state rules. Trigger requires matching rule.")
    else:
        print("[INFO] No finger-state rules found; trigger relies on ML prediction only.")
    print(f"[INFO] Prediction probability threshold >= {predict_min_prob:.2f}")

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Realtime', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Realtime', 1280, 960)

    last_print_ts = 0.0
    state = "IDLE"
    motion_buffer = deque(maxlen=BUFFER_SIZE)
    recorded_left_vec = None
    recorded_right_vec = None
    rule_label = None

    def reset_state() -> None:
        nonlocal state, recorded_left_vec, recorded_right_vec, rule_label
        state = "IDLE"
        recorded_left_vec = None
        recorded_right_vec = None
        rule_label = None
        motion_buffer.clear()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[ERROR] Could not read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            left_landmarks = None
            right_landmarks = None
            left_score = 0.0
            right_score = 0.0

            if results.multi_hand_landmarks:
                for i, hl in enumerate(results.multi_hand_landmarks):
                    handed = results.multi_handedness[i].classification[0].label
                    score = results.multi_handedness[i].classification[0].score
                    if handed == "Left":
                        left_landmarks = hl
                        left_score = score
                    elif handed == "Right":
                        right_landmarks = hl
                        right_score = score
                    mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            left_vec = [0, 0, 0, 0, 0]
            right_vec = [0, 0, 0, 0, 0]
            if left_landmarks:
                left_vec = get_finger_states(left_landmarks, "Left")
            if right_landmarks:
                right_vec = get_finger_states(right_landmarks, "Right")

            now = time.time()
            if now - last_print_ts >= FINGER_PRINT_INTERVAL:
                print(f"[FINGER] Left={left_vec}  Right={right_vec}")
                if rules:
                    matched_pose = match_rule(rules, left_vec, right_vec)
                    if matched_pose is None:
                        print("  -> No finger-state rule matched.")
                    else:
                        print(f"  -> Rule matched: {matched_pose}")
                last_print_ts = now

            left_is_conf = left_score > MIN_CONFIDENCE
            right_is_conf = right_score > MIN_CONFIDENCE
            left_is_fist = is_fist(left_landmarks) if left_landmarks else False

            if state == "IDLE":
                cv2.putText(frame, "State: IDLE (Close LEFT fist to start)", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if left_is_conf and left_is_fist:
                    if not right_is_conf:
                        print("[WARN] Right hand not detected with enough confidence.")
                    else:
                        recorded_left_vec = left_vec[:]
                        recorded_right_vec = right_vec[:]
                        motion_buffer.clear()
                        if rules:
                            rule_label = match_rule(rules, recorded_left_vec, recorded_right_vec)
                            if rule_label is None:
                                print("[RULE] Right-hand finger state does not match any rule -> ignoring trigger.")
                                recorded_left_vec = None
                                recorded_right_vec = None
                            else:
                                state = "RECORDING"
                                print()
                                print(f">>> Trigger ON. Pose='{rule_label}'. Start recording right-hand motion...")
                        else:
                            rule_label = None
                            state = "RECORDING"
                            print()
                            print(">>> Trigger ON. Start recording right-hand motion...")

            elif state == "RECORDING":
                cv2.putText(frame, "State: RECORDING (release LEFT fist to stop)", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                if right_is_conf and left_is_conf:
                    wrist = extract_wrist(right_landmarks)
                    if wrist is not None:
                        motion_buffer.append(wrist)
                if left_is_conf and not left_is_fist:
                    state = "PROCESSING"
                    print(">>> Trigger OFF. Processing buffer...")

            elif state == "PROCESSING":
                cv2.putText(frame, "State: PROCESSING...", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

                if len(motion_buffer) < MIN_FRAMES_TO_PROCESS:
                    print(f"[WARN] Buffer too short ({len(motion_buffer)} frames). No motion captured.")
                    reset_state()
                else:
                    smoothed = smooth_sequence(list(motion_buffer), window=SMOOTHING_WINDOW)
                    feats = compute_motion_features(smoothed)
                    if feats is None:
                        print("[WARN] Not enough data to compute motion features.")
                        reset_state()
                    else:
                        main_x, main_y, dx, dy = feats
                        delta_mag = float(np.hypot(dx, dy))
                        print(f"[MOTION] main_axis_x={main_x}, main_axis_y={main_y}, delta_mag={delta_mag:.3f}, delta_x={dx:.3f}, delta_y={dy:.3f}")
                        if delta_mag < min_delta_mag:
                            print(f"[WARN] Motion magnitude {delta_mag:.3f} below threshold {min_delta_mag}. Skip prediction.")
                            reset_state()
                            continue

                        time.sleep(PREDICT_DELAY_SECONDS)

                        if model is None:
                            print("[ERROR] Model file not loaded -> skip prediction.")
                            reset_state()
                        else:
                            if recorded_left_vec is None or recorded_right_vec is None:
                                print("[WARN] Missing recorded finger state -> skip prediction.")
                                reset_state()
                                continue

                            finger_source_left = recorded_left_vec
                            finger_source_right = recorded_right_vec

                            motion_map = {
                                "main_axis_x": main_x,
                                "main_axis_y": main_y,
                                "delta_x": dx * delta_weight,
                                "delta_y": dy * delta_weight,
                            }
                            motion_vec = np.array([[motion_map.get(col, 0.0) for col in motion_cols]], dtype=float)
                            if scaler is not None:
                                motion_vec = scaler.transform(motion_vec)

                            feature_parts = []
                            if finger_cols:
                                finger_map = {
                                    f"left_finger_state_{i}": (finger_source_left[i] if i < len(finger_source_left) else 0.0)
                                    for i in range(5)
                                }
                                finger_map.update({
                                    f"right_finger_state_{i}": (finger_source_right[i] if i < len(finger_source_right) else 0.0)
                                    for i in range(5)
                                })
                                finger_vec = np.array([[finger_map.get(col, 0.0) for col in finger_cols]], dtype=float)
                                feature_parts.append(finger_vec)

                            feature_parts.append(motion_vec)
                            model_input = np.hstack(feature_parts) if len(feature_parts) > 1 else feature_parts[0]

                            raw_pred = model.predict(model_input)[0]
                            top_prob = None
                            if hasattr(model, "predict_proba"):
                                try:
                                    proba = model.predict_proba(model_input)[0]
                                    if isinstance(raw_pred, (int, np.integer)) and 0 <= raw_pred < len(proba):
                                        top_prob = float(proba[int(raw_pred)])
                                    else:
                                        top_prob = float(np.max(proba))
                                except Exception:
                                    top_prob = None

                            if top_prob is not None and top_prob < predict_min_prob:
                                print(f"[WARN] Confidence {top_prob:.2f} below threshold {predict_min_prob:.2f} -> skip prediction.")
                                reset_state()
                                continue

                            if label_encoder is not None:
                                pred_label = str(label_encoder.inverse_transform([raw_pred])[0])
                            else:
                                pred_label = str(raw_pred)

                            prob_msg = f" (p={top_prob:.2f})" if top_prob is not None else ""
                            if rules and rule_label is not None:
                                if pred_label == rule_label:
                                    print(f"[PREDICT] {pred_label}{prob_msg} (rule matched)")
                                else:
                                    print(f"[NOTE] Rule expects '{rule_label}', model predicted '{pred_label}{prob_msg}'.")
                            else:
                                print(f"[PREDICT] {pred_label}{prob_msg}")

                            reset_state()

            if right_score > 0:
                cv2.putText(frame, f"Right conf: {right_score:.2f}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if left_score > 0:
                cv2.putText(frame, f"Left conf:  {left_score:.2f}", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Realtime", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Exited.")


if __name__ == "__main__":
    main()
