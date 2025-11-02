import argparse
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

BUFFER_SIZE = 60
SMOOTHING_WINDOW = 3
MIN_FRAMES_TO_PROCESS = 12
MIN_DELTA_MAG = 0.05
RESULT_DISPLAY_SECONDS = 2.0
MOTION_SIMILARITY_THRESHOLD = 0.75
STATIC_TEMPLATE_DELTA_THRESHOLD = 0.02
STATIC_MOTION_MAX = 0.02
STATIC_HOLD_SECONDS = 1.0

FINGER_LEFT_COLS = [f"left_finger_state_{i}" for i in range(5)]
FINGER_RIGHT_COLS = [f"right_finger_state_{i}" for i in range(5)]


@dataclass
class PoseTemplate:
    pose_label: str
    left: Tuple[int, ...]
    right: Tuple[int, ...]
    main_axis: Tuple[int, int]
    delta: np.ndarray
    delta_norm: np.ndarray
    delta_mag: float


@dataclass
class AttemptFeatures:
    main_axis: Tuple[int, int]
    delta: np.ndarray
    delta_norm: np.ndarray
    delta_mag: float


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


def is_fist(hand_landmarks) -> bool:
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


def get_finger_states(hand_landmarks, handedness_label: str) -> List[int]:
    states = [0, 0, 0, 0, 0]
    if not hand_landmarks:
        return states

    wrist = hand_landmarks.landmark[0]
    mcp_middle = hand_landmarks.landmark[9]
    mcp_pinky = hand_landmarks.landmark[17]

    v1 = [mcp_middle.x - wrist.x, mcp_middle.y - wrist.y]
    v2 = [mcp_pinky.x - wrist.x, mcp_pinky.y - wrist.y]
    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
    palm_facing = 1 if cross_z > 0 else -1

    if handedness_label == "Right":
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


def extract_wrist(hand_landmarks):
    if not hand_landmarks:
        return None
    wrist = hand_landmarks.landmark[0]
    return np.array([wrist.x, wrist.y], dtype=float)


def smooth_sequence(seq_xy: Sequence[np.ndarray], window: int = 3) -> List[np.ndarray]:
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


def compute_motion_features(smoothed_xy: Sequence[np.ndarray]) -> AttemptFeatures | None:
    n = len(smoothed_xy)
    if n < 2:
        return None
    start = smoothed_xy[0]
    end = smoothed_xy[-1]
    delta = np.array([float(end[0] - start[0]), float(end[1] - start[1])], dtype=float)
    delta_mag = float(np.linalg.norm(delta))
    if delta_mag > 1e-8:
        delta_norm = delta / delta_mag
    else:
        delta_norm = np.zeros(2, dtype=float)
    main_x = 1 if abs(delta[0]) >= abs(delta[1]) else 0
    main_axis = (main_x, 1 - main_x)
    return AttemptFeatures(main_axis=main_axis, delta=delta, delta_norm=delta_norm, delta_mag=delta_mag)


def load_pose_templates(csv_path: str) -> Dict[str, PoseTemplate]:
    df = pd.read_csv(csv_path)
    required_cols = {"pose_label", "main_axis_x", "main_axis_y", "delta_x", "delta_y"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    templates: Dict[str, PoseTemplate] = {}
    for _, row in df.iterrows():
        pose = str(row["pose_label"])
        left = tuple(int(row[col]) for col in FINGER_LEFT_COLS)
        right = tuple(int(row[col]) for col in FINGER_RIGHT_COLS)
        main_axis = (int(row["main_axis_x"]), int(row["main_axis_y"]))
        delta = np.array([float(row["delta_x"]), float(row["delta_y"])], dtype=float)
        delta_mag = float(np.linalg.norm(delta))
        if delta_mag > 1e-8:
            delta_norm = delta / delta_mag
        else:
            delta_norm = np.zeros(2, dtype=float)
        template = PoseTemplate(
            pose_label=pose,
            left=left,
            right=right,
            main_axis=main_axis,
            delta=delta,
            delta_norm=delta_norm,
            delta_mag=delta_mag,
        )
        if pose not in templates:
            templates[pose] = template
    return templates


def is_static_pose(template: PoseTemplate) -> bool:
    return template.delta_mag <= STATIC_TEMPLATE_DELTA_THRESHOLD


def ask_pose_selection(labels: Sequence[str]) -> int:
    print("Available pose labels:")
    for idx, label in enumerate(labels):
        print(f"  [{idx}] {label}")
    while True:
        raw = input("Select pose index (Enter for 0): ").strip()
        if not raw:
            return 0
        if raw.isdigit():
            value = int(raw)
            if 0 <= value < len(labels):
                return value
        print("Invalid selection, please try again.")


def evaluate_attempt(
    right_vec: Sequence[int],
    attempt: AttemptFeatures,
    template: PoseTemplate,
    duration: float,
    is_static: bool,
) -> Tuple[bool, str]:
    if tuple(right_vec) != template.right:
        return False, "finger_mismatch"

    if is_static:
        if duration < STATIC_HOLD_SECONDS:
            return False, "static_duration"
        if attempt.delta_mag > STATIC_MOTION_MAX:
            return False, "static_motion"
        return True, "ok"

    if attempt.delta_mag < MIN_DELTA_MAG:
        return False, "motion_small"

    if template.delta_mag <= 1e-6:
        return True, "ok"

    if attempt.main_axis != template.main_axis:
        return False, "axis_mismatch"

    cos_sim = float(np.dot(attempt.delta_norm, template.delta_norm))
    if cos_sim >= MOTION_SIMILARITY_THRESHOLD:
        return True, "ok"

    return False, "direction_mismatch"


def format_stats_line(stats: AttemptStats) -> str:
    accuracy = stats.accuracy() * 100.0
    total = stats.correct + stats.wrong
    return f"Correct: {stats.correct}  Wrong: {stats.wrong}  Total: {total}  Acc: {accuracy:.1f}%"


def run_session(
    dataset_path: str,
    camera_index: int,
) -> None:
    template_by_pose = load_pose_templates(dataset_path)
    if not template_by_pose:
        raise ValueError("No pose templates loaded.")
    pose_labels = sorted(template_by_pose.keys())
    current_idx = ask_pose_selection(pose_labels)

    stats = AttemptStats()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    cv2.namedWindow("Training Session", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Training Session", 1280, 960)

    motion_buffer: deque[np.ndarray] = deque(maxlen=BUFFER_SIZE)
    state = "IDLE"
    recorded_right_vec: List[int] | None = None
    recording_start_ts: float | None = None
    status_text = ""
    status_timestamp = 0.0

    def update_status(message: str) -> None:
        nonlocal status_text, status_timestamp
        status_text = message
        status_timestamp = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[ERROR] Camera frame not available.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            left_landmarks = None
            right_landmarks = None
            left_score = 0.0
            right_score = 0.0

            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handed = results.multi_handedness[i].classification[0].label
                    score = results.multi_handedness[i].classification[0].score
                    if handed == "Left":
                        left_landmarks = hand_landmarks
                        left_score = score
                    else:
                        right_landmarks = hand_landmarks
                        right_score = score
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            left_vec = get_finger_states(left_landmarks, "Left") if left_landmarks else [0, 0, 0, 0, 0]
            right_vec = get_finger_states(right_landmarks, "Right") if right_landmarks else [0, 0, 0, 0, 0]

            left_is_conf = left_score > 0.6
            right_is_conf = right_score > 0.6
            left_is_fist = is_fist(left_landmarks) if left_landmarks else False

            overlay_pose = pose_labels[current_idx]
            current_template = template_by_pose[overlay_pose]
            current_is_static = is_static_pose(current_template)
            cv2.putText(frame, f"Pose: {overlay_pose}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, format_stats_line(stats), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            if current_is_static:
                cv2.putText(
                    frame,
                    f"Pose tinh: giu tay phai >= {STATIC_HOLD_SECONDS:.1f}s",
                    (20, 820),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (150, 255, 150),
                    2,
                )

            if status_text and (time.time() - status_timestamp) <= RESULT_DISPLAY_SECONDS:
                cv2.putText(frame, status_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 200, 255), 2)

            if stats.last_result and (time.time() - stats.last_timestamp) <= RESULT_DISPLAY_SECONDS:
                color = (0, 200, 0) if stats.last_result == "CORRECT" else (0, 0, 255)
                msg = stats.last_result if not stats.last_reason else f"{stats.last_result}: {stats.last_reason}"
                cv2.putText(frame, msg, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.putText(frame, "Commands: n/p change pose, r reset stats, q quit", (20, 900),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            cv2.putText(frame, "Start attempt: close LEFT fist, release to evaluate", (20, 930),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

            if state == "IDLE":
                cv2.putText(frame, "State: IDLE", (20, 870), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if left_is_conf and left_is_fist:
                    if not right_is_conf:
                        update_status("Khong thay tay phai ro net")
                        continue
                    recorded_right_vec = right_vec[:]
                    motion_buffer.clear()
                    recording_start_ts = time.time()
                    state = "RECORDING"
                    if current_is_static:
                        update_status(f"Giu tay phai >= {STATIC_HOLD_SECONDS:.1f}s")
                    else:
                        update_status("Dang ghi chuyen dong")
                    stats.last_result = ""
                    stats.last_reason = ""
                    stats.last_timestamp = 0.0

            elif state == "RECORDING":
                cv2.putText(frame, "State: RECORDING", (20, 870), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                if current_is_static and recording_start_ts is not None:
                    elapsed = time.time() - recording_start_ts
                    progress = min(elapsed, STATIC_HOLD_SECONDS)
                    cv2.putText(
                        frame,
                        f"Giu yen: {progress:.1f}/{STATIC_HOLD_SECONDS:.1f}s",
                        (20, 850),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 200),
                        2,
                    )
                if right_is_conf:
                    wrist = extract_wrist(right_landmarks)
                    if wrist is not None:
                        motion_buffer.append(wrist)
                if left_is_conf and not left_is_fist:
                    state = "PROCESSING"

            elif state == "PROCESSING":
                cv2.putText(frame, "State: PROCESSING", (20, 870), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                duration = (time.time() - recording_start_ts) if recording_start_ts is not None else 0.0
                if len(motion_buffer) < MIN_FRAMES_TO_PROCESS or recorded_right_vec is None:
                    stats.record(False, "")
                    update_status("Chua du khung hinh")
                    state = "IDLE"
                else:
                    smoothed = smooth_sequence(list(motion_buffer), window=SMOOTHING_WINDOW)
                    attempt_feats = compute_motion_features(smoothed)
                    if attempt_feats is None:
                        stats.record(False, "")
                        update_status("Khong doc duoc chuyen dong")
                    else:
                        success, reason_code = evaluate_attempt(
                            recorded_right_vec or [0, 0, 0, 0, 0],
                            attempt_feats,
                            current_template,
                            duration,
                            current_is_static,
                        )
                        stats.record(success, "")
                        if success:
                            update_status("")
                        else:
                            status_map = {
                                "finger_mismatch": "Sai 5 ngon tay phai",
                                "static_duration": f"Giu tay >= {STATIC_HOLD_SECONDS:.1f}s",
                                "static_motion": "Tay phai dang di chuyen",
                                "motion_small": "Chuyen dong qua nho",
                                "axis_mismatch": "Sai truc chinh",
                                "direction_mismatch": "Sai huong chuyen dong",
                            }
                            update_status(status_map.get(reason_code, "Thu lai"))
                    state = "IDLE"
                recorded_right_vec = None
                recording_start_ts = None
                motion_buffer.clear()

            cv2.imshow("Training Session", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):
                current_idx = (current_idx + 1) % len(pose_labels)
                stats.reset()
                state = "IDLE"
                recorded_right_vec = None
                recording_start_ts = None
                motion_buffer.clear()
                update_status("")
            elif key == ord("p"):
                current_idx = (current_idx - 1) % len(pose_labels)
                stats.reset()
                state = "IDLE"
                recorded_right_vec = None
                recording_start_ts = None
                motion_buffer.clear()
                update_status("")
            elif key == ord("r"):
                stats.reset()
                recording_start_ts = None
                update_status("")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        total = stats.correct + stats.wrong
        print("\n=== Training Session Summary ===")
        print(f"Pose dang luyen: {pose_labels[current_idx]}")
        print(f"Lan dung: {stats.correct} | Lan sai: {stats.wrong} | Tong: {total}")
        print(f"Ti le chinh xac: {stats.accuracy() * 100:.1f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive gesture training session.")
    parser.add_argument(
        "--dataset",
        default="training_results/gesture_data_compact.csv",
        help="Canonical pose dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index for OpenCV (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_session(dataset_path=args.dataset, camera_index=args.camera_index)


if __name__ == "__main__":
    main()
