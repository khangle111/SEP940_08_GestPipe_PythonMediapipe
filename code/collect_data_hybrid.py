import os
import csv
import collections

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# === CONFIG ===
CAPTURE_CSV = 'gesture_data_09_10_2025.csv'
BUFFER_SIZE = 60
SMOOTHING_WINDOW = 3
MIN_FRAMES = 12
MIN_CONFIDENCE = 0.7

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


def ensure_capture_csv_exists():
    if os.path.isfile(CAPTURE_CSV):
        return
    columns = ['instance_id', 'pose_label'] + LEFT_COLUMNS + RIGHT_COLUMNS + MOTION_COLUMNS + FEATURE_COLUMNS
    with open(CAPTURE_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)


def next_instance_id():
    if not os.path.isfile(CAPTURE_CSV):
        return 1
    try:
        df = pd.read_csv(CAPTURE_CSV)
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


def save_capture(instance_id, pose_label, left_states, right_states, features):
    ensure_capture_csv_exists()
    row = [instance_id, pose_label]
    row += [int(v) for v in left_states]
    row += [int(v) for v in right_states]
    start, mid, end = features['start'], features['mid'], features['end']
    row += [float(start[0]), float(start[1]), float(mid[0]), float(mid[1]), float(end[0]), float(end[1])]
    row += [features['main_axis_x'], features['main_axis_y'], features['delta_x'], features['delta_y']]
    with open(CAPTURE_CSV, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(row)


def main():
    print('=== THU THAP DU LIEU (FINGER + MOTION) ===')
    pose_label = input('Nhap ten pose_label: ').strip()
    if not pose_label:
        print('[WARN] Khong co pose_label -> thoat.')
        return

    instance_counter = next_instance_id()
    print("\nHuong dan:")
    print("  - Dua ca 2 tay vao khung hinh.")
    print("  - Dieu chinh tay phai theo pose, tay trai mo.")
    print("  - Nam tay trai de bat dau, giu chuyen dong tay phai.")
    print("  - Mo tay trai de ket thuc 1 lan ghi. Bam 'q' de thoat hoan toan.\n")

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Collect Data', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Collect Data', 1280, 960)

    state = 'WAIT'
    buffer = collections.deque(maxlen=BUFFER_SIZE)
    saved_count = 0
    current_left_states = None
    current_right_states = None

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
                        save_capture(instance_counter, pose_label,
                                     current_left_states or [0, 0, 0, 0, 0],
                                     current_right_states,
                                     features)
                        print(f"[INFO] Luu mau #{instance_counter} cho pose '{pose_label}'.")
                        instance_counter += 1
                        saved_count += 1
                buffer.clear()
                current_left_states = None
                current_right_states = None
                state = 'WAIT'

            cv2.putText(frame, f'Pose: {pose_label}', (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f'Saved: {saved_count}', (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow('Collect Data', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDa thoat. Tong so lan ghi pose '{pose_label}': {saved_count}.")


if __name__ == '__main__':
    main()
