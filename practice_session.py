import os
import sys
import argparse
from training_session import load_pose_templates, show_pose_instructions, run_session

def main():
    parser = argparse.ArgumentParser(description="Interactive gesture training session with target gesture.")
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
    parser.add_argument(
        "--gesture",
        type=str,
        help="Target gesture to practice (optional).",
    )
    
    args = parser.parse_args()
    
    # Get target gesture from environment variable or command line
    target_gesture = args.gesture or os.environ.get('PRACTICE_GESTURE')
    
    if target_gesture:
        # Load templates to validate gesture exists
        try:
            template_by_pose = load_pose_templates(args.dataset)
            pose_labels = sorted(template_by_pose.keys())
            
            if target_gesture not in template_by_pose:
                print(f"❌ Gesture '{target_gesture}' not found in dataset.")
                print(f"📋 Available gestures: {', '.join(pose_labels)}")
                sys.exit(1)
            
            print(f"🎯 Starting practice session for gesture: {target_gesture}")
            
            # Show instructions for the target gesture
            if not show_pose_instructions(target_gesture, template_by_pose[target_gesture]):
                print("❌ Practice session cancelled.")
                sys.exit(0)
            
            # Run focused practice session for this gesture
            run_focused_session(args.dataset, args.camera_index, target_gesture)
            
        except Exception as e:
            print(f"❌ Failed to start practice session: {e}")
            sys.exit(1)
    else:
        # Run normal interactive session
        print("🎮 Starting interactive practice session...")
        run_session(args.dataset, args.camera_index)

def run_focused_session(dataset_path: str, camera_index: int, target_gesture: str):
    """Run practice session focused on a specific gesture"""
    import time
    import cv2
    import mediapipe as mp
    import numpy as np
    from collections import deque
    from training_session import (
        AttemptStats, get_finger_states, is_fist, extract_wrist, 
        smooth_sequence, compute_motion_features, evaluate_attempt,
        is_static_pose, format_stats_line, load_pose_templates,
        BUFFER_SIZE, SMOOTHING_WINDOW, MIN_FRAMES_TO_PROCESS,
        STATIC_HOLD_SECONDS, RESULT_DISPLAY_SECONDS, STATIC_MOTION_MAX
    )
    
    # Load templates
    template_by_pose = load_pose_templates(dataset_path)
    current_template = template_by_pose[target_gesture]
    current_is_static = is_static_pose(current_template)
    
    print(f"🎯 Practicing: {target_gesture}")
    print(f"📊 Gesture type: {'Static' if current_is_static else 'Dynamic'}")
    
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
        raise RuntimeError(f"❌ Could not open camera index {camera_index}.")

    print(f"📹 Camera opened successfully on index {camera_index}")
    
    cv2.namedWindow("Gesture Practice", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gesture Practice", 1280, 960)

    motion_buffer: deque[np.ndarray] = deque(maxlen=BUFFER_SIZE)
    state = "IDLE"
    recorded_right_vec = None
    recording_start_ts = None
    status_text = ""
    status_timestamp = 0.0

    def update_status(message: str) -> None:
        nonlocal status_text, status_timestamp
        status_text = message
        status_timestamp = time.time()
        print(f"📱 Status: {message}")

    update_status("Ready to practice! Close LEFT fist to start recording.")
    
    try:
        frame_count = 0
        while True:
            success, frame = cap.read()
            if not success:
                print("❌ Camera frame not available.")
                break

            frame_count += 1
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            left_landmarks = None
            right_landmarks = None
            left_score = 0.0
            right_score = 0.0

            # Process hand landmarks
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
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get finger states
            left_vec = get_finger_states(left_landmarks, "Left") if left_landmarks else [0, 0, 0, 0, 0]
            right_vec = get_finger_states(right_landmarks, "Right") if right_landmarks else [0, 0, 0, 0, 0]

            left_is_conf = left_score > 0.6
            right_is_conf = right_score > 0.6
            left_is_fist = is_fist(left_landmarks) if left_landmarks else False

            # UI overlays
            cv2.putText(frame, f"🎯 Practicing: {target_gesture}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, format_stats_line(stats), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            
            # Show target finger pattern
            target_fingers = "Target: " + " ".join([
                f"{name}={'✓' if state else '✗'}" 
                for name, state in zip(['T','I','M','R','P'], current_template.right)
            ])
            cv2.putText(frame, target_fingers, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if current_is_static:
                cv2.putText(
                    frame,
                    f"Static gesture: Hold right hand steady >= {STATIC_HOLD_SECONDS:.1f}s",
                    (20, 820),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (150, 255, 150),
                    2,
                )

            # Show current status
            if status_text and (time.time() - status_timestamp) <= RESULT_DISPLAY_SECONDS:
                cv2.putText(frame, status_text, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 200, 255), 2)

            # Show last result
            if stats.last_result and (time.time() - stats.last_timestamp) <= RESULT_DISPLAY_SECONDS:
                color = (0, 200, 0) if stats.last_result == "CORRECT" else (0, 0, 255)
                msg = stats.last_result if not stats.last_reason else f"{stats.last_result}: {stats.last_reason}"
                cv2.putText(frame, msg, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Control instructions
            cv2.putText(frame, "Controls: Q=quit, R=reset stats, SPACE=manual attempt", (20, 900),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
            cv2.putText(frame, "Recording: Close LEFT fist to start, release to evaluate", (20, 930),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

            # State machine
            if state == "IDLE":
                cv2.putText(frame, "State: READY", (20, 870), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if left_is_conf and left_is_fist:
                    if not right_is_conf:
                        update_status("❌ Right hand not detected clearly")
                        continue
                    
                    # Start recording
                    recorded_right_vec = right_vec[:]
                    motion_buffer.clear()
                    recording_start_ts = time.time()
                    state = "RECORDING"
                    
                    if current_is_static:
                        update_status(f"📹 Recording... Hold steady >= {STATIC_HOLD_SECONDS:.1f}s")
                    else:
                        update_status("📹 Recording motion... Release left fist when done")
                    
                    stats.last_result = ""
                    stats.last_reason = ""
                    stats.last_timestamp = 0.0

            elif state == "RECORDING":
                cv2.putText(frame, "State: RECORDING", (20, 870), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
                
                # Show progress for static gestures
                if current_is_static and recording_start_ts is not None:
                    elapsed = time.time() - recording_start_ts
                    progress = min(elapsed, STATIC_HOLD_SECONDS)
                    cv2.putText(
                        frame,
                        f"Progress: {progress:.1f}/{STATIC_HOLD_SECONDS:.1f}s",
                        (20, 850),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 200),
                        2,
                    )
                
                # Record right hand motion
                if right_is_conf:
                    wrist = extract_wrist(right_landmarks)
                    if wrist is not None:
                        motion_buffer.append(wrist)
                
                # Stop recording when left fist is released
                if left_is_conf and not left_is_fist:
                    state = "PROCESSING"

            elif state == "PROCESSING":
                cv2.putText(frame, "State: EVALUATING", (20, 870), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                duration = (time.time() - recording_start_ts) if recording_start_ts is not None else 0.0
                
                if len(motion_buffer) < MIN_FRAMES_TO_PROCESS or recorded_right_vec is None:
                    stats.record(False, "")
                    update_status("❌ Not enough frames recorded")
                    state = "IDLE"
                else:
                    # Process the attempt
                    smoothed = smooth_sequence(list(motion_buffer), window=SMOOTHING_WINDOW)
                    attempt_feats = compute_motion_features(smoothed)
                    
                    if attempt_feats is None:
                        stats.record(False, "")
                        update_status("❌ Could not compute motion features")
                    else:
                        # Evaluate the attempt
                        success, reason_code = evaluate_attempt(
                            recorded_right_vec or [0, 0, 0, 0, 0],
                            attempt_feats,
                            current_template,
                            duration,
                            current_is_static,
                        )
                        
                        stats.record(success, "")
                        
                        if success:
                            update_status("✅ CORRECT! Great job!")
                        else:
                            # More detailed feedback
                            status_map = {
                                "finger_mismatch": "❌ Wrong finger positions",
                                "static_duration": f"❌ Hold for >= {STATIC_HOLD_SECONDS:.1f}s",
                                "static_motion": "❌ Too much movement for static gesture", 
                                "motion_small": "❌ Motion too small",
                                "axis_mismatch": "❌ Wrong motion direction",
                                "direction_mismatch": "❌ Wrong motion direction",
                            }
                            update_status(status_map.get(reason_code, "❌ Try again"))
                    
                    state = "IDLE"
                
                # Cleanup
                recorded_right_vec = None
                recording_start_ts = None
                motion_buffer.clear()

            # Display the frame
            cv2.imshow("Gesture Practice", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # Q or ESC
                print("🛑 Practice session ended by user")
                break
            elif key == ord("r"):
                stats.reset()
                recording_start_ts = None
                update_status("📊 Stats reset")
            elif key == ord(" "):  # Space for manual attempt
                if state == "IDLE":
                    update_status("👋 Manual attempt - show your gesture now!")
                    
            # Print periodic status
            if frame_count % 120 == 0:  # Every ~4 seconds at 30fps
                print(f"📊 Current stats: {format_stats_line(stats)}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        
        # Final summary
        total = stats.correct + stats.wrong
        print("\n" + "="*50)
        print(f"🎯 Practice Session Complete: {target_gesture}")
        print(f"✅ Correct: {stats.correct}")
        print(f"❌ Wrong: {stats.wrong}")
        print(f"📊 Total attempts: {total}")
        print(f"🎯 Accuracy: {stats.accuracy() * 100:.1f}%")
        print("="*50)

if __name__ == "__main__":
    main()