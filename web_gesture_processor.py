#!/usr/bin/env python3
"""
Web-integrated gesture practice processor
Nhận frames từ web qua stdin, xử lý MediaPipe, trả kết quả qua stdout
"""

import sys
import json
import base64
import cv2
import numpy as np
from training_session import GestureTrainingSession
import io
from PIL import Image

class WebGestureProcessor:
    def __init__(self, gesture_name):
        self.gesture_name = gesture_name
        self.training_session = GestureTrainingSession()
        self.session_active = False
        
        # Load gesture template
        self.gesture_template = self.training_session.get_gesture_template(gesture_name)
        if not self.gesture_template:
            self.send_error(f"Gesture '{gesture_name}' not found")
            sys.exit(1)
            
        # Initialize MediaPipe
        self.training_session.initialize_mediapipe()
        
        # Stats
        self.stats = {
            'correct': 0,
            'wrong': 0,
            'total': 0
        }
        
        self.send_ready()
    
    def send_message(self, message_type, data):
        """Send JSON message to stdout"""
        message = {
            'type': message_type,
            'timestamp': self.training_session.get_current_time(),
            **data
        }
        print(json.dumps(message), flush=True)
    
    def send_ready(self):
        self.send_message('ready', {
            'gesture': self.gesture_name,
            'template': {
                'fingers': self.gesture_template.get('fingers', []),
                'delta': self.gesture_template.get('delta', [0, 0])
            }
        })
    
    def send_error(self, error_msg):
        self.send_message('error', {'error': error_msg})
    
    def send_status(self, status, details=None):
        data = {'status': status}
        if details:
            data.update(details)
        self.send_message('status', data)
    
    def send_result(self, result):
        """Send gesture evaluation result"""
        self.stats['total'] += 1
        if result['success']:
            self.stats['correct'] += 1
        else:
            self.stats['wrong'] += 1
            
        accuracy = (self.stats['correct'] / self.stats['total'] * 100) if self.stats['total'] > 0 else 0
        
        self.send_message('result', {
            'success': result['success'],
            'message': result['message'],
            'reason': result.get('reason', ''),
            'stats': {
                **self.stats,
                'accuracy': round(accuracy, 1)
            }
        })
    
    def process_frame(self, frame_data):
        """Process single frame from web"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(frame_data.split(',')[1])  # Remove data:image/jpeg;base64,
            image = Image.open(io.BytesIO(image_data))
            
            # Convert PIL to OpenCV format
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Process with training session logic
            result = self.training_session.process_frame(frame, self.gesture_template)
            
            if result:
                if result['type'] == 'status_update':
                    self.send_status(result['status'], result.get('details'))
                elif result['type'] == 'gesture_result':
                    self.send_result(result)
                    
        except Exception as e:
            self.send_error(f"Frame processing error: {str(e)}")
    
    def run(self):
        """Main loop - read frames from stdin"""
        self.send_message('info', {'message': f'Ready to process {self.gesture_name} gestures'})
        
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    command = json.loads(line)
                    
                    if command['type'] == 'frame':
                        self.process_frame(command['data'])
                    elif command['type'] == 'reset':
                        self.training_session.reset_session()
                        self.stats = {'correct': 0, 'wrong': 0, 'total': 0}
                        self.send_message('reset', {'message': 'Session reset'})
                    elif command['type'] == 'stop':
                        break
                        
                except json.JSONDecodeError:
                    self.send_error("Invalid JSON command")
                except Exception as e:
                    self.send_error(f"Command processing error: {str(e)}")
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.send_message('shutdown', {'message': 'Session ended'})

def main():
    if len(sys.argv) < 2:
        print(json.dumps({'type': 'error', 'error': 'Usage: python web_gesture_processor.py <gesture_name>'}))
        sys.exit(1)
    
    gesture_name = sys.argv[1]
    processor = WebGestureProcessor(gesture_name)
    processor.run()

if __name__ == "__main__":
    main()