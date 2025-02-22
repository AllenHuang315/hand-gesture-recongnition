import numpy as np
import cv2
import mediapipe as mp
import time
from picamera2 import Picamera2
from keras.models import load_model
import random

# Init camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640,480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load the trained LSTM model
model = load_model('lstm_model.h5')

# Prepare the mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load labels
labels_dict = {0: 'rock', 1: 'scissor', 2: 'paper'}
num_classes = len(labels_dict)

def get_ai_move():
    """Generate a random move for AI"""
    ai_move_idx = random.randint(0, num_classes - 1)
    return labels_dict[ai_move_idx]

def determine_winner(player_move, ai_move):
    """Determine the winner based on rock-paper-scissors rules"""
    if player_move == ai_move:
        return "Tie!", 0
    elif (player_move == 'rock' and ai_move == 'scissor') or \
         (player_move == 'scissor' and ai_move == 'paper') or \
         (player_move == 'paper' and ai_move == 'rock'):
        return "You Win!", 1
    else:
        return "AI Wins!", -1

# Initialize game variables
player_score = 0
ai_score = 0
count = 0
prev_time = time.time()
fps = 0

# Initialize hand detection state variables
hand_present = False
current_gesture_processed = False
current_player_move = None
current_ai_move = None
current_result_text = None
current_confidence = None

# Variables for tracking detection state
is_first_detection = True
initial_detection_time = None
INITIAL_DETECTION_DELAY = 0.5  # 500ms delay to avoid duplicate detections
hand_left_frame = True  # Track if hand has left the frame

while True:
    start_time = time.time()
    
    # Capture frame
    frame = picam2.capture_array()
    count += 1
    if count % 3 != 0:  # Process every 3rd frame to reduce load
        continue
    frame = cv2.flip(frame, -1)
    
    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Check if hand is present in frame
    if results.multi_hand_landmarks:
        # If hand was previously out of frame, this is a new detection cycle
        if hand_left_frame:
            is_first_detection = True
            initial_detection_time = None
            hand_left_frame = False
        
        # Handle first detection with delay to stabilize
        if is_first_detection:
            if initial_detection_time is None:
                initial_detection_time = time.time()
            # Ensure sufficient delay time has passed
            elif time.time() - initial_detection_time >= INITIAL_DETECTION_DELAY:
                is_first_detection = False
                hand_present = True
                current_gesture_processed = False
        else:
            hand_present = True
        
        # Process hand gesture
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []
            data_aux = []
            
            # Extract landmark coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize the landmark data
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Only predict if gesture hasn't been processed yet
            if not current_gesture_processed and hand_present and len(data_aux) == 42:
                input_data = np.array(data_aux).reshape((1, 6, 7))
                prediction = model.predict(input_data)
                predicted_index = np.argmax(prediction)
                current_player_move = labels_dict[predicted_index]
                
                # Update confidence score
                current_confidence = prediction[0][predicted_index] * 100
                
                # Get AI move and determine winner
                current_ai_move = get_ai_move()
                current_result_text, winner = determine_winner(current_player_move, current_ai_move)
                
                # Update score
                if winner == 1:
                    player_score += 1
                elif winner == -1:
                    ai_score += 1
                
                # Mark gesture as processed to prevent multiple detections
                current_gesture_processed = True
            
            # Draw landmarks and bounding box
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
            )

            # Draw bounding box around hand
            h, w, _ = frame.shape
            x_min = int(min(x_) * w)
            y_min = int(min(y_) * h)
            x_max = int(max(x_) * w)
            y_max = int(max(y_) * h)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
            # Display confidence score if available
            if current_confidence is not None:
                cv2.putText(frame, f'Confidence: {current_confidence:.1f}%', 
                           (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        # When hand leaves frame
        hand_present = False
        hand_left_frame = True  # Mark that hand has left the frame
        
        # Reset detection state only when necessary
        if current_gesture_processed:
            # Keep current results displayed, but prepare for new detection
            is_first_detection = True
            initial_detection_time = None
        
        # Only reset gesture processing state when hand has left
        if hand_left_frame:
            current_gesture_processed = False
    
    # Display current game state
    if current_player_move:
        cv2.putText(frame, f'You: {current_player_move}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
    if current_ai_move:
        cv2.putText(frame, f'AI: {current_ai_move}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Display score
    cv2.putText(frame, f'Score: Player {player_score} - {ai_score} AI', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display result
    if current_result_text:
        cv2.putText(frame, current_result_text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Display status message
    if current_gesture_processed:
        status_msg = "Move detected! Remove hand for new round"
        cv2.putText(frame, status_msg, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
    else:
        if hand_present:
            status_msg = "Processing..."
        else:
            status_msg = "Show your hand gesture"
        cv2.putText(frame, status_msg, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

    # Calculate and display FPS
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time
    cv2.putText(frame, f'FPS: {int(fps)}', (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Check for exit key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        picam2.stop()
        cv2.destroyAllWindows()
        exit()