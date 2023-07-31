import cv2
import mediapipe as mp
import pyautogui
from collections import deque

# queue tha 7 last cursor positions for smoothing the movement
cursor_history = deque(maxlen=7)
screen_width, screen_height = pyautogui.size()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(0)
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
  while cap.isOpened():
    success, image = cap.read()
    #Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)
    frame_height, frame_width, _ = image.shape
    map_width, map_height = screen_width/frame_width, screen_height/frame_height
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        
        x_7 = int(hand_landmarks.landmark[7].x * frame_width)
        y_7 = int(hand_landmarks.landmark[7].y * frame_height)
        x_8 = int(hand_landmarks.landmark[8].x * frame_width)
        y_8 = int(hand_landmarks.landmark[8].y * frame_height)
        X = int(hand_landmarks.landmark[4].x * frame_width)
        Y = int(hand_landmarks.landmark[4].y * frame_height)
    
        cursor_history.append((X * map_width, Y * map_height))
        # Smooth coordinates by averaging last N positions
        X = int(sum(p[0] for p in cursor_history) / len(cursor_history))
        Y = int(sum(p[1] for p in cursor_history) / len(cursor_history))
        pyautogui.moveTo(X ,Y , duration=0)
        
        #calculate the distance between the landmark 7 and 8 for left clicks
        dist = ((x_7 - x_8)**2 + (y_7 - y_8)**2)**(0.5)//4
        if dist < 3.0:
          #do left click
          pyautogui.click(x=X, y=Y, clicks=1, interval=0, button='left')
        cv2.circle(img=image, center=(x_7, y_7), radius=8, color=(0, 255, 255), thickness=3)
        cv2.circle(img=image, center=(x_8, y_8), radius=8, color=(0, 255, 255), thickness=3)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
      
cap.release()









