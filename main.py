import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import time

# Load model
model = tf.keras.models.load_model('asl_skeleton_cnn_best.h5')
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
               'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
               'u', 'v', 'w', 'x', 'y', 'z']

# Init mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Webcam
cap = cv.VideoCapture(0)
img_size = (128, 128)

# Buffers
subtitle = ""
last_prediction = ""
cooldown_seconds = 1.5
last_added_time = time.time()

# Prediction buffer for smoothing
pred_buffer = deque(maxlen=8)  # Sliding window of last 8 predictions
conf_buffer = deque(maxlen=8)  # Buffer for confidences
buffer_threshold = 6  # Require at least 6/8 agreement

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect hands
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # Create a blank black canvas to draw landmarks
    skeleton = np.zeros((480, 480, 3), dtype=np.uint8)

    pred_class = ""
    confidence = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw the skeleton
            mp_draw.draw_landmarks(
                skeleton,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

        # Resize skeleton image to 128x128 for the model
        skeleton_resized = cv.resize(skeleton, img_size)
        input_tensor = np.expand_dims(skeleton_resized.astype('float32') / 255.0, axis=0)

        # Predict
        preds = model.predict(input_tensor, verbose=0)
        pred_class = class_names[np.argmax(preds)]
        confidence = float(np.max(preds))

        # Add prediction and confidence to buffer
        pred_buffer.append(pred_class)
        conf_buffer.append(confidence)

        # Smoothing: Only accept if majority in buffer and high confidence
        most_common = max(set(pred_buffer), key=pred_buffer.count) if pred_buffer else pred_class
        agree_count = pred_buffer.count(most_common)
        avg_conf = np.mean([c for p, c in zip(pred_buffer, conf_buffer) if p == most_common]) if pred_buffer else confidence

        # Show current letter and confidence bar
        cv.putText(frame, f"{most_common.upper()} ({avg_conf:.2f})", (30, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        bar_length = int(200 * avg_conf)
        cv.rectangle(frame, (30, 70), (30 + bar_length, 90), (0, 255, 0), -1)
        cv.rectangle(frame, (30, 70), (230, 90), (255, 255, 255), 2)

        # Add to subtitle only if majority and confidence high, cooldown passed
        current_time = time.time()
        if (agree_count >= buffer_threshold and avg_conf > 0.90 and
            (current_time - last_added_time > cooldown_seconds)):
            subtitle += most_common.upper()
            last_added_time = current_time

    # Display subtitle
    cv.putText(frame, f"Subtitle: {subtitle}", (30, 120),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    # Show last N predictions
    cv.putText(frame, f"Buffer: {' '.join([p.upper() for p in pred_buffer])}", (30, 160),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

    # Combine original frame and skeleton
    combined = np.hstack([cv.resize(frame, (640, 480)), skeleton])
    cv.imshow("ASL Skeleton Predictor", combined)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        subtitle = ""
        last_prediction = ""
    elif key == ord('s'):
        # Save subtitle to file
        with open('asl_subtitle.txt', 'w') as f:
            f.write(subtitle)
    elif key == ord('x'):
        # Copy subtitle to clipboard (Windows only)
        try:
            import subprocess
            subprocess.run('clip', universal_newlines=True, input=subtitle)
        except Exception:
            pass

cap.release()
cv.destroyAllWindows()
