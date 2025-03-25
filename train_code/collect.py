import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

data_dir = "collected_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

actions = ["run","throw"]
for action in actions:
    if not os.path.exists(os.path.join(data_dir, action)):
        os.makedirs(os.path.join(data_dir, action))

cap = cv2.VideoCapture(0)

if cap.isOpened():
    zoom_level = 100
    cap.set(cv2.CAP_PROP_ZOOM, zoom_level)

def extract_keypoints(results):
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        return np.zeros(33 * 3)

action_index = 0
sequence_index = 0
frame_index = 0
keypoints_data = []
collecting = False

# Custom drawing specifications
landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=5)
connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    zoom_factor = 1.5
    new_height, new_width = int(height / zoom_factor), int(width / zoom_factor)
    start_x, start_y = (width - new_width) // 2, (height - new_height) // 2
    cropped_frame = frame[start_y:start_y + new_height, start_x:start_x + new_width]
    resized_frame = cv2.resize(cropped_frame, (width, height))

    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    black_background = np.zeros((height, width, 3), dtype=np.uint8)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            resized_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_drawing_spec,
            connection_drawing_spec=connection_drawing_spec
        )
        mp_drawing.draw_landmarks(
            black_background,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_drawing_spec,
            connection_drawing_spec=connection_drawing_spec
        )

    if collecting:
        keypoints = extract_keypoints(results)
        keypoints_data.append(keypoints)

        if frame_index % 30 == 0 and frame_index != 0:
            save_path = os.path.join(data_dir, actions[action_index], f"sequence_{sequence_index}.npy")
            np.save(save_path, np.array(keypoints_data))
            keypoints_data = []
            sequence_index += 1

            if sequence_index >= 60:
                collecting = False
                action_index = (action_index + 1) % len(actions)
                sequence_index = 0

    cv2.putText(resized_frame, f"Action: {actions[action_index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if collecting:
        cv2.putText(resized_frame, "Collecting...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Data Collection", resized_frame)
    cv2.imshow("Skeleton", black_background)

    key = cv2.waitKey(1)
    if key == ord('n'):
        if not collecting and sequence_index < 10:
            for i in range(5, 0, -1):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame, f"Starting in {i}...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(1000)
            collecting = True
            sequence_index = 0
    elif key == ord('q'):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()