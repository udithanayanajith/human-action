from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import cv2
import mediapipe as mp
import os

app = Flask(__name__)
CORS(app)

easy_classes = 3
medium_classes = 2
hard_classes = 3

class ActionRecognitionModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ActionRecognitionModel, self).__init__()
        self.lstm1 = torch.nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(hidden_size * 2, hidden_size // 2, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h_lstm1, _ = self.lstm1(x)
        h_lstm2, _ = self.lstm2(h_lstm1)
        h_lstm2 = h_lstm2[:, -1, :]
        out = self.fc(h_lstm2)
        return out

easy_model = ActionRecognitionModel(input_size=99, hidden_size=64, num_classes=easy_classes)
easy_model.load_state_dict(torch.load("models/easy_action_model.pth"))
easy_model.eval()

medium_model = ActionRecognitionModel(input_size=99, hidden_size=64, num_classes=medium_classes)
medium_model.load_state_dict(torch.load("models/medium_action_model.pth"))
medium_model.eval()

hard_model = ActionRecognitionModel(input_size=99, hidden_size=64, num_classes=hard_classes)
hard_model.load_state_dict(torch.load("models/hard_action_model.pth"))
hard_model.eval()

mp_pose = mp.solutions.pose

def extract_keypoints(video_path):
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    frames_with_keypoints = 0
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            frames_with_keypoints += 1
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            keypoints_list.append(keypoints)
    
    cap.release()
    return np.array(keypoints_list), frames_with_keypoints, total_frames

def get_message(score, detection_ratio):
    if detection_ratio < 0.1:
        return "No human detected in most frames. Please ensure you're in front of the camera."
    if score == 0:
        return "Action not recognized. Please try again."
    elif 1 <= score <= 10:
        return "It is very hard to find the action."
    elif 11 <= score <= 40:
        return "Child is finding it hard to perform the action. Please repeat the action as shown in the video and try again."
    elif 41 <= score <= 79:
        return "Child is performing at a medium level. Keep practicing to improve."
    else:
        return "Child is performing well! The action is easy for them."

def get_diff_level(score, detection_ratio):
    if detection_ratio < 0.1:
        return "No detection"
    if score == 0:
        return "No level"
    elif 1 <= score <= 40:
        return "Easy"
    elif 41 <= score <= 79:
        return "Medium"
    else:
        return "Hard"



@app.route("/action/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    actual_class = request.form.get("actual_class", "").strip()
    
    if not actual_class:
        return jsonify({"error": "No actual class provided"}), 400
    
    video_path = "temp_video.mp4"
    file.save(video_path)
    
    keypoints, frames_with_keypoints, total_frames = extract_keypoints(video_path)
    detection_ratio = frames_with_keypoints / total_frames if total_frames > 0 else 0
    
    if detection_ratio < 0.1:
        return jsonify({
            "predictions": [],
            "percentage": "0",
            "message": get_message(0, detection_ratio),
            "level": get_diff_level(0, detection_ratio),
            "predicted_action": "none",
            "actual_class": actual_class,
            "receivedPredictions": {},
            "detection_ratio": f"{detection_ratio:.2f}"
        })
    
    if len(keypoints) == 0:
        return jsonify({"error": "No keypoints detected"}), 400
    
    keypoints = torch.tensor(np.expand_dims(keypoints, axis=0), dtype=torch.float32)
    
    easy_actions = {"catch": 0, "stand": 1, "walk": 2}
    medium_actions = {"run": 0, "throw": 1}
    hard_actions = {"dribble": 0, "handstand": 1, "kick-ball": 2}
    
    if actual_class in easy_actions:
        model = easy_model
        label_to_int = easy_actions
        endpoint = "easy"
    elif actual_class in medium_actions:
        model = medium_model
        label_to_int = medium_actions
        endpoint = "medium"
    elif actual_class in hard_actions:
        model = hard_model
        label_to_int = hard_actions
        endpoint = "hard"
    else:
        return jsonify({"error": "Invalid action class"}), 400
    
    with torch.no_grad():
        prediction = model(keypoints)
        probabilities = torch.softmax(prediction, dim=1)
        predicted_label = torch.argmax(prediction, dim=1).item()
    
    int_to_label = {v: k for k, v in label_to_int.items()}
    predicted_action = int_to_label[predicted_label]
    
    if predicted_action != actual_class:
        score = 0  
    else:
        score = probabilities[0][predicted_label].item() * 100
    
    message = get_message(int(round(score)), detection_ratio)
    level = get_diff_level(int(round(score)), detection_ratio)
    class_probabilities = {int_to_label[i]: float(probabilities[0][i].item()) for i in range(probabilities.size(1))}
    
    os.remove(video_path)
    return jsonify({
        "predictions": [{"prediction": predicted_action, "confidence": class_probabilities.get(predicted_action, 0)}],
        "percentage": f"{score:.0f}",
        "message": message,
        "level": level,
        "predicted_action": predicted_action,
        "actual_class": actual_class,
        "receivedPredictions": class_probabilities,
        "detection_ratio": f"{detection_ratio:.2f}",
        "endpoint": endpoint
    })

if __name__ == "__main__":
    app.run(debug=True)