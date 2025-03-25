import os
import numpy as np

def is_valid_skeleton(sequence, threshold=10):
    valid_frames = []
    for frame in sequence:
        non_zero_count = np.count_nonzero(frame[:, :2])
        if non_zero_count >= (33 - threshold) * 2:
            valid_frames.append(frame)
    return np.array(valid_frames) if len(valid_frames) > 0 else None

def pad_sequences(keypoints_data, max_length):
    padded_data = []
    for sequence in keypoints_data:
        if len(sequence) < max_length:
            padded_sequence = np.pad(sequence, ((0, max_length - len(sequence)), (0, 0)), mode='constant')
        else:
            padded_sequence = sequence[:max_length]
        padded_data.append(padded_sequence)
    return np.array(padded_data)

def preprocess_dataset(dataset_dir, output_file):
    labels = []
    keypoints_data = []
    max_length = 0

    for action in os.listdir(dataset_dir):
        action_dir = os.path.join(dataset_dir, action)
        if not os.path.isdir(action_dir):
            continue
        
        for sequence_file in os.listdir(action_dir):
            if sequence_file.endswith(".npy"):
                sequence_path = os.path.join(action_dir, sequence_file)
                keypoints = np.load(sequence_path)
                keypoints = keypoints.reshape(-1, 33, 3) if keypoints.shape[1] == 99 else keypoints
                filtered_keypoints = is_valid_skeleton(keypoints)
                if filtered_keypoints is not None:
                    filtered_keypoints = filtered_keypoints.reshape(filtered_keypoints.shape[0], -1)
                    keypoints_data.append(filtered_keypoints)
                    labels.append(action)
                    max_length = max(max_length, len(filtered_keypoints))

    keypoints_data = pad_sequences(keypoints_data, max_length)
    np.savez(output_file, keypoints=keypoints_data, labels=np.array(labels))

preprocess_dataset("collected_data", "preprocessed_medium_data.npz")