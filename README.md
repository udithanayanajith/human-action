  Action Recognition System body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; color: #333; } h1, h2, h3 { color: #2c3e50; } h1 { border-bottom: 2px solid #3498db; padding-bottom: 10px; } h2 { border-bottom: 1px solid #eee; padding-bottom: 5px; } code { background-color: #f5f5f5; padding: 2px 5px; border-radius: 3px; font-family: monospace; } pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; } .highlight { background-color: #fffde7; padding: 15px; border-left: 4px solid #ffd600; margin: 20px 0; } table { border-collapse: collapse; width: 100%; margin: 20px 0; } th, td { border: 1px solid #ddd; padding: 8px; text-align: left; } th { background-color: #f2f2f2; }

Action Recognition System
=========================

A Flask-based web application that recognizes human actions from video input using LSTM neural networks and MediaPipe for pose estimation.

This system can recognize actions at three difficulty levels (Easy, Medium, Hard) with appropriate feedback about the performance quality.

Features
--------

*   Real-time action recognition from video input
*   Three difficulty levels with different action sets
*   Pose estimation using MediaPipe
*   LSTM-based action classification
*   Performance scoring and feedback system
*   REST API endpoint for predictions
*   Web interface for easy interaction

Action Classes
--------------

Difficulty Level

Actions

Model File

Easy

catch, stand, walk

easy\_action\_model.pth

Medium

run, throw

medium\_action\_model.pth

Hard

dribble, handstand, kick-ball

hard\_action\_model.pth

Installation
------------

1.  Clone the repository:
    
        git clone https://github.com/yourusername/action-recognition.git
        cd action-recognition
    
2.  Create and activate a virtual environment:
    
        python -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    
3.  Install the required dependencies:
    
        pip install -r requirements.txt
    
4.  Download the model files and place them in the root directory:
    *   easy\_action\_model.pth
    *   medium\_action\_model.pth
    *   hard\_action\_model.pth

Usage
-----

### Running the Application

    python app.py

The application will start on `http://127.0.0.1:5000`

### API Endpoint

**POST /action/predict** - Submit a video file for action recognition

Parameters:

*   `file`: Video file to analyze
*   `actual_class`: The expected action class (from the supported actions)

### Web Interface

Open `index.html` in a web browser to use the interactive interface:

1.  Select the expected action from the dropdown
2.  Click "Start Recording" to capture 5 seconds of video
3.  View the recognition results and feedback

Technical Details
-----------------

### Architecture

*   **Frontend**: HTML/JavaScript web interface
*   **Backend**: Flask web server
*   **Pose Estimation**: MediaPipe Pose
*   **Action Recognition**: Bidirectional LSTM neural network

### Models

The system uses three pre-trained models for different difficulty levels:

*   Input: 99-dimensional pose keypoints (33 landmarks × 3 coordinates)
*   Architecture: Two bidirectional LSTM layers followed by a fully connected layer
*   Hidden size: 64 units

Response Format
---------------

    {
        "predictions": [
            {
                "prediction": "run",
                "confidence": 0.95
            }
        ],
        "percentage": "95",
        "message": "Child is performing well! The action is easy for them.",
        "level": "Hard",
        "predicted_action": "run",
        "actual_class": "run",
        "receivedPredictions": {
            "run": 0.95,
            "throw": 0.05
        },
        "detection_ratio": "1.00",
        "endpoint": "medium"
    }

Requirements
------------

*   Python 3.11+
*   Flask
*   PyTorch
*   OpenCV
*   MediaPipe
*   NumPy

License
-------

MIT License
