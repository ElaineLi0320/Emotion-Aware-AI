"""
    This script is used to demonstrate how to get predictions from our deployed model.
    It captures frames from your webcam and sends them to the API for prediction at a
    specified interval.
"""

import cv2
import base64
import requests
from dotenv import load_dotenv
import os
import time
from flask import Flask, render_template, Response, jsonify

load_dotenv()

# Get API URL from .env file
API_URL = os.getenv("API_URL")

app = Flask(__name__)
last_prediction = {"emotion": "No prediction yet", "probabilities": {}}

def encode_frame(frame):
    """Convert OpenCV frame to base64 string."""
    # Convert frame to JPEG format and get a numpy array buffer
    _, buffer = cv2.imencode('.jpg', frame)

    # Encode the buffer in base64 binary format and decode it to a string
    return base64.b64encode(buffer).decode('utf-8')

def get_prediction(frame_data):
    """
    Get prediction from API.

    Args:
        frame_data (str): Base64 encoded frame data.
    Returns:
        dict: Prediction result.
    """
    payload = {
        "frame": f"data:image/jpeg;base64,{frame_data}"
    }
    try:
        response = requests.post(
            API_URL + "/predict",
            json=payload
        )
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    return None

def gen_frames():
    """Generate frames from webcam with emotion predictions."""
    cap = cv2.VideoCapture(0)
    last_prediction_time = 0
    prediction_interval = 1  # 1 second between predictions

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Get new prediction every interval
        current_time = time.time()
        if current_time - last_prediction_time >= prediction_interval:
            frame_data = encode_frame(frame)
            prediction = get_prediction(frame_data)
            if prediction:
                global last_prediction
                last_prediction = prediction
            last_prediction_time = current_time

        # Convert frame to jpg for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_current_prediction():
    """Return current prediction."""
    return jsonify(last_prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5000)