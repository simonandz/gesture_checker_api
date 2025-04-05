from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import pickle

app = Flask(__name__)

# Load model and scaler
with open('sign_rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

@app.route('/predict', methods=['POST'])
def predict():
    # Read image from request
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Process image with MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    
    if not result.multi_hand_landmarks:
        return jsonify({"error": "No hand detected"}), 400
    
    # Extract landmarks
    landmarks = []
    for lm in result.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    
    # Predict
    landmarks = scaler.transform([landmarks])
    prediction = rf_model.predict(landmarks)[0]
    
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=False, port=5000)