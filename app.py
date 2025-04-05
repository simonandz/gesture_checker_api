import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress MediaPipe warnings
os.environ['GLOG_minloglevel'] = '2'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL

app = Flask(__name__)

# Constants
MODEL_FILE = 'sign_rf_model.pkl'
SCALER_FILE = 'scaler.pkl'

def load_model_files():
    """Load model and scaler files with error handling"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), MODEL_FILE)
        scaler_path = os.path.join(os.path.dirname(__file__), SCALER_FILE)
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Loading scaler from: {scaler_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        return model, scaler
        
    except Exception as e:
        logger.error(f"Failed to load model files: {str(e)}")
        raise

# Initialize MediaPipe Hands
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    logger.info("MediaPipe hands model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MediaPipe: {str(e)}")
    raise

# Load model and scaler at startup
try:
    rf_model, scaler = load_model_files()
except Exception as e:
    logger.critical(f"Critical startup error: {str(e)}")
    raise

@app.route('/')
def home():
    """Simple health check endpoint"""
    return jsonify({
        "status": "running",
        "message": "Send POST requests to /predict with an image file"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Validate request
        if 'image' not in request.files:
            logger.warning("No image file in request")
            return jsonify({"error": "No image file provided"}), 400

        # Read and process image
        file = request.files['image']
        img_bytes = file.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            logger.warning("Failed to decode image")
            return jsonify({"error": "Invalid image file"}), 400

        # Convert color and detect hands
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if not results.multi_hand_landmarks:
            logger.info("No hands detected in image")
            return jsonify({"error": "No hand detected"}), 400

        # Extract landmarks
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        
        # Scale and predict
        landmarks_scaled = scaler.transform([landmarks])
        prediction = rf_model.predict(landmarks_scaled)[0]
        
        logger.info(f"Successful prediction: {prediction}")
        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)