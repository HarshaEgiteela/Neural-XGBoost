from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import xgboost as xgb
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app) # Allows the React frontend to communicate with this server

# --- 1. DEFINE ARCHITECTURE (Must match your training script exactly) ---
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(x + self.block(x))

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(FeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32) 
        )
        
    def forward(self, x):
        return self.encoder(x)

# --- 2. LOAD MODELS FROM CUSTOM PATH ---
# Updated to your specific local path
MODEL_DIR = r'C:\Users\harsh\OneDrive\Desktop\Xgb\models'

try:
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"The directory {MODEL_DIR} does not exist. Please check the path.")

    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    target_encoder = joblib.load(os.path.join(MODEL_DIR, 'target_encoder.pkl'))
    cat_encoders = joblib.load(os.path.join(MODEL_DIR, 'categorical_encoders.pkl'))
    xgb_classifier = joblib.load(os.path.join(MODEL_DIR, 'xgb_classifier.pkl'))
    
    input_dim = scaler.n_features_in_
    extractor = FeatureExtractor(input_dim)
    extractor.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'neural_extractor.pth'), map_location=torch.device('cpu')))
    extractor.eval()
    print(f"N-XGB Hybrid Components Loaded Successfully from: {MODEL_DIR}")
except Exception as e:
    print(f"Error loading model artifacts: {e}")

# --- 3. PREDICTION ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Feature Engineering (consistent with training)
        start_month = float(data.get('Start Month', 1))
        month_sin = np.sin(2 * np.pi * start_month / 12)
        month_cos = np.cos(2 * np.pi * start_month / 12)
        
        affected_log = np.log1p(float(data.get('Total Affected', 0)))
        deaths_log = np.log1p(float(data.get('Total Deaths', 0)))
        mag = float(data.get('Magnitude', 0))

        input_dict = {
            'Start Year': float(data.get('Start Year', 2024)),
            'Month_Sin': month_sin,
            'Month_Cos': month_cos,
            'Start Day': float(data.get('Start Day', 1)),
            'Country': data.get('Country', 'India'),
            'Region': data.get('Region', 'Asia'),
            'Subregion': data.get('Subregion', 'Southern Asia'),
            'Magnitude_Clean': mag,
            'Latitude': float(data.get('Latitude', 0)),
            'Longitude': float(data.get('Longitude', 0)),
            'Deaths_Log': deaths_log,
            'No. Injured': float(data.get('No. Injured', 0)),
            'Affected_Log': affected_log,
            'Mag_Impact_Ratio': mag * (affected_log + 1)
        }

        # Encode Categoricals
        processed_features = []
        for feat, val in input_dict.items():
            if feat in cat_encoders:
                le = cat_encoders[feat]
                try: val = le.transform([str(val)])[0]
                except: val = 0
            processed_features.append(val)

        # Scaler -> Neural Extraction -> XGBoost Prediction
        scaled_feat = scaler.transform(np.array(processed_features).reshape(1, -1))
        with torch.no_grad():
            deep_features = extractor(torch.FloatTensor(scaled_feat)).numpy()

        pred_idx = xgb_classifier.predict(deep_features)[0]
        label = target_encoder.inverse_transform([pred_idx])[0]
        probs = xgb_classifier.predict_proba(deep_features)[0]
        confidence = float(np.max(probs))

        return jsonify({
            "status": "success",
            "prediction": label,
            "confidence": f"{confidence*100:.2f}%",
            "risk_analysis": "High" if confidence > 0.85 else "Moderate"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    # Running on 5000 is standard for the React frontend connection
    app.run(host='0.0.0.0', port=5000)