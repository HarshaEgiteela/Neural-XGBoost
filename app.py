from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import joblib
import numpy as np
import os
import xgboost  

app = Flask(__name__)
CORS(app)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

scaler = None
target_encoder = None
cat_encoders = None
xgb_classifier = None
extractor = None
model_loaded = False

FEATURE_ORDER = [
    "Start Year",
    "Month_Sin",
    "Month_Cos",
    "Start Day",
    "Country",
    "Region",
    "Subregion",
    "Magnitude_Clean",
    "Latitude",
    "Longitude",
    "Deaths_Log",
    "No. Injured",
    "Affected_Log",
    "Mag_Impact_Ratio",
]


def load_models():
    global scaler, target_encoder, cat_encoders, xgb_classifier, extractor, model_loaded

    try:
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"models folder not found at: {MODEL_DIR}")

        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        target_encoder = joblib.load(os.path.join(MODEL_DIR, "target_encoder.pkl"))
        cat_encoders = joblib.load(os.path.join(MODEL_DIR, "categorical_encoders.pkl"))
        xgb_classifier = joblib.load(os.path.join(MODEL_DIR, "xgb_classifier.pkl"))

        input_dim = scaler.n_features_in_
        extractor = FeatureExtractor(input_dim)
        extractor.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "neural_extractor.pth"), map_location=torch.device("cpu"))
        )
        extractor.eval()

        model_loaded = True
        print(f"✅ Models loaded successfully from: {MODEL_DIR}")
        print(f"✅ Scaler expects {input_dim} features")

    except Exception as e:
        model_loaded = False
        print(f"❌ Error loading model artifacts: {e}")


load_models()


def safe_float(val, default=0.0):
    try:
        if val is None:
            return float(default)
        if isinstance(val, str) and val.strip() == "":
            return float(default)
        x = float(val)
        if np.isnan(x) or np.isinf(x):
            return float(default)
        return x
    except Exception:
        return float(default)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if model_loaded else "error",
        "model_loaded": model_loaded,
        "model_dir": MODEL_DIR,
    }), (200 if model_loaded else 500)


@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({
            "status": "error",
            "message": "Model not loaded. Check backend logs and ensure models folder contains scaler.pkl, target_encoder.pkl, categorical_encoders.pkl, xgb_classifier.pkl, neural_extractor.pth"
        }), 500

    try:
        data = request.get_json(force=True) or {}

        start_month = safe_float(data.get("Start Month", 1), 1)
        month_sin = float(np.sin(2 * np.pi * start_month / 12))
        month_cos = float(np.cos(2 * np.pi * start_month / 12))

        affected_log = float(np.log1p(safe_float(data.get("Total Affected", 0), 0)))
        deaths_log = float(np.log1p(safe_float(data.get("Total Deaths", 0), 0)))
        mag = safe_float(data.get("Magnitude", 0), 0)

        input_dict = {
            "Start Year": safe_float(data.get("Start Year", 2024), 2024),
            "Month_Sin": month_sin,
            "Month_Cos": month_cos,
            "Start Day": safe_float(data.get("Start Day", 1), 1),
            "Country": data.get("Country", "India"),
            "Region": data.get("Region", "Asia"),
            "Subregion": data.get("Subregion", "Southern Asia"),
            "Magnitude_Clean": mag,
            "Latitude": safe_float(data.get("Latitude", 0), 0),
            "Longitude": safe_float(data.get("Longitude", 0), 0),
            "Deaths_Log": deaths_log,
            "No. Injured": safe_float(data.get("No. Injured", 0), 0),
            "Affected_Log": affected_log,
            "Mag_Impact_Ratio": mag * (affected_log + 1),
        }

        missing = [f for f in FEATURE_ORDER if f not in input_dict]
        if missing:
            return jsonify({"status": "error", "message": f"Missing features: {missing}"}), 400

        processed_features = []
        for feat in FEATURE_ORDER:
            val = input_dict[feat]

            if cat_encoders and feat in cat_encoders:
                le = cat_encoders[feat]
                try:
                    val = le.transform([str(val)])[0]
                except Exception:
                    val = 0

            processed_features.append(val)

        scaled_feat = scaler.transform(np.array(processed_features).reshape(1, -1))

        with torch.no_grad():
            deep_features = extractor(torch.FloatTensor(scaled_feat)).numpy()

        pred_idx = int(xgb_classifier.predict(deep_features)[0])
        label = target_encoder.inverse_transform([pred_idx])[0]

        probs = xgb_classifier.predict_proba(deep_features)[0]
        confidence = float(np.max(probs))

        return jsonify({
            "status": "success",
            "prediction": str(label),
            "confidence": f"{confidence * 100:.2f}%",
            "risk_analysis": "High" if confidence > 0.85 else "Moderate"
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
