"""
Product Price Predictor — Web App
"""

import pickle, re, warnings, os
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder="ui")
CORS(app)

# ---------------------------------------------------------------------------
# Load models + scaler + BERT (once at startup)
# ---------------------------------------------------------------------------
MODELS_DIR = Path("models")
MODEL_NAMES = ["xgboost", "lightgbm", "neural_network", "ridge_regression", "gradient_boosting"]

print("Loading models...")
models = {}
for name in MODEL_NAMES:
    path = MODELS_DIR / f"{name}_model.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
            print(f"  ✓ {name}")
        except Exception as e:
            # Skip models that fail to load (e.g., neural network with torch issues)
            print(f"  ⚠ {name} failed: {e}")
            if name == "neural_network":
                print("    (Neural network skipped - continuing with other models)")
            continue

print(f"Loaded {len(models)} models successfully")

# Load scaler
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded")
except Exception as e:
    print(f"✗ Scaler failed: {e}")
    raise

# Load BERT model
print("Loading BERT model...")
try:
    from sentence_transformers import SentenceTransformer
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✓ BERT ready")
except Exception as e:
    print(f"✗ BERT failed: {e}")
    raise

print(f"\nServer ready! ({len(models)} models loaded)")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _extract_float(text, prefix):
    idx = text.find(prefix)
    if idx == -1: return 0.0
    snippet = text[idx+len(prefix):idx+len(prefix)+30]
    m = re.search(r"[\d]+\.?[\d]*", snippet)
    return float(m.group()) if m else 0.0

def _extract_pack(text):
    for pat in [r"pack\s+of\s+(\d+)", r"\((\d+)\s+pack\)",
                r"(\d+)\s*(?:count|pcs|pieces|ct\b)", r"set\s+of\s+(\d+)"]:
        m = re.search(pat, text)
        if m: return float(m.group(1))
    return 1.0

def extract_features(content):
    text = content.lower()
    value = _extract_float(text, "value:")
    unit_map = {"fl oz":1.0,"oz":1.0,"ounce":1.0,"lb":16.0,"pound":16.0,
                "kg":35.27,"gram":0.035,"grams":0.035,"g ":0.035,
                "ml":0.034,"liter":33.8,"litre":33.8,
                "count":1.0,"pack":1.0,"piece":1.0,"pcs":1.0}
    unit_score = next((mult for unit, mult in unit_map.items() if unit in text), 0.0)
    pack_qty = _extract_pack(text)
    title_line = next((l for l in content.split("\n") if "item name:" in l.lower() or l.strip()), "")
    title_words, title_chars = len(title_line.split()), len(title_line)
    total_chars, total_words = len(content), len(content.split())
    digit_ratio = sum(c.isdigit() for c in content) / max(total_chars, 1)
    brands = ["apple","samsung","sony","lg","hp","dell","nike","adidas",
              "amazon","google","microsoft","cisco","bosch","philips"]
    brand_hit = float(any(b in text for b in brands))
    cats = {"electronic":1,"cable":1,"adapter":1,"charger":1,"food":2,"sauce":2,
            "coffee":2,"tea":2,"clothing":3,"shirt":3,"dress":3,"shoes":3,
            "toy":4,"game":4,"supplement":5,"vitamin":5}
    category = next((v for k, v in cats.items() if k in text), 0)
    feats = [value, unit_score, pack_qty, title_words, title_chars,
             total_chars, total_words, digit_ratio, brand_hit, float(category),
             float(any(k in text for k in ["inch",'"',"cm","mm","size"])),
             float(any(k in text for k in ["oz","lb","gram","kg"])),
             float(any(k in text for k in ["ml","liter","gallon","fl oz"])),
             value * max(pack_qty, 1), np.log1p(value), np.log1p(total_words)]
    return np.array(feats, dtype=np.float32).reshape(1, -1)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory("ui", "index.html")

@app.route("/health")
def health():
    """Health check endpoint"""
    try:
        if len(models) == 0:
            return jsonify({"status": "warning", "message": "No models loaded"}), 503
        if scaler is None:
            return jsonify({"status": "error", "message": "Scaler not loaded"}), 503
        if bert_model is None:
            return jsonify({"status": "error", "message": "BERT model not loaded"}), 503
        
        return jsonify({
            "status": "ok", 
            "models_loaded": len(models),
            "models": list(models.keys())
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 503

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    description = data.get("description", "").strip()
    if not description:
        return jsonify({"error": "No description provided"}), 400

    try:
        # Extract and validate features
        text_feats = scaler.transform(extract_features(description))
        
        # Check feature shape
        if text_feats.shape[1] != 16:
            return jsonify({"error": f"Feature extraction failed: expected 16 features, got {text_feats.shape[1]}"}), 500
        
        # Get BERT embeddings
        bert_feats = bert_model.encode([description], convert_to_numpy=True).astype(np.float32)
        X = np.hstack([text_feats, bert_feats])
        
        # Validate combined features
        if X.shape[1] != 400:
            return jsonify({"error": f"Feature mismatch: expected 400 features, got {X.shape[1]}"}), 500

        # Get predictions from each model
        preds = {}
        for name, model in models.items():
            try:
                raw = model.predict(X)[0]
                # Convert from log space to price
                price = float(np.expm1(np.clip(raw, 0, 15)))
                preds[name] = round(max(price, 0.01), 2)
            except Exception as e:
                print(f"Model {name} failed: {e}")
                pass  # Skip failed models

        if not preds:
            return jsonify({"error": "All models failed to predict"}), 500

        # ✅ FIX: Ensemble averages in PRICE space
        ensemble = round(float(np.mean(list(preds.values()))), 2)

        return jsonify({
            "ensemble": ensemble,
            "models": preds,
            "description": description[:80],
            "models_used": len(preds)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
