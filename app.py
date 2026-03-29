"""
Product Price Predictor — Web App
"""

import pickle, re, warnings
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder="ui")
CORS(app)

# ---------------------------------------------------------------------------
# Load models + scaler + BERT
# ---------------------------------------------------------------------------
MODELS_DIR = Path("models")
MODEL_NAMES = ["xgboost", "lightgbm", "neural_network", "ridge_regression", "gradient_boosting"]

print("Loading models...")
models = {}
for name in MODEL_NAMES:
    path = MODELS_DIR / f"{name}_model.pkl"
    if path.exists():
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
        print(f"  ✓ {name}")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("Loading BERT model...")
from sentence_transformers import SentenceTransformer
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"  ✓ BERT ready\n")
print("Server ready at http://localhost:5050")


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

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    description = data.get("description", "").strip()
    if not description:
        return jsonify({"error": "No description provided"}), 400

    try:
        text_feats = scaler.transform(extract_features(description))
        bert_feats = bert_model.encode([description], convert_to_numpy=True).astype(np.float32)
        X = np.hstack([text_feats, bert_feats])

        preds = {}
        for name, model in models.items():
            try:
                raw = model.predict(X)[0]
                preds[name] = round(float(np.expm1(np.clip(raw, 0, 15))), 2)
            except Exception:
                pass  # skip models trained with different feature counts

        ensemble = round(float(np.mean(list(preds.values()))), 2)

        return jsonify({
            "ensemble": ensemble,
            "models": preds,
            "description": description[:80]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5050)
