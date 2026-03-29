"""
Real-Time Product Price Predictor
Author: Ayush Anand

Usage:
  python predict.py                        # interactive mode (type description)
  python predict.py --text "Sony headphones 1 count"   # single prediction
"""

import pickle
import re
import sys
import argparse
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Load models + scaler (once at startup)
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
    else:
        print(f"  ✗ {name} not found, skipping")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("Loading BERT model (first run downloads ~90MB)...")
from sentence_transformers import SentenceTransformer
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"  ✓ BERT ready")

print(f"\n{len(models)} models loaded. Ready to predict!\n")


# ---------------------------------------------------------------------------
# Feature extraction (same as training)
# ---------------------------------------------------------------------------
def _extract_float(text, prefix):
    idx = text.find(prefix)
    if idx == -1:
        return 0.0
    snippet = text[idx + len(prefix):idx + len(prefix) + 30]
    m = re.search(r"[\d]+\.?[\d]*", snippet)
    return float(m.group()) if m else 0.0


def _extract_pack(text):
    for pat in [r"pack\s+of\s+(\d+)", r"\((\d+)\s+pack\)",
                r"(\d+)\s*(?:count|pcs|pieces|ct\b)", r"set\s+of\s+(\d+)"]:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    return 1.0


def extract_features(content):
    text = content.lower()
    value = _extract_float(text, "value:")
    unit_map = {
        "fl oz": 1.0, "oz": 1.0, "ounce": 1.0, "lb": 16.0, "pound": 16.0,
        "kg": 35.27, "gram": 0.035, "grams": 0.035, "g ": 0.035,
        "ml": 0.034, "liter": 33.8, "litre": 33.8,
        "count": 1.0, "pack": 1.0, "piece": 1.0, "pcs": 1.0,
    }
    unit_score = next((mult for unit, mult in unit_map.items() if unit in text), 0.0)
    pack_qty = _extract_pack(text)
    title_line = next((l for l in content.split("\n") if "item name:" in l.lower() or l.strip()), "")
    title_words, title_chars = len(title_line.split()), len(title_line)
    total_chars, total_words = len(content), len(content.split())
    digit_ratio = sum(c.isdigit() for c in content) / max(total_chars, 1)
    brands = ["apple", "samsung", "sony", "lg", "hp", "dell", "nike", "adidas",
              "amazon", "google", "microsoft", "cisco", "bosch", "philips"]
    brand_hit = float(any(b in text for b in brands))
    cats = {"electronic": 1, "cable": 1, "adapter": 1, "charger": 1,
            "food": 2, "sauce": 2, "coffee": 2, "tea": 2,
            "clothing": 3, "shirt": 3, "dress": 3, "shoes": 3,
            "toy": 4, "game": 4, "supplement": 5, "vitamin": 5}
    category = next((v for k, v in cats.items() if k in text), 0)
    feats = {
        "value": value, "unit_score": unit_score, "pack_qty": pack_qty,
        "title_words": title_words, "title_chars": title_chars,
        "total_chars": total_chars, "total_words": total_words,
        "digit_ratio": digit_ratio, "brand_hit": brand_hit,
        "category": float(category),
        "has_size": float(any(k in text for k in ["inch", '"', "cm", "mm", "size"])),
        "has_weight": float(any(k in text for k in ["oz", "lb", "gram", "kg"])),
        "has_volume": float(any(k in text for k in ["ml", "liter", "gallon", "fl oz"])),
        "value_x_pack": value * max(pack_qty, 1),
        "log_value": np.log1p(value),
        "log_total_words": np.log1p(total_words),
    }
    return np.array(list(feats.values()), dtype=np.float32).reshape(1, -1)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def predict_price(description):
    text_feats = scaler.transform(extract_features(description))
    bert_feats = bert_model.encode([description], convert_to_numpy=True).astype(np.float32)
    X = np.hstack([text_feats, bert_feats])

    preds = {}
    for name, model in models.items():
        try:
            raw = model.predict(X)[0]
            preds[name] = float(np.expm1(np.clip(raw, 0, 15)))
        except Exception:
            pass

    if not preds:
        return None, {}

    # Weighted ensemble (equal weights since we don't have CV scores here)
    ensemble = np.mean(list(preds.values()))
    return ensemble, preds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_interactive():
    print("=" * 50)
    print("  PRODUCT PRICE PREDICTOR — Ayush Anand")
    print("=" * 50)
    print("Type a product description below.")
    print("Format tip: 'Item Name: <name>\\nValue: <number>\\nUnit: <unit>'")
    print("Type 'quit' to exit.\n")

    while True:
        print("─" * 50)
        print("Enter product description (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line.lower() == "quit":
                print("Bye!")
                sys.exit(0)
            if line == "" and lines:
                break
            lines.append(line)

        description = "\n".join(lines)
        if not description.strip():
            continue

        price, breakdown = predict_price(description)
        if price is None:
            print("Could not predict. Check your input.")
            continue

        print(f"\n{'─'*50}")
        print("  Individual model predictions:")
        for name, p in sorted(breakdown.items(), key=lambda x: x[1]):
            print(f"    {name:25s}: ${p:.2f}")
        print(f"{'─'*50}")
        print(f"  PREDICTED PRICE (ensemble): ${price:.2f}")
        print(f"{'─'*50}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Product description text")
    args = parser.parse_args()

    if args.text:
        price, breakdown = predict_price(args.text)
        print(f"\nPredicted price: ${price:.2f}")
        for name, p in sorted(breakdown.items(), key=lambda x: x[1]):
            print(f"  {name}: ${p:.2f}")
    else:
        run_interactive()


if __name__ == "__main__":
    main()
