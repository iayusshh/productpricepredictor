"""
Retrain gradient_boosting on the same 400-feature pipeline as the other models.
(16 text features scaled by scaler.pkl) + (384 BERT dims) = 400 total

Run:
    source venv/bin/activate
    python retrain_gradient_boosting.py
"""

import pickle
import time
import warnings
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

TRAIN_CSV = "dataset/train.csv"
SCALER_PATH = "scaler.pkl"
OUT_PATH = Path("models/gradient_boosting_model.pkl")
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Feature extraction — must match app.py exactly
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


def _parse_row(content):
    text = content.lower()
    value = _extract_float(text, "value:")
    unit_map = {"fl oz": 1.0, "oz": 1.0, "ounce": 1.0, "lb": 16.0, "pound": 16.0,
                "kg": 35.27, "gram": 0.035, "grams": 0.035, "g ": 0.035,
                "ml": 0.034, "liter": 33.8, "litre": 33.8,
                "count": 1.0, "pack": 1.0, "piece": 1.0, "pcs": 1.0}
    unit_score = next((mult for unit, mult in unit_map.items() if unit in text), 0.0)
    pack_qty = _extract_pack(text)
    title_line = next((l for l in content.split("\n") if "item name:" in l.lower() or l.strip()), "")
    title_words, title_chars = len(title_line.split()), len(title_line)
    total_chars, total_words = len(content), len(content.split())
    digit_ratio = sum(c.isdigit() for c in content) / max(total_chars, 1)
    brands = ["apple", "samsung", "sony", "lg", "hp", "dell", "nike", "adidas",
              "amazon", "google", "microsoft", "cisco", "bosch", "philips"]
    brand_hit = float(any(b in text for b in brands))
    cats = {"electronic": 1, "cable": 1, "adapter": 1, "charger": 1, "food": 2, "sauce": 2,
            "coffee": 2, "tea": 2, "clothing": 3, "shirt": 3, "dress": 3, "shoes": 3,
            "toy": 4, "game": 4, "supplement": 5, "vitamin": 5}
    category = next((v for k, v in cats.items() if k in text), 0)
    return [value, unit_score, pack_qty, title_words, title_chars,
            total_chars, total_words, digit_ratio, brand_hit, float(category),
            float(any(k in text for k in ["inch", '"', "cm", "mm", "size"])),
            float(any(k in text for k in ["oz", "lb", "gram", "kg"])),
            float(any(k in text for k in ["ml", "liter", "gallon", "fl oz"])),
            value * max(pack_qty, 1), np.log1p(value), np.log1p(total_words)]


def main():
    print("=" * 60)
    print("Retraining gradient_boosting on 400-feature pipeline")
    print("=" * 60)

    # ── Load training data ──────────────────────────────────────
    print(f"\n[1/4] Loading {TRAIN_CSV} ...")
    df = pd.read_csv(TRAIN_CSV)
    print(f"      {len(df):,} rows loaded")
    y = np.log1p(df["price"].values.astype(np.float64))

    # ── Text features (16) via existing scaler ──────────────────
    print("\n[2/4] Extracting text features ...")
    t0 = time.time()
    rows = [_parse_row(c) for c in df["catalog_content"].fillna("")]
    X_text_raw = np.array(rows, dtype=np.float32)
    print(f"      Raw text features: {X_text_raw.shape}  ({time.time()-t0:.1f}s)")

    print("      Loading scaler.pkl ...")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    X_text = scaler.transform(X_text_raw)
    print(f"      Scaled text features: {X_text.shape}")

    # ── BERT embeddings (384) ───────────────────────────────────
    print("\n[3/4] Encoding BERT embeddings (all-MiniLM-L6-v2) ...")
    print("      This takes ~15-25 min on CPU — please wait ...")
    from sentence_transformers import SentenceTransformer
    bert = SentenceTransformer("all-MiniLM-L6-v2")
    texts = df["catalog_content"].fillna("").tolist()
    t1 = time.time()
    X_bert = bert.encode(texts, batch_size=256, show_progress_bar=True,
                         convert_to_numpy=True).astype(np.float32)
    print(f"      BERT embeddings: {X_bert.shape}  ({time.time()-t1:.1f}s)")

    # ── Combine features ────────────────────────────────────────
    X = np.hstack([X_text, X_bert])
    print(f"\n      Combined feature matrix: {X.shape}")
    assert X.shape[1] == 400, f"Expected 400 features, got {X.shape[1]}"

    # ── Train GradientBoostingRegressor ─────────────────────────
    print("\n[4/4] Training GradientBoostingRegressor ...")
    print("      (n_estimators=150, max_depth=5, lr=0.1) — ~5-15 min on CPU ...")
    model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=RANDOM_SEED,
        verbose=1,
    )
    t2 = time.time()
    model.fit(X, y)
    print(f"      Training done in {time.time()-t2:.1f}s")
    print(f"      n_features_in_: {model.n_features_in_}")

    # ── Save ────────────────────────────────────────────────────
    with open(OUT_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✓ Saved to {OUT_PATH}")
    print("\nDone! gradient_boosting is now trained on 400 features.")
    print("Restart app.py — it will automatically pick up the new model.")


if __name__ == "__main__":
    main()
