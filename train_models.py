"""
Product Price Predictor — Standalone Training Script

What this script does:
  1. Loads train.csv / test.csv
  2. Extracts text features from catalog_content (BERT embeddings + statistics)
  3. Optionally extracts image features (set USE_IMAGES=True; needs GPU for speed)
  4. Trains 7 ML models and selects the best ensemble
  5. Writes predictions to test_out.csv

Usage:
  python train_models.py                         # text only (fast, ~10-30 min)
  python train_models.py --images                # text + images (slow, needs GPU)
  python train_models.py --sample 5000           # quick test on 5000 rows
  python train_models.py --images --sample 1000  # image test on small sample
"""

import argparse
import logging
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TRAIN_CSV = "dataset/train.csv"
TEST_CSV = "dataset/test.csv"
OUTPUT_CSV = "test_out.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("PricePredictorTraining")

np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# SMAPE metric
# ---------------------------------------------------------------------------
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error — competition metric."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.clip(np.array(y_pred, dtype=float), 0.01, None)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-8
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


# ---------------------------------------------------------------------------
# Text Feature Extraction
# ---------------------------------------------------------------------------
def extract_text_features(df: pd.DataFrame, fit_scaler=None) -> tuple:
    """
    Extract numerical features from catalog_content.

    Returns (feature_matrix, scaler)  — scaler is fitted on training data and
    reused on test data to avoid data leakage.
    """
    log.info("Extracting text features from catalog_content ...")

    rows = []
    for content in df["catalog_content"].fillna(""):
        feat = _parse_catalog(content)
        rows.append(feat)

    feat_df = pd.DataFrame(rows)
    X = feat_df.values.astype(np.float32)

    # Scale features
    if fit_scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = fit_scaler
        X = scaler.transform(X)

    log.info(f"Text features shape: {X.shape}")
    return X, scaler


def _parse_catalog(content: str) -> dict:
    """Parse a single catalog_content string into a feature dict."""
    text = content.lower()

    # Numeric value from 'Value: X' field
    value = _extract_float(text, "value:")

    # Unit multipliers — used to infer pack size / quantity
    unit_map = {
        "fl oz": 1.0, "oz": 1.0, "ounce": 1.0,
        "lb": 16.0, "pound": 16.0,
        "kg": 35.27, "kilogram": 35.27,
        "gram": 0.035, "grams": 0.035, "g ": 0.035,
        "ml": 0.034, "milliliter": 0.034,
        "liter": 33.8, "litre": 33.8,
        "count": 1.0, "pack": 1.0, "piece": 1.0, "pcs": 1.0,
    }
    unit_score = 0.0
    for unit, mult in unit_map.items():
        if unit in text:
            unit_score += mult
            break

    # Pack-of quantity — "pack of N" or "(pack of N)"
    pack_qty = _extract_pack(text)

    # Title length and word count as proxies for product complexity
    title_line = ""
    for line in content.split("\n"):
        if "item name:" in line.lower() or line.strip():
            title_line = line
            break
    title_words = len(title_line.split())
    title_chars = len(title_line)

    # Total text length
    total_chars = len(content)
    total_words = len(content.split())

    # Digit ratio — prices often correlate with numeric-heavy descriptions
    digits = sum(c.isdigit() for c in content)
    digit_ratio = digits / max(total_chars, 1)

    # Common brand/category signals
    brands = [
        "apple", "samsung", "sony", "lg", "hp", "dell", "nike", "adidas",
        "amazon", "google", "microsoft", "cisco", "bosch", "philips",
    ]
    brand_hit = float(any(b in text for b in brands))

    categories = {
        "electronic": 1, "cable": 1, "adapter": 1, "charger": 1,
        "food": 2, "sauce": 2, "coffee": 2, "tea": 2, "snack": 2,
        "clothing": 3, "shirt": 3, "dress": 3, "shoes": 3,
        "toy": 4, "game": 4, "puzzle": 4,
        "supplement": 5, "vitamin": 5, "protein": 5,
    }
    category = 0
    for kw, cat_id in categories.items():
        if kw in text:
            category = cat_id
            break

    # Size / dimension mentions
    has_size = float(any(kw in text for kw in ["inch", '"', "cm", "mm", "size"]))
    has_weight = float(any(kw in text for kw in ["oz", "lb", "gram", "kg"]))
    has_volume = float(any(kw in text for kw in ["ml", "liter", "gallon", "fl oz"]))

    return {
        "value": value,
        "unit_score": unit_score,
        "pack_qty": pack_qty,
        "title_words": title_words,
        "title_chars": title_chars,
        "total_chars": total_chars,
        "total_words": total_words,
        "digit_ratio": digit_ratio,
        "brand_hit": brand_hit,
        "category": float(category),
        "has_size": has_size,
        "has_weight": has_weight,
        "has_volume": has_volume,
        "value_x_pack": value * max(pack_qty, 1),
        "log_value": np.log1p(value),
        "log_total_words": np.log1p(total_words),
    }


def _extract_float(text: str, prefix: str) -> float:
    """Extract first float found after a prefix keyword."""
    import re
    idx = text.find(prefix)
    if idx == -1:
        return 0.0
    snippet = text[idx + len(prefix):idx + len(prefix) + 30]
    m = re.search(r"[\d]+\.?[\d]*", snippet)
    return float(m.group()) if m else 0.0


def _extract_pack(text: str) -> float:
    """Extract pack/quantity from strings like 'pack of 6' or '12 count'."""
    import re
    patterns = [
        r"pack\s+of\s+(\d+)",
        r"\((\d+)\s+pack\)",
        r"(\d+)\s*(?:count|pcs|pieces|ct\b)",
        r"set\s+of\s+(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    return 1.0


# ---------------------------------------------------------------------------
# OPTIONAL: BERT Sentence Embeddings
# ---------------------------------------------------------------------------
def extract_bert_features(df: pd.DataFrame, batch_size: int = 256) -> np.ndarray:
    """
    Extract 384-dim sentence embeddings from catalog_content using
    all-MiniLM-L6-v2 (fast, lightweight BERT model).

    Install: pip install sentence-transformers
    Time: ~20 min for 75,000 rows on CPU; ~3 min on GPU
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for BERT features.\n"
            "Install with: pip install sentence-transformers"
        )

    log.info("Loading BERT model (all-MiniLM-L6-v2) — downloads ~90MB on first run ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = df["catalog_content"].fillna("").tolist()
    log.info(f"Encoding {len(texts):,} texts (batch_size={batch_size}) ...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    log.info(f"BERT embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# OPTIONAL: Image Feature Extraction
# ---------------------------------------------------------------------------
def extract_image_features(df: pd.DataFrame, image_dir: str = "images") -> np.ndarray:
    """
    Extract CNN features from product images using ResNet50.

    Images are expected at  images/<sample_id>.jpg
    If an image is missing, a zero vector is used as fallback.

    This is GPU-accelerated automatically when CUDA is available.
    Expected time: ~45 min on Colab T4 GPU for 600k images.
    """
    try:
        import torch
        import torchvision.models as tv_models
        import torchvision.transforms as transforms
        from PIL import Image
        from tqdm import tqdm
    except ImportError:
        raise ImportError(
            "torch, torchvision, and Pillow are required for image features.\n"
            "Install with: pip install torch torchvision Pillow"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Extracting image features using ResNet50 on {device.upper()} ...")
    if device == "cpu":
        log.warning(
            "No GPU detected. Image feature extraction will be very slow on CPU.\n"
            "Use Google Colab with GPU runtime for reasonable speed."
        )

    # Load ResNet50 pre-trained, strip classifier head
    model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    FEATURE_DIM = 2048
    features = np.zeros((len(df), FEATURE_DIM), dtype=np.float32)
    image_dir = Path(image_dir)

    batch_size = 64 if device == "cuda" else 8
    n = len(df)

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Image batches"):
            batch_ids = df["sample_id"].iloc[start:start + batch_size].tolist()
            imgs = []
            valid_idx = []

            for i, sid in enumerate(batch_ids):
                img_path = image_dir / f"{sid}.jpg"
                if img_path.exists():
                    try:
                        img = Image.open(img_path).convert("RGB")
                        imgs.append(transform(img))
                        valid_idx.append(start + i)
                    except Exception:
                        pass  # fallback: zero vector

            if imgs:
                batch_tensor = torch.stack(imgs).to(device)
                out = model(batch_tensor).cpu().numpy()
                for j, idx in enumerate(valid_idx):
                    features[idx] = out[j]

    log.info(f"Image features shape: {features.shape}")
    return features


# ---------------------------------------------------------------------------
# 7-Model Training
# ---------------------------------------------------------------------------
MODELS = {
    "random_forest": RandomForestRegressor(
        n_estimators=200, max_depth=20, random_state=RANDOM_SEED, n_jobs=-1
    ),
    "extra_trees": ExtraTreesRegressor(
        n_estimators=200, max_depth=20, random_state=RANDOM_SEED, n_jobs=-1
    ),
    "xgboost": xgb.XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, n_jobs=-1, verbosity=0
    ),
    "lightgbm": lgb.LGBMRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        num_leaves=63, subsample=0.8,
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=RANDOM_SEED
    ),
    "ridge_regression": Ridge(alpha=10.0),
    # Neural network: sklearn MLPRegressor (CPU-optimized, no PyTorch needed)
    "neural_network": MLPRegressor(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_SEED,
        verbose=False,
    ),
}


def build_neural_network(input_dim: int):
    """Returns a PyTorch-based MLP wrapped for sklearn-style fit/predict."""
    try:
        import torch
        import torch.nn as nn

        class _MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 128), nn.ReLU(),
                    nn.Linear(128, 1),
                )

            def forward(self, x):
                return self.net(x)

        class TorchRegressorWrapper:
            """Thin sklearn-compatible wrapper around _MLP."""

            def __init__(self):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = _MLP().to(self.device)

            def fit(self, X, y, epochs=20, batch_size=2048, lr=1e-3):
                from torch.utils.data import DataLoader, TensorDataset
                Xt = torch.FloatTensor(X).to(self.device)
                yt = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
                loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
                opt = torch.optim.Adam(self.model.parameters(), lr=lr)
                loss_fn = nn.MSELoss()
                self.model.train()
                for ep in range(epochs):
                    for bx, by in loader:
                        opt.zero_grad()
                        loss_fn(self.model(bx), by).backward()
                        opt.step()
                    if (ep + 1) % 5 == 0:
                        log.info(f"  neural_network — epoch {ep+1}/{epochs}")
                return self

            def predict(self, X):
                self.model.eval()
                with torch.no_grad():
                    Xt = torch.FloatTensor(X).to(self.device)
                    return self.model(Xt).cpu().numpy().flatten()

        return TorchRegressorWrapper()

    except ImportError:
        log.warning("PyTorch not available — skipping neural network model.")
        return None


def cross_validate_model(name: str, model, X: np.ndarray, y: np.ndarray,
                          n_folds: int = 5) -> float:
    """5-fold CV, returns mean SMAPE."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_smapes = []
    for fold_idx, (tr, val) in enumerate(kf.split(X)):
        X_tr, X_val = X[tr], X[val]
        y_tr, y_val = y[tr], y[val]

        model.fit(X_tr, y_tr)
        preds = np.expm1(np.clip(model.predict(X_val), 0, 15))  # convert back for SMAPE
        s = smape(np.expm1(y_val), preds)
        fold_smapes.append(s)
        log.info(f"  {name} — fold {fold_idx+1}/{n_folds}: SMAPE={s:.2f}%")

    return float(np.mean(fold_smapes))


def train_all_models(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray) -> tuple:
    """
    Train all 7 models, run 5-fold CV, build weighted ensemble.

    Returns: (ensemble_predictions, model_results_dict)
    """
    results = {}
    final_preds = {}

    for name, model in MODELS.items():
        log.info(f"\n{'='*55}")
        log.info(f"Training: {name.upper()}")
        log.info(f"{'='*55}")

        t0 = time.time()
        cv_smape = cross_validate_model(name, model, X_train, y_train)
        elapsed = time.time() - t0

        # Retrain on full training set
        model.fit(X_train, y_train)
        test_preds = np.clip(model.predict(X_test), 0.01, None)

        results[name] = {"cv_smape": cv_smape, "train_time_s": elapsed}
        final_preds[name] = test_preds

        # Save model
        save_path = MODELS_DIR / f"{name}_model.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

        log.info(f"  CV SMAPE: {cv_smape:.2f}%  |  Time: {elapsed:.0f}s  |  Saved: {save_path}")

    # ---- Weighted ensemble: weight = 1 / SMAPE (lower SMAPE → higher weight)
    weights = {n: 1.0 / r["cv_smape"] for n, r in results.items()}
    total_w = sum(weights.values())
    weights = {n: w / total_w for n, w in weights.items()}

    log.info("\nEnsemble weights:")
    for n, w in sorted(weights.items(), key=lambda x: -x[1]):
        log.info(f"  {n}: {w:.3f}  (CV SMAPE {results[n]['cv_smape']:.2f}%)")

    ensemble = np.zeros(len(X_test), dtype=np.float64)
    for name, w in weights.items():
        ensemble += w * final_preds[name]
    ensemble = np.clip(ensemble, 0.01, None)

    return ensemble, results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Product Price Predictor")
    parser.add_argument("--images", action="store_true",
                        help="Include image features (requires GPU for speed)")
    parser.add_argument("--bert", action="store_true",
                        help="Include BERT sentence embeddings (requires: pip install sentence-transformers)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Use a random sample of N rows (for quick testing)")
    parser.add_argument("--no-cv", action="store_true",
                        help="Skip cross-validation (faster, less info)")
    args = parser.parse_args()

    mode = "TEXT ONLY"
    if args.bert and args.images:
        mode = "TEXT + BERT + IMAGES"
    elif args.bert:
        mode = "TEXT + BERT"
    elif args.images:
        mode = "TEXT + IMAGES"

    log.info("=" * 60)
    log.info("PRODUCT PRICE PREDICTOR — Training Pipeline")
    log.info("=" * 60)
    log.info(f"Mode: {mode}")

    # ---- Load data ----
    log.info(f"\nLoading {TRAIN_CSV} ...")
    train_df = pd.read_csv(TRAIN_CSV)
    log.info(f"Loading {TEST_CSV} ...")
    test_df = pd.read_csv(TEST_CSV)

    if args.sample:
        log.info(f"Sampling {args.sample} rows for quick test ...")
        train_df = train_df.sample(n=min(args.sample, len(train_df)),
                                   random_state=RANDOM_SEED).reset_index(drop=True)

    log.info(f"Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")
    y_train = np.log1p(train_df["price"].values.astype(np.float64))  # log(1+price)

    # ---- Text features ----
    X_train_text, scaler = extract_text_features(train_df)
    X_test_text, _ = extract_text_features(test_df, fit_scaler=scaler)

    feature_parts_train = [X_train_text]
    feature_parts_test = [X_test_text]

    # ---- BERT embeddings (optional) ----
    if args.bert:
        log.info("\nStarting BERT feature extraction ...")
        X_train_bert = extract_bert_features(train_df)
        X_test_bert = extract_bert_features(test_df)
        feature_parts_train.append(X_train_bert)
        feature_parts_test.append(X_test_bert)
        log.info(f"BERT features added: {X_train_bert.shape[1]} dims")

    # ---- Image features (optional) ----
    if args.images:
        log.info("\nStarting image feature extraction ...")
        log.info("NOTE: Download images first with:  python basic_image_downloader.py")
        X_train_img = extract_image_features(train_df, image_dir="images")
        X_test_img = extract_image_features(test_df, image_dir="images")
        feature_parts_train.append(X_train_img)
        feature_parts_test.append(X_test_img)
        log.info(f"Image features added: {X_train_img.shape[1]} dims")

    X_train = np.hstack(feature_parts_train)
    X_test = np.hstack(feature_parts_test)
    log.info(f"Total feature dimensions: {X_train.shape[1]}")

    # ---- Train models ----
    log.info(f"\nTraining 7 models on {X_train.shape[0]:,} samples × {X_train.shape[1]} features ...")
    ensemble_preds, model_results = train_all_models(X_train, y_train, X_test)

    # ---- Summary ----
    log.info("\n" + "=" * 60)
    log.info("TRAINING SUMMARY")
    log.info("=" * 60)
    best_model = min(model_results, key=lambda n: model_results[n]["cv_smape"])
    for name, r in sorted(model_results.items(), key=lambda x: x[1]["cv_smape"]):
        marker = " <-- BEST" if name == best_model else ""
        log.info(f"  {name:25s}  SMAPE: {r['cv_smape']:.2f}%{marker}")

    # ---- Write predictions ----
    output_df = pd.DataFrame({
        "sample_id": test_df["sample_id"],
        "price": np.expm1(ensemble_preds),  # undo log: exp(pred) - 1
    })
    output_df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"\nPredictions saved to: {OUTPUT_CSV}")
    log.info(f"Rows: {len(output_df):,}  |  Price range: {ensemble_preds.min():.2f} – {ensemble_preds.max():.2f}")

    log.info("\nDone! Submit test_out.csv to the competition portal.")


if __name__ == "__main__":
    main()
