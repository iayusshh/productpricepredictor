# Product Price Predictor

Predict product prices from text descriptions using an ensemble of 5 machine learning models + BERT embeddings.

**Status:** ✅ Production-Ready | 🚀 Deployment-Ready | 📊 ML-Correct

---

## What This Project Does

Given a product description like:
```
Item Name: Nescafe Classic Instant Coffee, 200g
Value: 200.0
Unit: Grams
```
The model predicts its price — e.g. **$9.23**

It does this by:
1. Extracting 16 hand-crafted text features (pack size, weight, brand signals, etc.)
2. Encoding the description using **BERT** (all-MiniLM-L6-v2) → 384 semantic features
3. Feeding 400 total features into **5 ML models**
4. Combining predictions into a **weighted ensemble** (averaged in price space for accuracy)

---

## Models Used

| Model | Type | Performance |
|-------|------|-------------|
| XGBoost | Gradient Boosting | Best performer, GPU-accelerated |
| LightGBM | Gradient Boosting | Fast, handles large datasets |
| Ridge Regression | Linear | Fast baseline, good regularization |
| Gradient Boosting | sklearn GBR | Robust, slower |
| Neural Network | MLP (512→256→128→1) | Deep learning baseline |

**Ensemble Method:** Weighted average in PRICE space (not log space)
- Weight = 1 / SMAPE (better models get higher weight)
- Predictions averaged in original price space for accuracy
- Current SMAPE: **~40-42%**

---

## Project Structure

```
productpricepredictor/
├── app.py                          # Flask web server (production-ready)
├── predict.py                      # Command-line predictor
├── train_models.py                 # Training script
├── validate_fixes.py               # Validation script
│
├── Procfile                        # Render deployment config
├── Dockerfile                      # Docker deployment config
├── requirements.txt                # Python dependencies
│
├── DEPLOYMENT_GUIDE.md             # How to deploy (Render/Docker/Manual)
├── FIXES_SUMMARY.md                # Summary of all fixes applied
├── INTERVIEW_NOTES.md              # Interview preparation guide
├── README_FIXES.md                 # Quick reference for fixes
│
├── dataset/
│   ├── train.csv                   # 75,000 training samples
│   ├── test.csv                    # 75,000 test samples
│   └── sample_test.csv             # Small sample for testing
│
├── models/
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── ridge_regression_model.pkl
│   ├── gradient_boosting_model.pkl
│   └── scaler.pkl                  # StandardScaler (fit on training data)
│
├── src/
│   ├── models/
│   │   ├── cross_validator.py      # ✅ Fixed: CV with per-fold scaling
│   │   ├── ensemble_manager.py     # ✅ Fixed: Ensemble in price space
│   │   └── ...
│   ├── features/
│   ├── data_processing/
│   └── ...
│
├── ui/
│   └── index.html                  # Beautiful web UI
│
├── notebooks/
│   └── colab_full_pipeline.ipynb   # Google Colab training notebook
│
└── tests/
    └── ...                         # Unit tests
```

---

## Quick Start

### 1. Validate All Fixes (2 minutes)

```bash
python validate_fixes.py
```

Expected output: `✅ ALL VALIDATION TESTS PASSED!`

### 2. Run Web UI (Recommended)

```bash
python app.py
```

Open **http://localhost:5050** in your browser.

Type any product description → click **Predict Price** → see results from all 5 models instantly.

> **First run** downloads the BERT model (~90MB). Takes ~30 seconds.

### 3. Command Line

Single prediction:
```bash
python predict.py --text "Item Name: Nescafe Coffee 200g
Value: 200.0
Unit: Grams"
```

Interactive mode:
```bash
python predict.py
```

### 4. Deploy to Production

**Option A: Render (Easiest - 5 minutes)**
```bash
git push origin main
# Go to https://render.com → Create Web Service → Connect GitHub
# Auto-deploys on push
```

**Option B: Docker**
```bash
docker build -t price-predictor:1.0 .
docker run -p 5050:5050 price-predictor:1.0
```

**Option C: Manual**
```bash
gunicorn -w 4 -b 0.0.0.0:5050 --timeout 120 app:app
```

See **DEPLOYMENT_GUIDE.md** for detailed instructions.

---

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy pandas scikit-learn xgboost lightgbm flask flask-cors sentence-transformers gunicorn
```

### Python Version
Python 3.10+ required.

### First Time Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Validate everything works
python validate_fixes.py

# 3. Run the app
python app.py
```

---

## How It Works

### Pipeline Overview

```
Product Description
    ↓
[Feature Extraction]
├─ 16 handcrafted text features
│  (value, unit, pack size, brand, category, etc.)
└─ 384 BERT embeddings
   (semantic meaning of description)
    ↓
[StandardScaler]
(fit on training data only)
    ↓
[5 ML Models]
├─ XGBoost
├─ LightGBM
├─ Ridge Regression
├─ Gradient Boosting
└─ Neural Network
    ↓
[Ensemble - Weighted Average in PRICE SPACE]
(weight = 1/SMAPE, better models get higher weight)
    ↓
Price Prediction
```

### Key Features

1. **Correct Cross-Validation** ✅
   - StandardScaler fit on TRAINING fold only
   - No data leakage
   - Honest CV scores

2. **Correct Ensemble** ✅
   - Predictions averaged in PRICE space (not log space)
   - Eliminates systematic bias
   - Mathematically sound

3. **Feature Validation** ✅
   - Checks feature shape (400 expected)
   - Fails fast on errors
   - Clear error messages

4. **Production-Ready** ✅
   - Health endpoint (`GET /health`)
   - Environment variable support
   - Gunicorn + Docker support
   - Comprehensive error handling

---

## Critical Fixes Applied

### 1. ✅ Fixed Data Leakage in Cross-Validation
**Problem:** StandardScaler was fit on full dataset before CV loop
**Solution:** Fit scaler on TRAINING fold only
**Impact:** CV scores now honest and reliable
**File:** `src/models/cross_validator.py`

### 2. ✅ Fixed Ensemble Averaging
**Problem:** Predictions averaged in LOG space instead of PRICE space
**Solution:** Convert to price space BEFORE averaging
**Impact:** Eliminated ~$0.06 per prediction systematic bias
**Files:** `src/models/ensemble_manager.py`, `app.py`, `predict.py`

### 3. ✅ Added Feature Validation
**Problem:** Silent failures if features change
**Solution:** Added shape and type validation
**Impact:** Fail fast with clear error messages
**File:** `app.py`

### 4. ✅ Made Production-Ready
**Problem:** Not suitable for production deployment
**Solution:** Added health endpoint, env vars, Gunicorn support
**Impact:** Works on Render, AWS, any cloud platform
**Files:** `app.py`, `Procfile`, `Dockerfile`, `requirements.txt`

See **FIXES_SUMMARY.md** for detailed technical explanation.

---

## Performance

| Metric | Value |
|--------|-------|
| Inference latency | ~0.7 seconds |
| SMAPE | ~40-42% |
| Models | 5 (ensemble) |
| Features | 400 (16 text + 384 BERT) |
| Training data | 75,000 samples |
| Price range | $0.13 - $2,796 |
| Deployment | Render/Docker/Manual |

## Evaluation Metric

**SMAPE** (Symmetric Mean Absolute Percentage Error):
```
SMAPE = 100 × mean( |actual - predicted| / ((|actual| + |predicted|) / 2) )
```
Lower is better. Current: **~40-42%** (after fixes)


---

## Documentation

- **DEPLOYMENT_GUIDE.md** - How to deploy (Render/Docker/Manual)
- **FIXES_SUMMARY.md** - Detailed summary of all fixes
- **INTERVIEW_NOTES.md** - Interview preparation guide
- **README_FIXES.md** - Quick reference for fixes
- **validate_fixes.py** - Validation script

---

## Future Improvements

| Improvement | Expected SMAPE drop | How |
|------------|--------------------|----|
| Image features (ResNet50 CNN) | -5 to -10% | Extract visual features from product images |
| Hyperparameter tuning | -2 to -5% | Grid search on XGBoost/LightGBM |
| More training data | -5 to -15% | Add diverse product categories |
| CatBoost as 6th model | -1 to -3% | `pip install catboost` |
| Caching | Performance | Cache repeated predictions |
| Monitoring | Reliability | Add Prometheus metrics |

---

## Testing

Run validation script:
```bash
python validate_fixes.py
```

This validates:
- ✅ Models load correctly
- ✅ Feature extraction works
- ✅ BERT embeddings work
- ✅ Model predictions work
- ✅ Ensemble works
- ✅ Flask app works
- ✅ All endpoints respond

---

## Support & Issues

1. **Validation fails?** Check that models are in `models/` directory
2. **BERT download fails?** First request downloads ~90MB, takes 30 seconds
3. **Feature mismatch error?** Ensure feature extraction code hasn't changed
4. **Deployment issues?** See DEPLOYMENT_GUIDE.md

---

## Status

✅ **Production Ready**
- All critical ML issues fixed
- Deployment-ready (Render/Docker/Manual)
- Comprehensive documentation
- Validation script included

🚀 **Ready to Deploy!**
