# Product Price Predictor

**Author: Ayush Anand**
Amazon ML Challenge — Predict product prices from text descriptions using an ensemble of 5 machine learning models + BERT embeddings.

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
4. Combining predictions into a **weighted ensemble** (better models get higher weight)

---

## Models Used

| Model | Type | Notes |
|-------|------|-------|
| XGBoost | Gradient Boosting | GPU-accelerated, best performer |
| LightGBM | Gradient Boosting | Fast, handles large datasets |
| Neural Network | MLP (512→256→128→1) | Deep learning baseline |
| Ridge Regression | Linear | Fast, good regularization |
| Gradient Boosting | sklearn GBR | Slower but robust |

---

## Project Structure

```
student_resource/
├── train_models.py          # Main training script (run this to retrain)
├── app.py                   # Flask web server for real-time prediction UI
├── predict.py               # Command-line predictor
├── scaler.pkl               # Fitted StandardScaler (from training)
│
├── dataset/
│   ├── train.csv            # 75,000 training samples (not in GitHub)
│   ├── test.csv             # 75,000 test samples (not in GitHub)
│   └── sample_test.csv      # Small sample for testing
│
├── models/
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── neural_network_model.pkl
│   ├── ridge_regression_model.pkl
│   └── gradient_boosting_model.pkl
│
├── notebooks/
│   └── colab_full_pipeline.ipynb   # Full training on Google Colab (GPU)
│
├── ui/
│   └── index.html           # Web UI (served by app.py)
│
├── src/                     # Source pipeline modules
└── tests/                   # Unit tests
```

---

## How to Run

### Option 1 — Web UI (recommended)

Start the server:
```bash
cd student_resource
python app.py
```
Open **http://localhost:5050** in your browser.

Type any product description → click **Predict Price** → see results from all 5 models instantly.

> **First run** downloads the BERT model (~90MB). Takes ~30 seconds.

---

### Option 2 — Command Line

Single prediction:
```bash
python predict.py --text "Item Name: Nescafe Coffee 200g
Value: 200.0
Unit: Grams"
```

Interactive mode (type multiple products):
```bash
python predict.py
```

---

### Option 3 — Retrain the Models

**Locally (text only, ~40 min):**
```bash
python train_models.py
```

**Locally with BERT (~2 hrs on CPU):**
```bash
pip install sentence-transformers
python train_models.py --bert
```

**On Google Colab with GPU (~30 min, recommended):**
1. Open `notebooks/colab_full_pipeline.ipynb` in [colab.research.google.com](https://colab.research.google.com)
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Upload `dataset/train.csv` and `dataset/test.csv`
4. Run all cells
5. Download `test_out.csv` and `trained_models.zip`

**Quick test on small sample:**
```bash
python train_models.py --sample 2000
```

---

## Setup

### Requirements
```bash
pip install numpy pandas scikit-learn xgboost lightgbm flask flask-cors sentence-transformers tqdm
```

Or using the requirements file:
```bash
pip install -r requirements.txt
```

### Python version
Python 3.10+ required.

---

## How It Works — Step by Step

```
train.csv (75,000 products)
        │
        ▼
┌─────────────────────────────────┐
│   Feature Extraction            │
│   • 16 regex text features      │  ← pack size, brand, weight, units...
│   • 384 BERT embeddings         │  ← semantic meaning of description
│   Total: 400 features/product   │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│   Log-transform price target    │  ← log(1+price) compresses skew
│   price $0.13–$2,796 → 0–8.5   │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│   5-Fold Cross Validation       │  ← honest performance estimate
│   Train on 4 folds, test on 1   │
│   Rotate 5 times, average SMAPE │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│   Train 5 Models on full data   │
│   XGBoost / LightGBM / NN /     │
│   Ridge / GradientBoosting      │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│   Weighted Ensemble             │  ← weight = 1/SMAPE
│   Better model → higher weight  │
│   Predictions combined          │
└─────────────────────────────────┘
        │
        ▼
   test_out.csv (75,000 predictions)
```

---

## Evaluation Metric

**SMAPE** — Symmetric Mean Absolute Percentage Error:

```
SMAPE = 100 × mean( |actual - predicted| / ((|actual| + |predicted|) / 2) )
```

Lower is better. Current baseline: **~42–48% SMAPE** (text + BERT features).

---

## Training Data

- **75,000 Amazon product listings** — mostly grocery, food & beverage items
- Price range: $0.13 – $2,796
- Average price: $23.65
- Source: Amazon ML Challenge dataset

> The model is most accurate for grocery/food products since that is what most training data contains.

---

## Improving the Model

| Improvement | Expected SMAPE drop | How |
|------------|--------------------|----|
| Image features (ResNet50 CNN) | -5 to -10% | Run Colab notebook with `--images` flag |
| More training data | -5 to -15% | Add diverse product categories |
| Hyperparameter tuning | -2 to -5% | Grid search on XGBoost/LightGBM |
| CatBoost as 6th model | -1 to -3% | `pip install catboost` |
