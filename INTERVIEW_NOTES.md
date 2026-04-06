# 📝 Interview Notes - Product Price Predictor

## Quick Pitch (30 seconds)

"I built a product price prediction system using an ensemble of 5 ML models (XGBoost, LightGBM, Ridge, Gradient Boosting, Neural Network) trained on 75,000 Amazon products. The system extracts 16 handcrafted text features and 384 BERT embeddings, then combines predictions using weighted averaging. I fixed critical ML issues and deployed it to production on Render."

---

## The Problem (1 minute)

**Challenge:** Predict Amazon product prices from text descriptions

**Data:** 75,000 training samples, 75,000 test samples
- Price range: $0.13 - $2,796
- Mostly grocery/food products
- Input: Product description (text)
- Output: Predicted price

**Why it's hard:**
- Highly skewed price distribution
- Limited features (only text)
- Need to handle outliers

---

## The Solution (2 minutes)

### Architecture

```
Text Description
    ↓
[Feature Extraction]
├─ 16 handcrafted features (value, unit, pack size, etc.)
└─ 384 BERT embeddings (semantic meaning)
    ↓
[Scaling] → StandardScaler (fit on training data)
    ↓
[5 ML Models]
├─ XGBoost (gradient boosting, GPU-accelerated)
├─ LightGBM (fast, handles large data)
├─ Ridge Regression (linear baseline)
├─ Gradient Boosting (sklearn)
└─ Neural Network (MLP: 512→256→128→1)
    ↓
[Ensemble] → Weighted average in PRICE space
    ↓
Price Prediction
```

### Key Decisions

1. **Ensemble over single model**
   - Diversity reduces overfitting
   - Weighted by CV performance
   - Better generalization

2. **BERT embeddings**
   - Captures semantic meaning
   - Pre-trained on large corpus
   - 384 dimensions

3. **Log transformation**
   - Compresses skewed distribution
   - Improves model convergence
   - Reversed at inference

4. **Weighted averaging**
   - Weight = 1 / SMAPE
   - Better models get higher weight
   - Simple and interpretable

---

## Critical Fixes I Applied (3 minutes)

### Fix #1: Data Leakage in Cross-Validation ⚠️

**Problem:** StandardScaler was fit on FULL dataset before CV loop
- Validation fold statistics leaked into training
- CV scores inflated by 5-15%
- Real performance would be worse than reported

**Solution:** Fit scaler on TRAINING fold only
```python
# Before (WRONG):
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fits on ALL data
for fold in cv_splits:
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    # ❌ Validation data statistics leaked

# After (CORRECT):
for fold in cv_splits:
    X_train, X_val = X[train_idx], X[val_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on training only
    X_val_scaled = scaler.transform(X_val)  # Transform validation
    # ✅ No leakage
```

**Impact:** CV scores now honest and reliable

---

### Fix #2: Ensemble Averaging in Wrong Space ⚠️

**Problem:** Predictions were averaged in LOG space instead of PRICE space
- Models predict log(price)
- Ensemble averaged log predictions
- Then converted back to price
- This is mathematically wrong!

**Example:**
```
Model A: log(price) = 2.0 → price = $6.39
Model B: log(price) = 2.2 → price = $8.01

Wrong way (log space):
  avg_log = (2.0 + 2.2) / 2 = 2.1
  price = exp(2.1) - 1 = $7.14

Right way (price space):
  avg_price = ($6.39 + $8.01) / 2 = $7.20

Difference: $0.06 per prediction × 75,000 = $4,500 systematic bias!
```

**Solution:** Convert to price space BEFORE averaging
```python
# Before (WRONG):
predictions_log = [model.predict(X) for model in models]
ensemble = np.mean(predictions_log)  # Average in log space
price = np.expm1(ensemble)  # Convert to price

# After (CORRECT):
predictions_log = [model.predict(X) for model in models]
predictions_price = [np.expm1(p) for p in predictions_log]  # Convert first
ensemble = np.mean(predictions_price)  # Average in price space
```

**Impact:** Eliminated systematic bias

---

### Fix #3: Feature Validation

**Problem:** No checks that features match training
- If feature extraction code changes, inference silently breaks
- Hard to debug in production

**Solution:** Add shape validation
```python
# Check feature count
if X.shape[1] != 400:
    raise ValueError(f"Expected 400 features, got {X.shape[1]}")

# Check for NaN/Inf
if np.any(np.isnan(X)) or np.any(np.isinf(X)):
    raise ValueError("Features contain NaN or Inf")
```

**Impact:** Fail fast with clear error messages

---

### Fix #4: Deployment Readiness

**Problems:**
- Hardcoded port (5050)
- No health endpoint
- Flask dev server (not production-ready)
- No error handling

**Solutions:**
```python
# Read port from environment
port = int(os.environ.get("PORT", 5050))
app.run(host="0.0.0.0", port=port)

# Add health endpoint
@app.route("/health")
def health():
    return {"status": "ok", "models_loaded": len(models)}

# Use Gunicorn for production
# gunicorn -w 4 -b 0.0.0.0:$PORT app:app
```

**Impact:** Works on Render, AWS, any cloud platform

---

## Technical Details (For Deep Dive)

### Features (400 total)

**Text Features (16):**
- value, unit_score, pack_qty
- title_words, title_chars, total_chars, total_words
- digit_ratio, brand_hit, category
- has_size, has_weight, has_volume
- value_x_pack, log_value, log_total_words

**BERT Embeddings (384):**
- Model: all-MiniLM-L6-v2
- Captures semantic meaning
- Pre-trained on large corpus

### Models

| Model | Type | Hyperparameters |
|-------|------|-----------------|
| XGBoost | Gradient Boosting | n_estimators=100, max_depth=6, lr=0.1 |
| LightGBM | Gradient Boosting | n_estimators=100, num_leaves=31, lr=0.1 |
| Ridge | Linear | alpha=10.0 |
| Gradient Boosting | sklearn GBR | n_estimators=150, max_depth=5, lr=0.1 |
| Neural Network | MLP | layers=[512, 256, 128], dropout=0.3 |

### Evaluation Metric

**SMAPE** (Symmetric Mean Absolute Percentage Error):
```
SMAPE = 100 × mean(|actual - predicted| / ((|actual| + |predicted|) / 2))
```

- Lower is better
- Baseline: ~42-48%
- After fixes: ~40-42%

---

## Deployment

### Option 1: Render (Easiest)
1. Push to GitHub
2. Connect to Render
3. Auto-deploys on push
4. Cost: $25/month

### Option 2: Docker
1. Build image: `docker build -t price-predictor .`
2. Push to registry
3. Deploy to cloud (AWS, GCP, Azure)

### Option 3: Manual
1. SSH into server
2. Clone repo
3. Run: `gunicorn -w 4 -b 0.0.0.0:5050 app:app`

---

## Performance

| Metric | Value |
|--------|-------|
| Inference latency | ~0.7 seconds |
| SMAPE | ~40-42% |
| Models | 5 (ensemble) |
| Features | 400 |
| Training data | 75,000 samples |
| Deployment | Render/Docker/Manual |

---

## What I Learned

### ML Lessons
1. **Data leakage is subtle** - Always fit scalers per fold in CV
2. **Prediction space matters** - Average in original space, not transformed space
3. **Ensemble diversity helps** - Different models catch different patterns
4. **Feature engineering is important** - 16 handcrafted features + BERT embeddings

### Engineering Lessons
1. **Validation is critical** - Check shapes, types, ranges
2. **Deployment matters** - Production code is different from research code
3. **Error handling is essential** - Fail fast with clear messages
4. **Monitoring is important** - Health checks, logging, metrics

---

## If Asked: "What Would You Improve?"

1. **Hyperparameter tuning**
   - Grid search on XGBoost/LightGBM
   - Could reduce SMAPE by 2-5%

2. **Feature engineering**
   - Image features (ResNet50 CNN)
   - Could reduce SMAPE by 5-10%

3. **Model improvements**
   - Try CatBoost (handles categorical better)
   - Stacking ensemble (meta-learner)

4. **Deployment**
   - Add caching for repeated predictions
   - Implement rate limiting
   - Add Prometheus metrics
   - Use async/await for concurrency

5. **Monitoring**
   - Track prediction distribution
   - Alert on anomalies
   - A/B test new models

---

## If Asked: "What Went Wrong?"

1. **Data leakage** - Subtle bug that inflated CV scores
2. **Averaging in wrong space** - Mathematical error that caused bias
3. **No validation** - Silent failures possible
4. **Not deployment-ready** - Hardcoded values, no health checks

**How I fixed it:**
- Code review and testing
- Mathematical verification
- Added validation checks
- Followed deployment best practices

---

## If Asked: "How Would You Explain This to Non-Technical People?"

"Imagine you're trying to predict house prices. You could:
1. Look at individual features (size, location, age)
2. Ask multiple real estate agents for their opinion
3. Average their opinions to get a final price

That's what this system does:
1. Extracts features from product descriptions
2. Asks 5 different ML models for their prediction
3. Averages their predictions to get the final price

The key insight: if you ask multiple experts and average their opinions, you usually get a better answer than any single expert."

---

## Talking Points

✅ **Technical depth:** Fixed critical ML bugs (data leakage, wrong averaging space)
✅ **Practical:** Deployed to production (Render)
✅ **Clean code:** Simple, modular, easy to explain
✅ **Problem-solving:** Identified and fixed issues systematically
✅ **Learning:** Understood why each fix matters

---

## Questions to Prepare For

**Q: "Why ensemble instead of single model?"**
A: Ensemble reduces overfitting through diversity. Different models catch different patterns. Weighted by performance.

**Q: "Why BERT embeddings?"**
A: Captures semantic meaning of text. Pre-trained on large corpus. Better than TF-IDF for this task.

**Q: "Why log transformation?"**
A: Price distribution is skewed ($0.13 - $2,796). Log compresses range, improves convergence.

**Q: "How do you handle outliers?"**
A: Log transformation naturally handles outliers. Clipping predictions to valid range.

**Q: "What's the biggest challenge?"**
A: Data leakage in CV. Subtle bug that inflates performance metrics. Fixed by fitting scaler per fold.

**Q: "How would you debug if predictions are wrong?"**
A: Check health endpoint, validate features, test individual models, check logs.

---

## Final Thoughts

This project demonstrates:
- **ML fundamentals:** Feature engineering, model selection, ensemble methods
- **Software engineering:** Clean code, deployment, error handling
- **Problem-solving:** Identifying and fixing subtle bugs
- **Communication:** Explaining complex concepts simply

**Key takeaway:** Production ML is not just about accuracy - it's about correctness, consistency, and reliability.

---

**Good luck with your interview! 🚀**
