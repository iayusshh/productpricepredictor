# 🚀 Deployment Guide - Product Price Predictor

## Quick Summary of Fixes Applied

### 1. ✅ Fixed Data Leakage in Cross-Validation
- **File:** `src/models/cross_validator.py`
- **Fix:** StandardScaler now fit on TRAINING fold only (not full dataset)
- **Impact:** CV scores are now honest, not inflated

### 2. ✅ Fixed Ensemble Prediction Space
- **Files:** `src/models/ensemble_manager.py`, `app.py`, `predict.py`
- **Fix:** Predictions averaged in PRICE space (not log space)
- **Impact:** Eliminated systematic bias in predictions

### 3. ✅ Added Feature Validation
- **File:** `app.py`
- **Fix:** Check feature shape before prediction
- **Impact:** Fail fast if feature extraction breaks

### 4. ✅ Added Health Check Endpoint
- **File:** `app.py`
- **Fix:** GET /health returns {"status": "ok"}
- **Impact:** Render can monitor app health

### 5. ✅ Made App Deployment-Ready
- **File:** `app.py`
- **Fix:** Read PORT from environment variable
- **Impact:** Works on Render, AWS, any cloud platform

### 6. ✅ Added Gunicorn Support
- **Files:** `Procfile`, `requirements.txt`, `Dockerfile`
- **Fix:** Production-grade WSGI server configuration
- **Impact:** Can handle concurrent requests

---

## Deployment Option 1: Render (Easiest)

### Step 1: Push to GitHub

```bash
cd productpricepredictor
git add .
git commit -m "Fix ML issues and prepare for production"
git push origin main
```

### Step 2: Create Render Service

1. Go to https://render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repo
4. Select branch: `main`
5. Fill in:
   - **Name:** `price-predictor`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn -w 4 -b 0.0.0.0:$PORT --timeout 120 app:app`
6. Click "Create Web Service"

### Step 3: Wait for Deployment

- Render will auto-build and deploy
- Check logs in dashboard
- Service will be live at: `https://price-predictor.onrender.com`

### Step 4: Test

```bash
curl -X POST https://price-predictor.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "Item Name: Coffee\nValue: 200\nUnit: Grams"}'

# Expected response:
# {"ensemble": 9.23, "models": {...}, "description": "..."}
```

---

## Deployment Option 2: Docker (Local or Cloud)

### Step 1: Build Docker Image

```bash
cd productpricepredictor
docker build -t price-predictor:1.0 .
```

### Step 2: Test Locally

```bash
docker run -p 5050:5050 price-predictor:1.0

# In another terminal:
curl -X POST http://localhost:5050/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "Item Name: Coffee\nValue: 200\nUnit: Grams"}'
```

### Step 3: Push to Docker Hub (Optional)

```bash
docker tag price-predictor:1.0 yourusername/price-predictor:1.0
docker push yourusername/price-predictor:1.0
```

### Step 4: Deploy to Cloud

**AWS ECS:**
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker tag price-predictor:1.0 $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/price-predictor:1.0
docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/price-predictor:1.0
```

**Google Cloud Run:**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/price-predictor
gcloud run deploy price-predictor --image gcr.io/PROJECT_ID/price-predictor --platform managed
```

---

## Deployment Option 3: Manual (VPS/Server)

### Step 1: SSH into Server

```bash
ssh user@your-server.com
```

### Step 2: Clone Repository

```bash
git clone https://github.com/yourusername/productpricepredictor.git
cd productpricepredictor
```

### Step 3: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 4: Run with Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:5050 --timeout 120 app:app
```

### Step 5: Setup Nginx Reverse Proxy (Optional)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Testing After Deployment

### 1. Health Check

```bash
curl https://your-app.com/health
# Expected: {"status": "ok", "models_loaded": 5}
```

### 2. Sample Prediction

```bash
curl -X POST https://your-app.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Item Name: Nescafe Classic Instant Coffee, 200g\nValue: 200.0\nUnit: Grams"
  }'

# Expected response:
# {
#   "ensemble": 9.23,
#   "models": {
#     "xgboost": 9.15,
#     "lightgbm": 9.31,
#     "neural_network": 9.25,
#     "ridge_regression": 9.20,
#     "gradient_boosting": 9.28
#   },
#   "description": "Item Name: Nescafe Classic..."
# }
```

### 3. Verify Predictions are Reasonable

- All prices should be positive
- Prices should be in range $0.01 - $2,796
- Ensemble should be average of individual models
- No NaN or Inf values

---

## Monitoring

### Check Logs

**Render:**
```
Dashboard → Logs tab
```

**Docker:**
```bash
docker logs <container_id>
```

**Manual:**
```bash
tail -f /var/log/gunicorn.log
```

### Common Issues

**Issue:** Models fail to load
```
Solution: Check models/ directory exists and has .pkl files
```

**Issue:** BERT model download fails
```
Solution: First request will download ~90MB, takes 30 seconds
```

**Issue:** Feature mismatch error
```
Solution: Ensure feature extraction code hasn't changed
```

---

## Performance Tuning

### Increase Workers (if CPU-bound)

```bash
gunicorn -w 8 -b 0.0.0.0:5050 app:app  # 8 workers instead of 4
```

### Increase Timeout (if requests timeout)

```bash
gunicorn -w 4 -b 0.0.0.0:5050 --timeout 300 app:app  # 5 minutes
```

### Enable Caching (if same predictions requested)

```python
# In app.py (optional enhancement)
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_predict(description):
    # ... prediction logic
```

---

## Rollback Plan

If something goes wrong:

**Render:**
1. Go to Dashboard
2. Click "Deployments"
3. Select previous version
4. Click "Deploy"

**Docker:**
```bash
docker run -p 5050:5050 price-predictor:v1.0  # Previous version
```

**Manual:**
```bash
git checkout previous-commit
pip install -r requirements.txt
gunicorn -w 4 -b 0.0.0.0:5050 app:app
```

---

## Production Checklist

- [ ] All fixes applied
- [ ] Tests pass locally
- [ ] Docker image builds
- [ ] Health endpoint works
- [ ] Sample prediction works
- [ ] Predictions are reasonable
- [ ] No errors in logs
- [ ] Latency is acceptable (<2 seconds)
- [ ] Monitoring is set up
- [ ] Rollback plan documented

---

## Interview Explanation

**"What did you fix?"**

1. **Data Leakage:** Fixed cross-validation to fit scaler per fold, not on full dataset
2. **Ensemble:** Fixed predictions to average in price space, not log space
3. **Validation:** Added feature shape validation to catch bugs early
4. **Deployment:** Added health endpoint, Gunicorn support, environment variable handling
5. **Monitoring:** Added error handling and logging

**"Why does it matter?"**

- Data leakage made CV scores unrealistic
- Averaging in wrong space caused systematic bias
- Feature validation prevents silent failures
- Proper deployment setup ensures reliability
- Monitoring helps debug issues in production

**"How would you improve it further?"**

- Add caching for repeated predictions
- Implement rate limiting
- Add Prometheus metrics
- Use async/await for better concurrency
- Add A/B testing framework

---

## Support

For issues:
1. Check logs
2. Verify models are loaded
3. Test health endpoint
4. Try sample prediction
5. Check feature extraction code

---

**Deployment is ready! 🚀**
