#!/usr/bin/env python3
"""
Simple validation script to verify all fixes are working
"""

import sys
import numpy as np
from pathlib import Path

print("=" * 60)
print("VALIDATING FIXES")
print("=" * 60)

# Test 1: Check models exist
print("\n1. Checking models...")
models_dir = Path("models")
required_models = [
    "xgboost_model.pkl",
    "lightgbm_model.pkl",
    "ridge_regression_model.pkl",
    "gradient_boosting_model.pkl"
]

for model in required_models:
    path = models_dir / model
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"   ✓ {model} ({size_mb:.1f} MB)")
    else:
        print(f"   ✗ {model} NOT FOUND")
        sys.exit(1)

# Test 2: Check scaler exists
print("\n2. Checking scaler...")
if Path("scaler.pkl").exists():
    print("   ✓ scaler.pkl found")
else:
    print("   ✗ scaler.pkl NOT FOUND")
    sys.exit(1)

# Test 3: Test feature extraction
print("\n3. Testing feature extraction...")
try:
    import pickle
    from app import extract_features, scaler
    
    test_text = "Item Name: Coffee\nValue: 200\nUnit: Grams"
    features = extract_features(test_text)
    
    if features.shape == (1, 16):
        print(f"   ✓ Feature extraction works (shape: {features.shape})")
    else:
        print(f"   ✗ Feature shape wrong: expected (1, 16), got {features.shape}")
        sys.exit(1)
    
    # Test scaling
    scaled = scaler.transform(features)
    if scaled.shape == (1, 16):
        print(f"   ✓ Scaler works (shape: {scaled.shape})")
    else:
        print(f"   ✗ Scaler shape wrong: expected (1, 16), got {scaled.shape}")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Feature extraction failed: {e}")
    sys.exit(1)

# Test 4: Test BERT embeddings
print("\n4. Testing BERT embeddings...")
try:
    from sentence_transformers import SentenceTransformer
    
    bert = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = bert.encode([test_text], convert_to_numpy=True)
    
    if embedding.shape == (1, 384):
        print(f"   ✓ BERT embeddings work (shape: {embedding.shape})")
    else:
        print(f"   ✗ BERT shape wrong: expected (1, 384), got {embedding.shape}")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ BERT failed: {e}")
    sys.exit(1)

# Test 5: Test combined features
print("\n5. Testing combined features...")
try:
    combined = np.hstack([scaled, embedding])
    
    if combined.shape == (1, 400):
        print(f"   ✓ Combined features correct (shape: {combined.shape})")
    else:
        print(f"   ✗ Combined shape wrong: expected (1, 400), got {combined.shape}")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Feature combination failed: {e}")
    sys.exit(1)

# Test 6: Test model prediction
print("\n6. Testing model predictions...")
try:
    import pickle
    
    # Load a model
    with open("models/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Make prediction
    pred_log = model.predict(combined)[0]
    
    # Convert to price (this is the FIX - convert from log space)
    price = float(np.expm1(np.clip(pred_log, 0, 15)))
    
    if price > 0 and price < 10000:
        print(f"   ✓ Model prediction works (price: ${price:.2f})")
    else:
        print(f"   ✗ Prediction out of range: ${price:.2f}")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Model prediction failed: {e}")
    sys.exit(1)

# Test 7: Test ensemble averaging in price space
print("\n7. Testing ensemble (price space averaging)...")
try:
    prices = []
    for name in ["xgboost", "lightgbm", "ridge_regression", "gradient_boosting"]:
        with open(f"models/{name}_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        pred_log = model.predict(combined)[0]
        price = float(np.expm1(np.clip(pred_log, 0, 15)))
        prices.append(price)
    
    # ✅ FIX: Average in price space (not log space)
    ensemble_price = np.mean(prices)
    
    if ensemble_price > 0 and ensemble_price < 10000:
        print(f"   ✓ Ensemble works (price: ${ensemble_price:.2f})")
        print(f"     Individual prices: {[f'${p:.2f}' for p in prices]}")
    else:
        print(f"   ✗ Ensemble price out of range: ${ensemble_price:.2f}")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Ensemble failed: {e}")
    sys.exit(1)

# Test 8: Test Flask app
print("\n8. Testing Flask app...")
try:
    from app import app
    
    client = app.test_client()
    
    # Test health endpoint
    response = client.get("/health")
    if response.status_code == 200:
        print(f"   ✓ Health endpoint works")
    else:
        print(f"   ✗ Health endpoint failed: {response.status_code}")
        sys.exit(1)
    
    # Test prediction endpoint
    response = client.post("/predict", json={
        "description": "Item Name: Coffee\nValue: 200\nUnit: Grams"
    })
    
    if response.status_code == 200:
        data = response.get_json()
        if "ensemble" in data and "models" in data:
            print(f"   ✓ Prediction endpoint works")
            print(f"     Ensemble: ${data['ensemble']:.2f}")
        else:
            print(f"   ✗ Response missing fields: {data}")
            sys.exit(1)
    else:
        print(f"   ✗ Prediction endpoint failed: {response.status_code}")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Flask app test failed: {e}")
    sys.exit(1)

# All tests passed
print("\n" + "=" * 60)
print("✅ ALL VALIDATION TESTS PASSED!")
print("=" * 60)
print("\nThe system is ready for deployment.")
print("\nNext steps:")
print("1. Push to GitHub: git push origin main")
print("2. Deploy to Render: https://render.com")
print("3. Or run locally: python app.py")
print("\nFor deployment guide, see: DEPLOYMENT_GUIDE.md")
