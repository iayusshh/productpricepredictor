# Product Price Predictor

## Full Detailed Project Report

Prepared for: Amazon ML Challenge project workstream  
Prepared by: Ayush Anand  
Date: 8 April 2026

---

## 1. Executive Summary

This project implements an end-to-end machine learning system to predict product prices from catalog text, with optional support for image-derived features. The repository includes two practical paths:

1. A production inference stack (Flask API + web UI) for real-time predictions.
2. A standalone training pipeline that supports text features, optional BERT embeddings, and optional image features.

Core strengths of the current solution:

1. Multi-model ensemble strategy (tree models, linear model, neural network).
2. Strong feature stack for text (16 handcrafted features + 384-dimensional BERT embeddings in production inference).
3. Validation and deployment readiness (health checks, model loading safeguards, deployment manifests).
4. Practical test and validation scripts for quick quality checks.

Current constraints and observations:

1. There are workflow inconsistencies between historical documentation and current runtime behavior (for example weighted-average references vs robust median-based serving ensemble in the app).
2. Dataset row counts in current files differ from older challenge notes (detailed in Section 4).
3. Some model artifacts used in historical reports are not part of the active production serving path.

Overall assessment: the project is technically strong, production-aware, and suitable for iterative improvement. The next highest-impact gains are better data consistency checks, harmonized training-serving logic, and stronger experiment tracking.

---

## 2. Problem Statement and Business Context

### 2.1 Problem

Given a product description (catalog content), predict an appropriate product price. In optional multimodal mode, image content can also be used.

### 2.2 Why this matters

Accurate price estimation supports:

1. Faster onboarding of new listings.
2. Reduced manual pricing effort.
3. Better consistency across catalogs.
4. Decision support for sellers and internal catalog teams.

### 2.3 Evaluation metric

The project uses SMAPE (Symmetric Mean Absolute Percentage Error):

$$
SMAPE = 100 \times \frac{1}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2 + \epsilon}
$$

Interpretation:

1. Lower is better.
2. Relative error is emphasized, making it suitable for skewed price distributions.

---

## 3. Project Scope and Deliverables

### 3.1 Included deliverables

1. Training pipeline script for model development.
2. Prediction output artifact in CSV format.
3. Flask API with endpoints for health and prediction.
4. Frontend UI for user interaction.
5. Validation scripts and tests.
6. Deployment configuration for Render/Gunicorn.

### 3.2 Key runtime artifacts

1. Model files in models directory.
2. Scaler artifact for preprocessing consistency.
3. Prediction output file test_out.csv.

---

## 4. Data Profile and Observed Integrity Notes

### 4.1 Expected schema (high level)

Training data typically includes:

1. sample_id
2. catalog_content
3. image_link (optional usage)
4. price (target)

Test data typically includes:

1. sample_id
2. catalog_content
3. image_link (optional usage)

### 4.2 Current file-level observations

Based on file line counts in the current workspace snapshot:

1. dataset/train.csv: 600,209 lines (about 600,208 data rows if 1 header line).
2. dataset/test.csv: 599,366 lines (about 599,365 data rows if 1 header line).
3. test_out.csv: 75,001 lines (about 75,000 predictions if 1 header line).

### 4.3 Implication

There is a mismatch between current dataset file sizes and generated output row count. This may indicate:

1. Different dataset versions used in different runs.
2. Subset-based inference or legacy output retained in repository.
3. Multi-stage data preparation not documented in one canonical place.

Recommendation: introduce a mandatory pre-run data consistency check to ensure output rows match the intended test set row count for the run.

---

## 5. Solution Architecture

### 5.1 Training architecture (standalone path)

1. Load train/test CSV.
2. Parse catalog text into engineered numeric features.
3. Optionally add BERT embeddings.
4. Optionally add image features (ResNet50) if images are available.
5. Train 7 models with cross-validation.
6. Build ensemble prediction.
7. Export predictions to CSV.

### 5.2 Production serving architecture (app path)

1. Load available models and scaler at startup.
2. Load sentence-transformer model for BERT embeddings.
3. For each request:
4. Extract 16 handcrafted text features.
5. Scale text features using saved scaler.
6. Generate 384-dimensional embedding.
7. Concatenate into 400-dimensional feature vector.
8. Predict with each loaded model.
9. Convert model outputs from log-space to price-space.
10. Apply robust ensemble logic and return response with confidence signals.

### 5.3 API layer

1. GET / serves UI.
2. GET /health reports runtime status and loaded models.
3. POST /predict performs inference and returns model-level plus ensemble outputs.

---

## 6. Feature Engineering Details

### 6.1 Handcrafted text features (16)

Main extracted signals include:

1. Numeric value detection from text.
2. Unit score from unit keyword mapping.
3. Pack quantity extraction.
4. Title and full-text length statistics.
5. Digit ratio.
6. Brand hit flag.
7. Category id inferred from keywords.
8. Size/weight/volume indicator flags.
9. Composite features such as value multiplied by pack quantity.
10. Log transforms for skew-stabilized numeric signals.

### 6.2 Embedding features

In serving mode, sentence-transformer model all-MiniLM-L6-v2 provides:

1. 384-dimensional semantic embeddings.
2. Better context capture beyond regex-style parsing.

### 6.3 Optional image features

Training path supports ResNet50 extraction:

1. 2048-dimensional image vectors.
2. GPU-accelerated extraction recommended for scale.
3. Missing images fall back to zero vectors.

---

## 7. Modeling Strategy

### 7.1 Model family used in training pipeline

1. Random Forest Regressor
2. Extra Trees Regressor
3. XGBoost Regressor
4. LightGBM Regressor
5. Gradient Boosting Regressor
6. Ridge Regression
7. Neural Network (MLPRegressor)

### 7.2 Target transformation

Training uses:

1. log1p(price) as learning target.
2. expm1(prediction) during evaluation/serving conversion.

This is appropriate for long-tailed price distributions.

### 7.3 Ensemble logic

Two ensemble patterns exist in project materials:

1. Training workflow documentation references weighted averaging by inverse CV SMAPE.
2. Current production app uses robust median-style aggregation with outlier-guard handling for neural network extremes.

Recommendation: define and document one canonical ensemble policy per environment (training, offline scoring, serving), and keep them synchronized by explicit versioning.

---

## 8. Validation, Quality Controls, and Reliability

### 8.1 Validation layers present

1. Unit and integration tests under tests directory.
2. Dedicated validation scripts for model/scaler/feature shape checks.
3. API endpoint validation checks for health and prediction behavior.

### 8.2 Important ML correctness fixes reflected in project notes

1. Data leakage prevention in CV scaler handling.
2. Log-space to price-space conversion correctness.
3. Feature dimension validation before prediction.
4. Graceful model loading with partial fallback.

### 8.3 Runtime resilience features in app

1. Model-loading failure tolerance (non-fatal for some models).
2. Input quality assessment and warning messaging.
3. Confidence indicator in prediction response.
4. Outlier-guarded aggregation to reduce instability on sparse inputs.

---

## 9. Deployment and Operations

### 9.1 Deployment readiness status

Configured for cloud deployment using:

1. Gunicorn-based serving.
2. Procfile and render.yaml.
3. runtime pinning for Python compatibility.
4. Health endpoint suitable for platform probes.

### 9.2 Operational considerations

1. First-run embedding model download adds startup latency.
2. Model and dependency compatibility should be pinned and validated in CI.
3. Cache directories are redirected to temporary writable paths for hosted environments.

### 9.3 Monitoring baseline

Minimum recommended production telemetry:

1. Request latency percentiles.
2. Error rates by endpoint.
3. Model-level prediction dispersion and drift checks.
4. Input sparsity trend over time.

---

## 10. Current Risks and Gaps

1. Data-version mismatch risk between training and inference artifacts.
2. Multiple report narratives in repository (historical vs current behavior) can create confusion.
3. Inconsistent ensemble descriptions may reduce reproducibility.
4. Potential dependency drift across environments without lockstep pinning.
5. Limited centralized experiment tracking metadata in repository root.

---

## 11. Recommendations and Improvement Roadmap

### 11.1 Immediate (high value, low effort)

1. Add a strict run manifest (data hash, row counts, feature schema, model version).
2. Enforce output row-count checks against active test set.
3. Export a machine-readable training report (JSON) after each run.

### 11.2 Short-term (medium effort)

1. Align training and serving ensemble logic through shared module.
2. Add automated regression tests for numeric invariants (feature vector dimensionality, log-price conversion, and ensemble calculation consistency).
3. Add model cards per artifact with CV scores, data slice info, and train date.

### 11.3 Mid-term (high impact)

1. Strengthen semantic feature stack (improved embedding usage or hybrid encoders).
2. Improve image pipeline throughput and missing-image strategy.
3. Add robust hyperparameter optimization with reproducible seed strategy.
4. Implement drift detection and scheduled retraining criteria.

---

## 12. Suggested Report Template for Future Iterations

When preparing future formal reports, ensure these sections always exist:

1. Executive Summary
2. Business Problem and Success Criteria
3. Data Description and Data Quality
4. Methodology and Feature Engineering
5. Model Development and Evaluation
6. Error Analysis and Failure Modes
7. Deployment Architecture and SLOs
8. Governance, Compliance, and Risk
9. Roadmap and Next Milestones
10. Appendix (commands, configs, reproducibility notes)

---

## 13. Conclusion

The Product Price Predictor project is a mature, practical ML system with strong foundations in feature engineering, ensemble modeling, and production deployment. The codebase already reflects critical ML correctness improvements and operational safeguards. To move from a strong project to a highly reliable production-grade benchmark, the highest priority is consistency across data versions, training-serving parity, and traceable experiment documentation.

With these controls in place, the system can scale confidently for both competition submissions and real-world service usage.

---

## 14. Appendix: Useful Commands

Environment and run:

```bash
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Training examples:

```bash
python train_models.py --bert
python train_models.py --sample 5000
python train_models.py --images --sample 1000
```

Validation:

```bash
python validate_fixes.py
python run_tests.py --fast
bash validate_solution.sh
```

Deployment (example start command):

```bash
gunicorn -w 1 -b 0.0.0.0:$PORT --timeout 180 app:app
```