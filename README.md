# Product Price Predictor

Predict product prices from catalog descriptions using an ensemble of machine learning models and sentence embeddings.

This repository currently contains two practical workflows:
- Production inference app (`app.py`, `predict.py`, `ui/index.html`)
- Full challenge pipeline (`src/main.py`, `run_all.sh`, `src/*` modules)

## What Is In This Repo

- Flask API + web UI for real-time prediction
- CLI predictor for quick local inference
- Standalone training script with optional BERT and image features
- Modular training/prediction pipeline under `src/`
- Test suite, validation scripts, and deployment configs

## Current Project Structure

```text
student_resource/
|- app.py
|- predict.py
|- train_models.py
|- run_all.sh
|- run_tests.py
|- validate_fixes.py
|- validate_solution.sh
|- Procfile
|- render.yaml
|- requirements.txt
|- requirements_model_training.txt
|- runtime.txt
|- dataset/
|- models/
|- src/
|- tests/
|- ui/
`- README.md
```

## Quick Start (Inference App)

1. Activate environment:

```bash
source venv/bin/activate
```

2. Install runtime dependencies:

```bash
pip install -r requirements.txt
```

3. Optional sanity check:

```bash
python validate_fixes.py
```

4. Start the web app:

```bash
python app.py
```

5. Open:

```text
http://localhost:5050
```

Note: on first run, sentence-transformers downloads `all-MiniLM-L6-v2` (~90 MB).

## CLI Usage

Interactive mode:

```bash
python predict.py
```

Single prediction:

```bash
python predict.py --text "Item Name: Nescafe Classic Instant Coffee, 200g
Value: 200.0
Unit: Grams"
```

## API Endpoints

- `GET /` -> serves `ui/index.html`
- `GET /health` -> app/model health status
- `POST /predict` -> returns ensemble and per-model predictions

Example request:

```bash
curl -X POST http://localhost:5050/predict \
  -H "Content-Type: application/json" \
  -d '{"description":"Item Name: Coffee\nValue: 200\nUnit: Grams"}'
```

## Training Workflows

### 1) Standalone Script (`train_models.py`)

Install extended training dependencies if needed:

```bash
pip install -r requirements_model_training.txt
```

Examples:

```bash
# Fast smoke test
python train_models.py --sample 5000

# App-compatible retraining (16 text + 384 BERT = 400 features)
python train_models.py --bert

# Optional image-augmented training (slower, usually GPU)
python basic_image_downloader.py
python train_models.py --bert --images
```

Important: `app.py` and `predict.py` expect 400-feature models (text + BERT). If you train without `--bert` or with a different feature shape, web/CLI inference may fail feature checks.

### 2) Modular Pipeline (`src/main.py`)

```bash
python src/main.py --mode train
python src/main.py --mode full
python src/main.py --mode predict --model-path models/<your_model>.pkl
```

With config file:

```bash
python src/main.py --mode full --config config.json
```

### 3) One-Command Pipeline Script (`run_all.sh`)

```bash
bash run_all.sh full
bash run_all.sh train
bash run_all.sh predict
bash run_all.sh --validate-only
```

## Testing And Validation

Run full test suite:

```bash
python run_tests.py
```

Run faster subset:

```bash
python run_tests.py --fast
```

Submission/compliance checks:

```bash
bash validate_solution.sh
```

Inference artifact check:

```bash
python validate_fixes.py
```

## Deployment

This repo is configured for Render/Gunicorn deployment.

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn -w 1 -b 0.0.0.0:$PORT --timeout 180 app:app`

Deployment files:
- `render.yaml`
- `Procfile`
- `runtime.txt`

Detailed guides:
- `DEPLOYMENT_GUIDE.md`
- `RENDER_DEPLOYMENT.md`

## Key Artifacts

- Models: `models/*_model.pkl`
- Scaler: `scaler.pkl`
- Output predictions: `test_out.csv`

## Notes

- The app loads available models from a fixed list and skips models that fail to deserialize.
- `SENTENCE_TRANSFORMERS_HOME`, `TRANSFORMERS_CACHE`, and `HF_HOME` are set to `/tmp/...` in `app.py` to support read-only deployment filesystems.
- For reproducible environment setup, `environment.yml` is included in the repository.