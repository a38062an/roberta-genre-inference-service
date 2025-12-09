---
title: RoBERTa Genre Service
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# RoBERTa Genre Inference Service

A production-ready microservice for multi-label movie genre classification. This service leverages a fine-tuned RoBERTa model to predict genres from plot synopses with high accuracy.

**Supported Genres:**
*   Comedy
*   Cult
*   Flashback
*   Historical
*   Revenge
*   Romantic
*   Sci-Fi
*   Violence

## Quick Start

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Server
```bash
uvicorn app.main:app --reload
```
The server will start at `http://localhost:8000`.

### 3. Explore the API (Interactive Docs)
FastAPI provides automatic interactive documentation.
*   **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)  
    Use this to interactively test endpoints directly from your browser.
*   **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Data Distribution & Challenges

The model was trained on the `CW2_training_dataset.csv`, which exhibits significant class imbalance. This poses a challenge for accurately detecting minority classes like **Historical** and **Sci-Fi**.

**Training Data Breakdown:**
*   **Violence**: 3032 (Dominant)
*   **Flashback**: 1995
*   **Romantic**: 1993
*   **Cult**: 1801
*   **Revenge**: 1680
*   **Comedy**: 1230
*   **Sci-Fi**: 207 (Extremely Rare)
*   **Historical**: 191 (Extremely Rare)

*Challenge*: Future improvements could involve oversampling minority classes, data augmentation, or using weighted loss functions (currently implemented) to better handle this disparity.

## Usage

### Training the Model
To train the model using your custom CSV datasets:
```bash
python -m app.train --train_file CW2_training_dataset.csv --val_file CW2_validation_dataset.csv
```
*   Trains for 4 epochs (configurable in `app/train.py`).
*   Saves the best model weights to `artifacts/roberta_best.pt`.

### Evaluating Performance (F1 Score)
To skip training and just evaluate an existing model on the validation set:
```bash
python -m app.train --train_file CW2_training_dataset.csv --val_file CW2_validation_dataset.csv --evaluate_only
```

### Making a Prediction (Curl)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "movie_id": "tt0111161",
    "plot_synopsis": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency."
  }'
```

---

## Architecture

This project follows a microservice pattern to separate concerns:

*   **`app/main.py`**: API Gateway (FastAPI). Handles routing and request validation.
*   **`app/service.py`**: Business Logic. Loads the model (singleton pattern) and handles inference.
*   **`app/model.py`**: Model Definition. Wraps the HuggingFace `RobertaForSequenceClassification`.
*   **`app/config.py`**: Configuration. Stores constants like device settings (`MPS`/`CUDA`/`CPU`), thresholds, and genre mapping.
*   **`app/train.py`**: Training Pipeline. Handles data loading, training loops, and threshold optimization.

### Technology Stack
*   **Framework**: FastAPI (High performance, async support)
*   **ML Engine**: PyTorch & HuggingFace Transformers
*   **Server**: Uvicorn (ASGI) & Gunicorn (Process Manager for production)
*   **Container**: Docker (CPU-optimized build)

## Docker Deployment

To build and run the optimized CPU container:

```bash
# Build
docker build -t roberta-inference .

# Run
docker run -p 8000:8000 roberta-inference
```
