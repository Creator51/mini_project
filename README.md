# Real-Time Cyber Threat Detection Framework Using Machine Learning

An end-to-end, high-performance cybersecurity Machine Learning framework designed for real-time network flow threat detection. Built with Python 3.12, scikit-learn, LightGBM, and FastAPI.

---

## 📌 Project Overview

This project implements a reproducible machine learning framework that inspects incoming network traffic flows, cleans feature representations without data leakage, and provides low-latency threat detection using Random Forest, LightGBM, and a hybrid soft-voting ensemble.

### Key Highlights
- **Zero Data Leakage Protocol**: Stratified train/test splits performed before fitting any scaler, imputer, or encoder.
- **Identified & Dropped Leakage Features**: Timestamps (`frame.time`), IP addresses (`ip.src_host`, `ip.dst_host`), and raw hex payload strings (`tcp.payload`, `tcp.options`) removed with cybersecurity justifications.
- **Empirical High Detection Performance**: 100% Binary Detection Recall and Accuracy on Edge-IIoTset test set.
- **Low-Latency Streaming Inference**: End-to-end single event latency under 51 ms; vectorized batch throughput scaling to **24,394+ events/second**.
- **Production FastAPI Service**: Pre-loaded model microservice exposing `/health`, `/model-info`, `/predict`, and `/predict/batch`.

---

## 🏗 Project Architecture

```
mini project/
├── data/
│   ├── raw/
│   └── processed/             # Cleaned & transformed joblib splits
├── src/
│   ├── data/                  # Loader, cleaner, preprocessing pipeline
│   ├── features/              # Feature selection experiments
│   ├── models/                # Random Forest, LightGBM, Hybrid Ensemble
│   ├── evaluation/            # EDA & comparative model visualization
│   └── realtime/              # Streaming inference & benchmarking engine
├── api/                       # FastAPI microservice routes & schemas
├── models/                    # Exported .joblib ML binaries
├── results/
│   ├── figures/               # Publication-quality PNG plots
│   └── metrics/               # Empirical JSON result logs
├── tests/                     # Pytest suite for API and inference
├── config.yaml
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone/Navigate to Project Workspace**:
   ```powershell
   cd "e:\Anurag University B.Tech\mini project"
   ```

2. **Install Python Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Verify Raw Dataset**:
   Ensure `ML-EdgeIIoT-dataset.csv` is placed in the project root directory or `data/raw/`.

---

## 🚀 Step-by-Step Execution Pipeline

Run the pipeline phases sequentially:

### Step 1: Inspect Raw Dataset (Phase 1)
```powershell
python src/data/inspect_dataset.py --file "ML-EdgeIIoT-dataset.csv"
```

### Step 2: Run Data Cleaning & Preprocessing (Phase 2)
```powershell
python src/data/run_preprocessing.py
```

### Step 3: Exploratory Data Analysis (Phase 3)
```powershell
python src/evaluation/eda.py
```

### Step 4: Train Random Forest Classifier (Phase 4)
```powershell
python src/models/random_forest.py
```

### Step 5: Train LightGBM Classifier (Phase 5)
```powershell
python src/models/lightgbm_model.py
```

### Step 6: Comparative Evaluation & Hybrid Ensemble (Phases 6 & 7)
```powershell
python src/evaluation/compare_models.py
```

### Step 7: Latency & Throughput Benchmarking (Phase 10)
```powershell
python src/realtime/benchmark.py
```

### Step 8: Feature Selection Experiment (Phase 11)
```powershell
python src/features/feature_selection_experiment.py
```

### Step 9: Run Automated Pytest Suite
```powershell
pytest tests/ -v
```

---

## 🌐 Running the FastAPI Microservice (Phase 9)

Start the Uvicorn ASGI server:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation will be available at: `http://localhost:8000/docs`

### Example API Request (Single Event)

```powershell
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{
    "features": {
      "arp.opcode": 0.0,
      "tcp.ack": 100.0,
      "tcp.srcport": "80",
      "http.request.method": "GET"
    }
  }'
```

### Example API Response

```json
{
  "prediction": "Attack",
  "is_attack": true,
  "confidence": 1.0,
  "threat_probability": 1.0,
  "random_forest_probability": 1.0,
  "lightgbm_probability": 1.0,
  "selected_model": "ensemble",
  "inference_latency_ms": 51.32
}
```

---

## 📊 Summary of Experimental Results

| Model | Accuracy | Recall (Detection Rate) | F1-Score | Training Time | Single-Sample Latency | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 100.00% | 100.00% | 1.0000 | 3.08 s | 0.0036 ms | 0.30 MB |
| **LightGBM** | 100.00% | 100.00% | 1.0000 | **0.49 s** | **0.0016 ms** | **0.12 MB** |
| **Hybrid Ensemble** | 100.00% | 100.00% | 1.0000 | 3.57 s | 0.0025 ms | 0.42 MB |

---

## 🔬 Reproducibility & Research Integrity

All random seeds are explicitly set to `42` (`config.yaml`). Every number, graph, table, and latency metric reported in this project is strictly derived from local empirical execution logs stored in `results/metrics/` and `results/figures/`.
