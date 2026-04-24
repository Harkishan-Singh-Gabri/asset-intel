# 🔍 Asset Intel — IT Asset Intelligence Engine

A production-grade MLOps system that lets warehouse workers **photograph hardware and instantly retrieve asset details** — detected class, confidence score, and CLIP similarity match — with zero manual entry.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.1.0-FF6B35)](https://ultralytics.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11-0194E2?logo=mlflow)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📸 Demo

![Asset Intel Scanner UI](assets/demo.png)

> Worker uploads a photo of an HP monitor with keyboard — system detects **Computer monitor** (97% YOLO confidence, 0.849 CLIP score) and **Computer keyboard** in one scan.

---

## 🧠 Why This Project?

Large IT firms manage millions of physical assets — laptops, servers, routers. Manual cataloging costs thousands of man-hours and introduces audit risk. Asset Intel solves this by combining computer vision with vector search:

- A warehouse worker **photographs any hardware** — the system identifies it instantly
- **YOLOv8 detects and localizes** multiple items in a single photo — one photo, many assets
- **CLIP semantic embeddings** match items even if the exact model was never seen during training
- Every scan is **logged to a database** — full audit trail for insurance and compliance
- **Evidently monitors drift** — automatically triggers retraining when new hardware types appear

---

## 🚀 What's Working

### 📷 Asset Scanner UI
- Upload a photo or use camera input directly in browser
- Instantly see detected hardware class
- YOLO confidence score + CLIP similarity score displayed
- Auto-flagged if confidence is below threshold
- Scan history viewable in sidebar

### ⚡ FastAPI Backend
- `POST /scan` — full two-stage pipeline in one API call
- `GET /scans` — complete audit log of all scans
- Redis caching — repeated scans served instantly without rerunning pipeline
- SQLite database for scan records (PostgreSQL-ready for production)
- Graceful fallback to in-memory cache if Redis unavailable

### 🎯 Two-Stage Vision Pipeline
- **Stage 1 — YOLOv8m**: detects and localizes all hardware items in photo
- **Stage 2 — CLIP + FAISS**: encodes each crop to 512-dim vector, finds nearest catalog match
- Handles multiple hardware items in a single photo

### 📊 MLOps Pipeline
- **DVC**: every dataset version tracked with full reproducibility
- **MLflow**: all training runs logged — metrics, parameters, artifacts
- **Evidently**: monitors CLIP score drift and flag rate in production
- **Prefect**: automated retraining pipeline — triggered when drift is detected

---

## ⚠️ Improvements

- Replace Open Images demo catalog with real company assets (serial numbers, specs, locations)
- Add OCR for serial number extraction from detected crops
- Add EfficientNet-B3 as Stage 3 re-ranker for fine-grained model classification
- Migrate to PostgreSQL + pgvector in production
- Persistent scan image storage with MinIO/S3
- BentoML for production model serving with auto-batching
- Fix DVC remote push on Windows (encoding issue with binary files)
- CI/CD pipeline with automated model performance gates
- Mobile-optimized UI for warehouse workers

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | FiftyOne, Open Images V7, Albumentations, DVC |
| Detection | YOLOv8m — mAP50: 0.510 |
| Embeddings | CLIP ViT-B-32 (open-clip-torch) |
| Vector Search | FAISS IndexFlatIP, ChromaDB |
| MLOps | MLflow, Evidently, Prefect, DVC |
| Backend | FastAPI, SQLAlchemy, Redis, SQLite |
| UI | Streamlit |
| DevOps | Docker, Docker Compose |

---

## 🏗️ Project Structure

```
ASSET-INTEL/
├── assets/
│   └── demo.png                # Demo screenshot
├── data/
│   ├── raw/                    # DVC tracked — Open Images V7 (1000 images)
│   ├── processed/              # DVC tracked — augmented warehouse images
│   ├── yolo/                   # DVC tracked — YOLO format dataset
│   └── embeddings/
│       ├── assets.faiss        # FAISS vector index
│       └── chroma/             # ChromaDB metadata store
├── models/
│   └── yolo/v1/weights/
│       └── best.pt             # DVC tracked — trained YOLOv8m weights
├── src/
│   ├── data/
│   │   ├── download.py         # FiftyOne Open Images V7 download
│   │   ├── augment.py          # Warehouse condition augmentation
│   │   └── prepare_yolo.py     # COCO → YOLO format + category filtering
│   ├── models/
│   │   ├── train_yolo.py       # YOLOv8 training + MLflow experiment logging
│   │   └── build_index.py      # CLIP encoder + FAISS + ChromaDB index builder
│   ├── inference/
│   │   ├── pipeline.py         # YOLO → CLIP → FAISS orchestration
│   │   └── vector_search.py    # FAISS + ChromaDB search logic
│   ├── api/
│   │   ├── main.py             # FastAPI endpoints + lifespan handler
│   │   ├── db.py               # SQLAlchemy ORM models
│   │   └── cache.py            # Redis caching with in-memory fallback
│   └── UI/
│       └── app.py              # Streamlit warehouse worker interface
├── mlops/
│   ├── drift_detection.py      # Evidently CLIP score + flag rate monitoring
│   ├── retrain_flow.py         # Prefect automated retraining DAG
│   └── reports/                # JSON drift detection reports
├── docker/
│   ├── docker-compose.yml      # Full stack: API, UI, PostgreSQL, Redis, MLflow
│   └── api.Dockerfile
├── dvc.yaml                    # DVC pipeline stages
├── .env.example                # Environment variable template
└── requirements.txt
```

---

## 📋 Prerequisites

- **Python** 3.11
- **Conda** or venv
- **Git**
- **NVIDIA GPU** (recommended — RTX 30/40/50 series)
- **Redis** (optional — falls back to in-memory cache if unavailable)

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Harkishan-Singh-Gabri/ASSET-INTEL.git
cd ASSET-INTEL
```

### 2. Create Environment
```bash
conda create -n asset-intel python=3.11 -y
conda activate asset-intel
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Set Up Environment
```bash
cp .env.example .env
# Edit .env with your values
```

### 5. Download & Prepare Data
```bash
python src/data/download.py       # downloads 1000 hardware images (~20 mins)
python src/data/augment.py        # augments for warehouse conditions
python src/data/prepare_yolo.py   # converts to YOLO format
```

### 6. Train Model & Build Index
```bash
python src/models/train_yolo.py   # ~15 mins on GPU
python src/models/build_index.py  # builds FAISS + ChromaDB index
```

### 7. Start the System
```bash
# Terminal 1 — FastAPI backend
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — Streamlit UI
streamlit run src/UI/app.py

# Terminal 3 — MLflow UI (optional)
mlflow ui --host 127.0.0.1 --port 5000
```

Open **http://localhost:8501** in your browser.

---

## 🔄 Data Flow

```
📱 Worker uploads photo
        ↓
⚡ FastAPI POST /scan
        ↓
Redis cache check → hit? return instantly
        ↓ miss
🎯 YOLOv8m — detect & localize all hardware items
        ↓ bounding box crops
🧠 CLIP ViT-B-32 — encode each crop → 512-dim vector
        ↓
🔍 FAISS IndexFlatIP — find top-5 nearest vectors
        ↓
💾 ChromaDB — retrieve metadata for matches
        ↓
🗄 SQLite — write audit record
        ↓
📱 Streamlit — display results to worker
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root health check |
| GET | `/health` | API status |
| POST | `/scan` | Scan hardware image — full pipeline |
| GET | `/scans` | Audit log of all scans |

### Example `/scan` Request
```python
import requests

with open("hardware_photo.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/scan",
        files={"file": f},
        params={"worker_id": "worker-01"}
    )
    print(response.json())
```

### Example Response
```json
{
  "status": "ok",
  "worker_id": "worker-01",
  "items_found": 2,
  "results": [
    {
      "detected_class": "Computer monitor",
      "yolo_confidence": 0.97,
      "top_match": {
        "metadata": {
          "category": "Television",
          "asset_id": "241",
          "image_file": "00ea0267e19c63cd.jpg"
        },
        "score": 0.849
      },
      "alternatives": [],
      "bbox": [230, 147, 649, 562]
    }
  ]
}
```

---

## 📊 Model Performance

| Model | Dataset | Epochs | mAP50 | mAP50-95 |
|-------|---------|--------|-------|----------|
| YOLOv8m | 1000 images | 50 | 0.510 | 0.285 |

**Detected Categories:**

| Category | Open Images Label |
|----------|------------------|
| Laptop | Laptop |
| Monitor | Computer monitor |
| Keyboard | Computer keyboard |
| Printer | Printer |
| Phone | Telephone |
| TV | Television |

---

## 🔧 MLOps

### Experiment Tracking
```bash
mlflow ui --host 127.0.0.1 --port 5000
```
Every training run automatically logs: mAP50, mAP50-95, Precision, Recall, model weights.

### Drift Detection
```bash
python mlops/drift_detection.py
```
Raises alert when:
- Average CLIP score drops below `0.65` — model struggling with new hardware
- Flag rate exceeds `40%` — too many low-confidence matches

### Automated Retraining
```bash
# Manual trigger
python -m mlops.retrain_flow

# Force retrain regardless of drift status
python -m mlops.retrain_flow --force
```

Pipeline stages: download → augment → prepare → train → rebuild index

### Data Versioning
```bash
dvc repro      # reproduce full pipeline from scratch
dvc commit     # save current data state
dvc push       # push to remote storage
dvc pull       # pull data on new machine
```

---

## 🐳 Docker (Production)

```bash
cd docker
docker compose up -d
```

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI backend |
| `ui` | 8501 | Streamlit UI |
| `postgres` | 5432 | PostgreSQL + pgvector |
| `redis` | 6379 | Redis cache |
| `mlflow` | 5000 | MLflow tracking server |

Switch to PostgreSQL by updating `.env`:
```dotenv
DATABASE_URL=postgresql://user:pass@postgres:5432/assetintel
```

---

## 🧪 Running the Full Pipeline

```bash
# Full data pipeline
python src/data/download.py
python src/data/augment.py
python src/data/prepare_yolo.py

# Training
python src/models/train_yolo.py
python src/models/build_index.py

# Drift check
python mlops/drift_detection.py

# Retraining (if drift detected)
python -m mlops.retrain_flow
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

### Guidelines
- Run the full pipeline before submitting a PR
- Ensure mAP50 does not drop below 0.45 after changes
- Test both Redis available and unavailable states for cache

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 About

**Harkishan Singh Gabri**

> Built to demonstrate how computer vision and vector search can eliminate manual IT asset auditing — proving Solution Architect level thinking by combining MLOps, CV, and backend engineering into one deployable system.

### Connect
<<<<<<< HEAD
- 🌐 **GitHub**: [@Harkishan-Singh-Gabri](https://github.com/Harkishan-Singh-Gabri)
- 💼 **LinkedIn**: [Harkishan Singh Gabri](https://www.linkedin.com/in/harkishan-singh-gabri/)
=======
- 🌐 **GitHub**: Harkishan-Singh-Gabri(https://github.com/Harkishan-Singh-Gabri)
- 💼 **LinkedIn**: Harkishan Singh Gabri(https://www.linkedin.com/in/harkishan-singh-gabri/)
>>>>>>> 8014f5177e2830776187ca2b1079b41b3393a686

---

## 🙏 Acknowledgements

- **Ultralytics** — YOLOv8 object detection framework
- **OpenAI / LAION** — CLIP model and open-clip-torch implementation
- **Facebook Research** — FAISS vector similarity search library
- **Open Images V7** — Hardware image dataset used for training and demo catalog
- **Evidently AI** — Model monitoring and drift detection framework
- **Prefect** — Modern workflow orchestration

---

## 📊 Project Status

🟢 **Active Development**

- **Version**: 1.0.0
- **Model**: YOLOv8m — mAP50: 0.510
- **Pipeline**: Fully operational locally
- **Deployment**: Docker config ready

