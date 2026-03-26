# 🔍 TrustLens — Deepfake Detection System

A production-grade deepfake detector using **EfficientNet-B4** + **FFT frequency analysis**, built with Flask, SQLAlchemy, and a premium forensic UI.

---

## Architecture

```
trustlens/
├── app.py                      # Flask API server
├── requirements.txt
├── .env.example                # Environment config template
├── model/
│   ├── deepfake_detector.py    # EfficientNet-B4 model + inference pipeline
│   └── train.py                # Fine-tuning script
├── database/
│   └── db.py                   # SQLAlchemy models + DB manager
├── templates/
│   └── index.html              # Premium forensic UI
└── checkpoints/                # Model weights (after training)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env

# 3. Run the server
python app.py

# 4. Open browser
open http://localhost:5000
```

---

## Model

The ML pipeline has **two branches**:

| Branch | Description |
|--------|-------------|
| **Spatial (EfficientNet-B4)** | Analyzes pixel-level facial artifacts using ImageNet pre-trained features |
| **Frequency (FFT)** | Detects GAN/diffusion artifacts in the frequency domain via 2D FFT |

Final prediction = `0.7 × spatial + 0.3 × frequency`

### Fine-tuning (recommended for best accuracy)

```bash
# Prepare dataset in this structure:
# data/train/real/  ← real face images
# data/train/fake/  ← deepfake images
# data/val/real/
# data/val/fake/

python model/train.py \
  --data_dir ./data \
  --epochs 20 \
  --batch_size 16 \
  --output_dir checkpoints/
```

Recommended datasets:
- **FaceForensics++** — https://github.com/ondyari/FaceForensics
- **Celeb-DF v2** — https://github.com/yuezunli/celeb-deepfakeforensics
- **DFDC** — https://www.kaggle.com/competitions/deepfake-detection-challenge

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Upload & analyze image/video |
| `GET` | `/api/history` | Recent analysis results |
| `GET` | `/api/stats` | Aggregate statistics |
| `GET` | `/api/health` | Health check |

---

## Database

Supports **SQLite**, **PostgreSQL**, and **MySQL** via SQLAlchemy.

Configure in `.env`:
```
DATABASE_URL=postgresql://user:pass@localhost:5432/trustlens
```

---

## Disclaimer

This is an educational/research project. For critical forensic verification, consult professional services.
