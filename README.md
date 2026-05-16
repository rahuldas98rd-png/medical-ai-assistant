# MediMind AI — Modular Medical AI Assistant

![MediMind AI hero](app_demo/app_hero.png)

A production-grade, plugin-style medical AI assistant. Every capability is a self-contained module that registers itself at startup — add a new one by dropping a folder in `backend/modules/`, nothing else changes.

> ⚠️ **Disclaimer**: Educational tool only. Does NOT replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## Screenshots

### Home — Module Dashboard
![Home page showing all 7 modules healthy](app_demo/app_preview_1.png)

### Diabetes Risk Estimator
![Diabetes risk result — 30.4% moderate risk with gauge chart and feature importances](app_demo/app_preview_2.png)

### Hypertension Risk Estimator
![Hypertension risk result — 18.4% low risk](app_demo/app_preview_6.png)

### Heart Disease Risk Estimator
![Heart disease risk result — 1.4% low risk, Cleveland dataset](app_demo/app_preview_7.png)

### Liver Disease Risk Estimator
![Liver disease risk result — 73% HIGH risk from LFT values](app_demo/app_preview_8.png)

### Prescription OCR
![Prescription image uploaded with extracted medicines table](app_demo/app_preview_3.png)

### Chest X-Ray — Classification
![Chest X-ray pathology probabilities list](app_demo/app_preview_4.png)

### Chest X-Ray — Grad-CAM Heatmaps
![Three Grad-CAM heatmap overlays with bar chart](app_demo/app_preview_5.png)

### Brain MRI Tumor Classifier
![Brain MRI — Meningioma prediction](app_demo/app_preview_9.png)

![Brain MRI — all class probabilities bar chart](app_demo/app_preview_10.png)

### Medical Chat Assistant (RAG)
![RAG response about heart disease prevention](app_demo/app_preview_11.png)

![Source citations panel — 3 knowledge-base articles with relevance scores](app_demo/app_preview_12.png)

### Smart Analysis — Multi-Modal Orchestrator
![Smart Analysis input form with symptoms, measurements, and brain MRI uploaded](app_demo/app_preview_13.png)

![Smart Analysis results — Hypertension screened, Chest X-Ray Infiltration 61.1%](app_demo/app_preview_14.png)

![Smart Analysis results continued — Heart Disease and Liver Disease flagged, key recommendations, previous consultations](app_demo/app_preview_15.png)

---

## What it does

| Module | Capability |
|---|---|
| **Diabetes Risk** | Gradient-boosted risk score from glucose, BMI, age, etc. |
| **Hypertension Risk** | Cardiovascular risk from BP, cholesterol, lifestyle factors |
| **Heart Disease Risk** | Coronary artery disease risk (Cleveland dataset, XGBoost, ROC-AUC 0.94) |
| **Liver Disease Risk** | Hepatic risk from liver function test values |
| **Prescription OCR** | Extract medicines + dosages from prescription photos (Tesseract + fuzzy match) |
| **Chest X-Ray** | 18-class pathology classification (DenseNet-121 pretrained) + Grad-CAM heatmap |
| **Brain MRI** | Glioma / meningioma / pituitary / no-tumor (ResNet-50 fine-tuned on Colab) |
| **Medical Chat** | RAG Q&A grounded in MedlinePlus / WHO / CDC articles (Llama-3 or Ollama) |
| **Smart Analysis** | Paste symptoms → auto-routed to all relevant modules → unified report |

---

## Tech stack (all permanently free)

| Layer | Choice |
|---|---|
| Backend | FastAPI + SQLAlchemy + SQLite (swap to Supabase for prod) |
| ML | scikit-learn, XGBoost, PyTorch (CPU inference), torchxrayvision |
| OCR | Tesseract 5 + pypdfium2 (PDF support) |
| RAG | ChromaDB + sentence-transformers/all-MiniLM-L6-v2 |
| LLM | HuggingFace Inference API (Llama-3-8B) + Ollama phi3:mini fallback |
| Frontend | Streamlit |
| Training | Google Colab (free T4 GPU) → models pushed to HuggingFace Hub |
| Deployment | Docker + Render.com (`render.yaml` included) |

---

## Quick start

### Option A — Docker (recommended)

```bash
# Runs backend + frontend in one command
docker compose up --build

# Frontend: http://localhost:8501
# API docs: http://localhost:8000/docs
```

### Option B — Local (Windows PowerShell)

```powershell
# 1. Create venv with Python 3.12
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PyTorch CPU (separate step — special index URL)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Configure environment
Copy-Item .env.example .env   # then edit .env — see Configuration section

# 5. Train tabular models (~2 min total)
python scripts/train_diabetes_model.py
python scripts/train_hypertension_model.py
python scripts/train_heart_disease_model.py
python scripts/train_liver_disease_model.py

# 6. Populate the chat knowledge base
python scripts/ingest_knowledge_base.py

# 7. Start backend (terminal 1)
uvicorn backend.main:app --reload --port 8000

# 8. Start frontend (terminal 2)
streamlit run frontend/app.py
```

### Option C — Linux / macOS

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
cp .env.example .env
python scripts/train_diabetes_model.py   # repeat for hypertension/heart/liver
python scripts/ingest_knowledge_base.py
uvicorn backend.main:app --reload &
streamlit run frontend/app.py
```

---

## Configuration (`.env`)

```env
# Required for the chat assistant (free tier):
HUGGINGFACE_TOKEN=hf_...

# Optional — enable API key auth on all POST endpoints (leave unset for dev):
API_KEY=your-secret-key

# Switch to Supabase for production (code needs no changes):
# DATABASE_URL=postgresql://postgres:[password]@db.[project].supabase.co:5432/postgres
```

For the Brain MRI classifier, the model is downloaded automatically from HuggingFace Hub on first run (requires `HUGGINGFACE_TOKEN`). Alternatively, place `brain_mri_resnet50_v0.1.0.pt` in `data/models/` manually.

For the chat assistant fallback (no HF token), install Ollama and run:
```bash
ollama pull phi3:mini && ollama serve
```

---

## Project layout

```
medical-ai-assistant/
├── backend/
│   ├── main.py                  # FastAPI app — auto-discovers all modules
│   ├── config.py                # Pydantic settings (.env loader)
│   ├── database.py              # SQLAlchemy models (PredictionLog, ConsultationHistory)
│   ├── core/
│   │   ├── base_module.py       # BaseModule interface every module implements
│   │   ├── registry.py          # Auto-discovery at startup
│   │   ├── auth.py              # X-API-Key middleware
│   │   ├── audit_log.py         # Tamper-evident SHA-256 chain logger
│   │   └── rate_limiter.py      # slowapi limiter singleton
│   └── modules/
│       ├── manual_diagnosis/    # Diabetes, hypertension, heart disease, liver disease
│       ├── prescription_ocr/    # Tesseract OCR + fuzzy medicine matching
│       ├── medical_imaging/     # Chest X-ray (DenseNet-121 + Grad-CAM)
│       ├── brain_mri/           # Brain MRI (ResNet-50, 4-class)
│       ├── chat_assistant/      # RAG pipeline (ChromaDB + LLM)
│       ├── orchestrator/        # Smart Analysis — routes symptoms to all modules
│       └── audit/               # Audit chain verification endpoint
├── frontend/
│   ├── app.py                   # Streamlit entry point
│   └── pages/                   # One page per module + Smart Analysis
├── backend/tests/               # 134 unit tests (pytest)
├── ml_training/                 # Colab notebooks (Brain MRI training)
├── scripts/                     # train_*.py, ingest_knowledge_base.py
├── data/                        # models/ + db (gitignored)
├── Dockerfile                   # Backend (multi-stage, includes Tesseract)
├── Dockerfile.frontend          # Frontend (lightweight Streamlit image)
├── docker-compose.yml           # Full stack — one command
└── render.yaml                  # Render.com deploy config
```

---

## Security & audit

- **API key auth** — set `API_KEY` in `.env`; every POST requires `X-API-Key` header. Unset = open dev mode.
- **No PII stored** — raw inputs are never persisted; only SHA-256 hashes.
- **Tamper-evident log** — every prediction row carries a `chain_hash = SHA256(prev_hash | module | input_hash)`. Verify integrity at:
  ```
  GET /api/v1/audit/verify
  ```
- **Rate limiting** — Smart Analysis endpoint: 15 requests/minute per IP.
- **Safety blocklist** — chat assistant blocks crisis/self-harm keywords and redirects to emergency services.

---

## Running tests

```powershell
.\.venv\Scripts\Activate.ps1
pytest backend/tests -v -m "not integration"   # 134 unit tests, no models needed
pytest backend/tests -v                         # includes integration tests (models required)
```

---

## Deploy to Render

1. Push to GitHub
2. Render → New → Web Service → connect repo
3. Render auto-detects `render.yaml`
4. Set `HUGGINGFACE_TOKEN` and `API_KEY` in the Render environment dashboard

---

## Adding a new module

1. `mkdir backend/modules/your_module`
2. Implement `BaseModule` subclass in `__init__.py` with `name`, `get_router()`, `on_startup()`
3. Restart — auto-discovered, endpoints live at `/api/v1/your_module/*`

See [ARCHITECTURE.md](ARCHITECTURE.md) and [ROADMAP.md](ROADMAP.md) for the full design rationale and delivery plan.
