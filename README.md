# MediMind AI — Modular Medical AI Assistant

A scalable, plugin-style medical AI assistant designed to grow. Every capability (diagnosis from input, prescription OCR, medical image analysis, chat) is a self-contained module that can be added, removed, or replaced without touching the rest of the system.

> ⚠️ **Disclaimer**: This is an educational/assistance tool. It does NOT replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

## Why this design

The project brief explicitly calls out future extensibility. So the core architectural decision is: **every medical capability is a Module**. Modules implement a common interface, register themselves at startup, and expose REST endpoints automatically. Adding a new capability (say, dermatology image analysis) means dropping a new folder in `backend/modules/` — nothing else changes.

## Core capabilities (Phase 1)

1. **Manual-input diagnosis** — risk prediction for diabetes, hypertension, heart disease from user-entered symptoms/vitals (scikit-learn models).
2. **Prescription OCR** — extract structured info (medicines, dosages, doctor, patient) from uploaded prescription images (Tesseract + spaCy NER).
3. **Medical imaging** — classify MRI / CT / chest X-ray / USG images using transfer learning (PyTorch + pre-trained CNNs from torchvision and HuggingFace).
4. **Medical chat assistant** — RAG-based Q&A grounded in trusted sources (free LLM via HuggingFace Inference API or local Ollama).

## Tech stack (all permanently free)

| Layer | Choice | Why |
|---|---|---|
| Backend | **FastAPI** | Async, auto OpenAPI docs, modular routers |
| ML | **scikit-learn, PyTorch, MONAI** | Open source, vast pretrained model ecosystem |
| OCR | **Tesseract + EasyOCR** | Free, offline, no API limits |
| Frontend | **Streamlit** | Fast to build, easy to host free |
| DB | **SQLite** (dev) → **Supabase free tier** (prod) | Zero setup → 500MB free DB |
| Training compute | **Google Colab** | Free T4 GPU, ~12hr sessions |
| Model hosting | **Hugging Face Hub** | Unlimited public models, free |
| LLM | **HuggingFace Inference API** + **Ollama** (local fallback) | Free tier + offline option |
| Deployment | **Hugging Face Spaces** / **Render free tier** | Permanently free public hosting |
| Containerization | **Docker** | Reproducible everywhere |

## Hardware fit

Built for your machine (i5-12400, 16GB RAM, no dGPU):
- **Local development & inference** runs comfortably (FastAPI + scikit-learn + small CNNs in ONNX run fine on CPU).
- **Training happens on Google Colab** — notebooks in `ml_training/` push trained models back to HuggingFace Hub for inference.
- **Large LLMs** are NOT run locally. Use HF Inference API for cloud LLM calls, or Ollama with small models (Phi-3 mini, Llama 3.2 3B) if you need offline.

## Python version

**Use Python 3.12.** As of May 2026, Python 3.14 still lacks wheels for several libraries we need in Phase 2-3 (PyTorch CUDA, MONAI, EasyOCR's full stack). Python 3.12 has the broadest, most stable ecosystem support and nothing we use benefits from 3.14-specific features.

If you have multiple Pythons installed (use `py -0` to check), pin 3.12 explicitly when creating the virtual environment.

## Quick start (Windows PowerShell)

```powershell
# 1. Clone and enter
git clone <your-repo> medical-ai-assistant
cd medical-ai-assistant

# 2. Create virtual env with Python 3.12 specifically
py -3.12 -m venv venv

# 3. Activate it (PowerShell)
.\venv\Scripts\Activate.ps1

# If you get "running scripts is disabled on this system", run this ONCE
# in the current PowerShell window, then retry activation:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Verify you're on 3.12
python --version    # should show Python 3.12.x

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Copy env file (optional for Phase 1 — diabetes model needs no API keys)
Copy-Item .env.example .env

# 6. Train the starter model (~30 seconds, downloads Pima Indians dataset)
python scripts\train_diabetes_model.py

# 7. Run backend (this PowerShell window)
uvicorn backend.main:app --reload --port 8000

# 8. Open a SECOND PowerShell tab/window for the frontend:
cd C:\Users\User\Desktop\portfolio_projects\ML_projects\medical-ai-assistant
.\venv\Scripts\Activate.ps1
streamlit run frontend\app.py

# 9. Browser:
#    Frontend: http://localhost:8501
#    API docs: http://localhost:8000/docs
```

### Linux / macOS users

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/train_diabetes_model.py
uvicorn backend.main:app --reload --port 8000   # terminal 1
streamlit run frontend/app.py                   # terminal 2
```

## Project layout

```
medical-ai-assistant/
├── backend/                    # FastAPI server
│   ├── main.py                 # App entry, auto-registers modules
│   ├── config.py               # Pydantic settings (.env loader)
│   ├── database.py             # SQLAlchemy engine + session
│   ├── core/                   # Cross-cutting concerns
│   │   ├── base_module.py      # Abstract base — every module inherits this
│   │   ├── exceptions.py
│   │   └── registry.py         # Module discovery & auto-registration
│   └── modules/                # ← Add new capabilities HERE
│       ├── manual_diagnosis/   # Diabetes / hypertension / heart risk
│       ├── prescription_ocr/   # Prescription image → structured data
│       ├── medical_imaging/    # X-ray / MRI / CT classification
│       └── chat_assistant/     # Medical Q&A
├── frontend/                   # Streamlit UI
├── ml_training/                # Colab notebooks (push models to HF)
├── data/                       # Datasets and trained models (gitignored)
├── scripts/                    # Setup, training, utility scripts
├── docs/                       # Detailed module docs
├── ARCHITECTURE.md             # Design rationale
└── ROADMAP.md                  # Phased delivery plan
```

## Adding a new module

That's the whole point. See [docs/adding_a_module.md](docs/adding_a_module.md) for the step-by-step. TL;DR:

1. `mkdir backend/modules/your_module`
2. Create `router.py` (FastAPI router) and `service.py` (logic) inheriting from `BaseModule`.
3. Restart server. The module is auto-discovered and its endpoints appear at `/api/v1/your_module/*`.

## Read next

- [ARCHITECTURE.md](ARCHITECTURE.md) — why everything is built this way
- [ROADMAP.md](ROADMAP.md) — what ships when
- [docs/free_services_setup.md](docs/free_services_setup.md) — getting free API keys
