# Architecture

## Design philosophy

Three forces shape every decision:

1. **Extensibility** — the brief says "in future new features can be added." So *the unit of growth is the module*. New medical capabilities should be droppable folders, not refactors.
2. **Permanently free** — rules out paid LLM APIs (OpenAI, Anthropic, paid Gemini), paid cloud GPU, managed model hosting with usage costs. We rely on open-source models, free tiers with no expiry, and free training compute.
3. **Runs on a mid-range Windows machine** (i5-12400, 16GB RAM, no dGPU) — we keep inference CPU-friendly. Training is offloaded to Google Colab.

## The Module pattern

Every capability is a self-contained `Module`. A module owns:

- Its **schemas** (Pydantic models for request/response)
- Its **service** (the ML pipeline / business logic)
- Its **router** (FastAPI routes)
- Its **trained models** (`.pkl`, `.onnx`, etc. — small ones in repo, large ones on HuggingFace Hub)
- Its **tests**

Modules implement `BaseModule` (`backend/core/base_module.py`), which forces:

```python
class BaseModule(ABC):
    name: str                    # URL slug, e.g. "manual_diagnosis"
    version: str                 # Semver
    description: str             # Shown in /modules endpoint

    @abstractmethod
    def get_router(self) -> APIRouter: ...

    def on_startup(self) -> None:  # optional: load models, warm caches
        pass

    def health_check(self) -> dict:  # required: is this module operational?
        ...
```

At startup, `backend/core/registry.py` scans `backend/modules/` and registers every `BaseModule` subclass it finds. No central import list to keep updated.

### Why not microservices?

Premature. A modular monolith gives 90% of the extensibility benefit with 10% of the operational pain. When a module gets heavy (say, a 2GB MRI classifier), split *just that one* into its own service later. The module interface is already the contract.

## Data flow

```
                          ┌──────────────────┐
                          │   Streamlit UI   │
                          └────────┬─────────┘
                                   │  HTTP / JSON
                          ┌────────▼─────────┐
                          │   FastAPI app    │
                          │  (auto-registers │
                          │     modules)     │
                          └────────┬─────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
       ┌──────▼──────┐      ┌──────▼──────┐      ┌──────▼──────┐
       │   Manual    │      │ Prescription│      │   Medical   │
       │  Diagnosis  │      │     OCR     │      │   Imaging   │
       │             │      │             │      │             │
       │ sklearn .pkl│      │  Tesseract  │      │  PyTorch /  │
       │             │      │   + spaCy   │      │   ONNX CNN  │
       └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                          ┌────────▼─────────┐
                          │  SQLite (dev) /  │
                          │  Supabase (prod) │
                          │                  │
                          │ - Predictions log│
                          │ - Audit trail    │
                          │ - User uploads   │
                          └──────────────────┘
```

## Where ML models live

| Model size | Where it lives | Loaded how |
|---|---|---|
| < 5 MB (sklearn risk scorers) | In repo at `data/models/` | At module startup |
| 5 MB – 200 MB (ONNX CNNs) | Hugging Face Hub | Downloaded once, cached locally |
| > 200 MB (large CNNs, LLMs) | Hugging Face Inference API | Called over HTTP |

This keeps repo size sane and lets us use models too big for local RAM.

## Training pipeline

Local machine has no GPU, so training runs on Colab. The flow:

1. Notebook in `ml_training/` opens dataset (often pulled from Kaggle).
2. Trains model with GPU (T4 free tier).
3. Evaluates and logs metrics.
4. Pushes model to Hugging Face Hub via `huggingface_hub` library.
5. Module's `service.py` references the HF repo ID; it downloads & caches on first use.

For tiny tabular models (diabetes risk from 8 features), training happens locally in 30 seconds — no Colab needed.

## Configuration & secrets

`backend/config.py` uses Pydantic Settings, loading from `.env`. Required keys are documented in `.env.example`. **No secret has a default that works** — startup fails loudly if a needed key is missing.

## Auth & rate limiting

Not in Phase 1 (this is a demo project). When needed, Phase 4 plan is:
- JWT auth via `fastapi-users` (open source)
- Rate limiting via `slowapi`
- Both deploy-able free

## Logging & observability

- Structured JSON logs via `structlog` (free, open source)
- Every prediction is logged to DB with: model name, model version, input hash (NOT raw input — privacy), prediction, confidence, latency
- Free observability stack later: Grafana Cloud free tier (50GB logs/month)

## Privacy & compliance posture

This is critical for a medical app even in demo:

- **No raw patient data in logs** — we hash PII before logging.
- **Uploaded images are deleted after inference** unless the user explicitly opts to save them.
- **Disclaimers everywhere** — every prediction response includes a `disclaimer` field that the frontend renders prominently.
- **HIPAA is NOT claimed** — this is not a HIPAA-compliant deployment. The README and UI say so. Compliance is a Phase 5 concern requiring legal review.

## Why these specific libraries

- **FastAPI** over Flask: native async (matters for image uploads), automatic OpenAPI docs, Pydantic validation built in.
- **Streamlit** over React (for now): we need to demonstrate ML capabilities fast; a designer-friendly React frontend can come in Phase 4.
- **scikit-learn** over deep learning for tabular: on small medical datasets (Pima Indians ~768 rows), gradient-boosted trees outperform neural nets and train in seconds.
- **PyTorch + MONAI** over TF/Keras for medical imaging: MONAI is a PyTorch-based library purpose-built for medical imaging (DICOM support, 3D conv, medical-specific augmentations).
- **Tesseract + EasyOCR (ensemble)** for prescription OCR: Tesseract handles printed text; EasyOCR handles handwritten and rotated text. Combining them recovers more than either alone.

## What we explicitly DON'T build

- Our own LLM. Use existing ones.
- Custom training infrastructure. Colab handles it.
- A mobile app. Streamlit on mobile-friendly responsive layout for now.
- Authentication in v1. Adds complexity without demo value.

## Open questions (decisions to revisit)

- Should we add **DICOM** support in Phase 1 imaging, or only PNG/JPG? (Leaning: Phase 2 — most demo images are PNGs.)
- Vector DB for chat assistant RAG: **ChromaDB** (free, embeds in process) vs **Qdrant** (free tier, separate service)? (Leaning: Chroma — simpler, sufficient at small scale.)
- Frontend rewrite in React: when? (Trigger: when Streamlit's UX limits become user-facing pain. Not before.)
