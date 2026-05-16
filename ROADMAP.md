# Roadmap

A phased plan from "starter" to "production-ready scalable medical AI assistant." Each phase is independently shippable and demos something new.

---

## Phase 1 — Foundation & first diagnosis (Week 1-2) ✅ COMPLETE

**Goal**: Working end-to-end pipeline with one real ML feature.

- [x] Modular project skeleton (FastAPI + Streamlit + plugin architecture)
- [x] Configuration system, env management, logging
- [x] Module auto-registration (`BaseModule` + registry)
- [x] **Diabetes risk prediction** (Pima Indians dataset, gradient boosted trees)
- [x] Streamlit frontend with the diabetes form
- [x] Database logging of predictions
- [x] Unit tests for the diagnosis module
- [ ] Dockerfile for one-command run ← completed in Phase 6

**Demoable**: User enters glucose, BMI, age, etc., gets a calibrated risk score with explanation.

---

## Phase 2 — More tabular conditions + Prescription OCR (Week 3-4) ✅ COMPLETE

**Goal**: Cover 3 more conditions and ship the first vision feature.

- [x] **Hypertension risk** module (SAheart dataset / ESL synthetic, GradientBoosting, ROC-AUC 0.72)
- [x] **Heart disease risk** module (Cleveland dataset, GradientBoosting, ROC-AUC 0.94)
- [x] **Liver disease risk** (Indian Liver Patient dataset, GradientBoosting, ROC-AUC 0.74)
- [x] **Prescription OCR** module:
  - Tesseract OCR + PDF support via pypdfium2
  - Fuzzy-match medical dictionary for entity recognition
  - Output: structured JSON `{ medicines: [{name, dosage, frequency}] }`
- [x] Frontend pages for all 4 tabular conditions + OCR upload widget

**Demoable**: Upload a prescription photo, see extracted medicines in a table, get a structured summary.

---

## Phase 3 — Medical imaging (Week 5-7) ✅ COMPLETE

**Goal**: Image-based diagnosis from at least 2 modalities.

- [x] **Chest X-ray classifier** — 18-class torchxrayvision DenseNet-121 pretrained on CheXpert/NIH
  - Grad-CAM heatmap overlay showing *where* the model is looking
  - DICOM support via `pydicom`
- [x] **Brain MRI tumor classifier** — glioma / meningioma / pituitary / no tumor
  - ResNet-50 transfer learning scaffold + Colab training notebook (`ml_training/train_brain_mri.ipynb`)
  - HuggingFace Hub distribution (`rAhuL45647/brain-mri-resnet50`)
- [ ] Optional: **ECG signal classifier** — deferred (1D CNN on PTB-XL, different pipeline)

**Demoable**: Upload an X-ray, see classification + confidence + heatmap overlay.

---

## Phase 4 — Conversational AI assistant (Week 8-10) ✅ COMPLETE

**Goal**: Natural language medical Q&A grounded in trusted sources.

- [x] RAG pipeline:
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (22 MB, runs on CPU, 384-dim)
  - Vector store: ChromaDB persistent (embedded, no external service needed)
  - Knowledge base: 18 articles from MedlinePlus, WHO, CDC, NIH (all public domain)
- [x] LLM via Hugging Face Inference API (Meta-Llama-3-8B-Instruct — free tier)
  - Fallback: Ollama running locally with Phi-3 mini (~2GB, runs on CPU)
- [x] Safety layer: keyword blocklist for crisis content; hard disclaimer on every response
- [x] Multi-turn conversation history (last 10 turns, prompt uses last 4)
- [x] Streamlit chat interface with source citations and relevance scores

**Setup required**:
```
pip install sentence-transformers==3.3.1 chromadb==0.5.23 huggingface-hub==0.27.0
python scripts/ingest_knowledge_base.py
# Then set HUGGINGFACE_TOKEN in .env OR: ollama pull phi3:mini && ollama serve
```

**Demoable**: Ask "what are early signs of diabetes?" — get a grounded answer with citations.

---

## Phase 5 — Multi-modal orchestration & production hygiene (Week 11-13) ✅ COMPLETE

**Goal**: Tie modules together; production hygiene.

- [x] **Orchestrator module**: accepts free-text symptoms + optional image, routes to all relevant sub-modules, returns unified screening report
  - Tabular screens: diabetes (glucose+bmi+age), hypertension (systolic_bp+age)
  - Image screens: chest X-ray, brain MRI, prescription OCR
  - Keyword-based condition flagging for heart disease & liver disease
  - Structured synthesis with per-condition risk + recommendations
- [x] **Rate limiting** via `slowapi` — 15 req/min per IP on the /analyze endpoint
- [x] **Streamlit orchestrator page** (`9_Smart_Analysis.py`) — unified input form + multi-panel results
- [x] **Authentication** — `X-API-Key` middleware (dev: disabled, prod: set `API_KEY` in .env)
- [x] **Per-user consultation history** — `ConsultationHistory` table + `GET /orchestrator/history`
  - User key derived from hashed API key or IP (no PII stored)
  - History panel at bottom of Smart Analysis page
- [x] **Tamper-evident audit trail** — `chain_hash` column in `PredictionLog`
  - Chain: `SHA256(prev_hash | module_name | input_hash)` per entry
  - Verify integrity at `GET /api/v1/audit/verify`
  - Stats at `GET /api/v1/audit/stats`
- [x] **Comprehensive test suite** — 134 unit tests across all modules
  - Schema validation, risk-level mapping, feature-vector ordering, blocklist, edge cases

---

## Phase 6 — Scaling & polish (Week 14+) ✅ COMPLETE (infrastructure)

**Goal**: Production-readiness.

- [x] **Dockerfile** — multi-stage build with Tesseract OCR + PyTorch CPU
- [x] **Dockerfile.frontend** — lightweight Streamlit image
- [x] **docker-compose.yml** — orchestrates backend + frontend with health checks
- [x] **render.yaml** — one-file Render.com deployment config
- [x] **Supabase migration** — swap `DATABASE_URL` in .env; uncomment `psycopg2-binary` in requirements.txt
- [ ] React/Next.js frontend rewrite — deferred (when Streamlit limits hurt)
- [ ] Background tasks (long-running inference) via Celery + Redis (Upstash free tier)
- [ ] Model A/B testing infrastructure
- [ ] Continuous evaluation: synthetic test cases run on every model update
- [ ] PWA support (installable on phones)

### Supabase migration (zero code change)
```bash
# 1. Create a free project at supabase.com
# 2. Copy the connection string from Settings → Database → URI
# 3. In .env:
DATABASE_URL=postgresql://postgres:[password]@db.[project].supabase.co:5432/postgres
# 4. In requirements.txt: uncomment psycopg2-binary==2.9.10
# 5. Restart the backend — SQLAlchemy creates all tables automatically
```

### Render deployment
```bash
# Push to GitHub, then:
# 1. render.com → New → Web Service → connect repo
# 2. Render detects render.yaml automatically
# 3. Set HUGGINGFACE_TOKEN and API_KEY in Render environment dashboard
```

---

## Stretch goals (not planned, but on the radar)

- Dermatology image classifier (ISIC dataset)
- Fundus image analysis for diabetic retinopathy
- Voice input → symptom extraction → diagnosis (Whisper open-source)
- Drug interaction checker (DrugBank free academic license)
- Doctor-patient conversation summarizer
- Integration with FHIR (healthcare interoperability standard)

---

## What ships in v0.6.0 (current)

Everything marked `[x]` across Phases 1–6:
- Plugin architecture (BaseModule + registry) with **7 modules**: diabetes, hypertension, heart disease, liver disease, chest X-ray, brain MRI, RAG chat, orchestrator, audit
- **9 Streamlit frontend pages**
- Tamper-evident audit log with SHA-256 chain hashing
- API key authentication middleware (optional, prod-grade)
- Per-user consultation history
- 134-test suite covering schema validation, risk mapping, safety blocklist, edge cases
- Docker-ready: `docker compose up` runs the full stack
- Render.com deploy config (`render.yaml`)
