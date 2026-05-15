# Roadmap

A phased plan from "starter" to "production-ready scalable medical AI assistant." Each phase is independently shippable and demos something new.

---

## Phase 1 — Foundation & first diagnosis (Week 1-2) ← **YOU ARE HERE**

**Goal**: Working end-to-end pipeline with one real ML feature.

- [x] Modular project skeleton (FastAPI + Streamlit + plugin architecture)
- [x] Configuration system, env management, logging
- [x] Module auto-registration (`BaseModule` + registry)
- [x] **Diabetes risk prediction** (Pima Indians dataset, gradient boosted trees)
- [x] Streamlit frontend with the diabetes form
- [x] Database logging of predictions
- [ ] Unit tests for the diagnosis module
- [ ] Dockerfile for one-command run

**Demoable**: User enters glucose, BMI, age, etc., gets a calibrated risk score with explanation.

---

## Phase 2 — More tabular conditions + Prescription OCR (Week 3-4)

**Goal**: Cover 3 more conditions and ship the first vision feature.

- [ ] **Hypertension risk** module (model trained on UCI heart dataset features)
- [ ] **Heart disease risk** module (Cleveland heart dataset, XGBoost)
- [ ] **Liver disease risk** (Indian Liver Patient dataset — locally relevant)
- [ ] **Prescription OCR** module:
  - Tesseract + EasyOCR ensemble for text extraction
  - spaCy + medical dictionary for entity recognition (medicines, dosage, frequency)
  - Output: structured JSON `{ patient_name, doctor, date, medicines: [{name, dosage, frequency, duration}] }`
- [ ] Frontend: file upload widget + result viewer

**Demoable**: Upload a prescription photo, see extracted medicines in a table, get a structured summary.

---

## Phase 3 — Medical imaging (Week 5-7)

**Goal**: Image-based diagnosis from at least 2 modalities.

- [ ] **Chest X-ray classifier** — pneumonia/normal first, then 14-class CheXpert
  - Backbone: DenseNet-121 pretrained on CheXpert (download from HF Hub)
  - Export to ONNX for fast CPU inference
- [ ] **Brain MRI tumor classifier** — glioma / meningioma / pituitary / no tumor
  - Backbone: ResNet-50 transfer learning, trained on Colab
- [ ] Optional: **ECG signal classifier** — needs different pipeline (1D CNN on PTB-XL dataset)
- [ ] Heatmap visualization (Grad-CAM) — show *where* the model is looking
- [ ] DICOM support (via `pydicom`)

**Demoable**: Upload an X-ray, see classification + confidence + heatmap overlay.

---

## Phase 4 — Conversational AI assistant (Week 8-10)

**Goal**: Natural language medical Q&A grounded in trusted sources.

- [ ] RAG pipeline:
  - Embeddings: `sentence-transformers` (all-MiniLM, free, runs on CPU)
  - Vector store: ChromaDB (embedded, no external service)
  - Knowledge base: ingest MedlinePlus, WHO fact sheets, NIH content (all public domain / free)
- [ ] LLM via Hugging Face Inference API (Llama 3.1 8B Instruct or Mistral 7B — free tier)
  - Fallback: Ollama running locally with Phi-3 mini (~2GB, runs on CPU)
- [ ] Safety layer: refuse to give specific dosing/treatment advice; always recommend professional consultation
- [ ] Streaming responses to frontend

**Demoable**: Ask "what are early signs of diabetes?" — get a grounded answer with citations.

---

## Phase 5 — Multi-modal orchestration & auth (Week 11-13)

**Goal**: Tie modules together; production hygiene.

- [ ] **Orchestrator module**: takes any input (prescription image + symptoms), routes to relevant modules, synthesizes a unified report
- [ ] Authentication (fastapi-users)
- [ ] Per-user history of consultations
- [ ] Rate limiting
- [ ] Audit trail with hash-linked logs (tamper-evident)
- [ ] Multi-language support (start with English + Hindi + Bengali for India)

---

## Phase 6 — Scaling & polish (Week 14+)

**Goal**: Production-readiness.

- [ ] React/Next.js frontend rewrite (when Streamlit limits hurt)
- [ ] Migrate SQLite → Supabase (still free tier)
- [ ] Deploy backend on Hugging Face Spaces or Render (free)
- [ ] Background tasks (long-running inference) via Celery + Redis (Upstash free tier)
- [ ] Model A/B testing infrastructure
- [ ] Continuous evaluation: synthetic test cases run on every model update
- [ ] PWA support (installable on phones)

---

## Stretch goals (not planned, but on the radar)

- Dermatology image classifier (ISIC dataset)
- Fundus image analysis for diabetic retinopathy
- Voice input → symptom extraction → diagnosis (Whisper open-source)
- Drug interaction checker (DrugBank free academic license)
- Doctor-patient conversation summarizer
- Integration with FHIR (healthcare interoperability standard)

---

## What ships in v1 (today)

Everything marked `[x]` in Phase 1 above:
- Plugin architecture
- One trained model (diabetes)
- Working API + UI
- Docker config
- Clear path forward

The next concrete step after merging Phase 1 is **Hypertension module** — same pattern as diabetes, ~2 hours of work, validates the "adding a module is easy" claim of the architecture.
