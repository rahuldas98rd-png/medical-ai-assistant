# Setting up free third-party services

Everything MediMind AI uses has a permanently-free tier. This guide walks through obtaining keys/accounts. **None of these are required for Phase 1** (the diabetes module runs purely locally). They become useful from Phase 3 onward.

## 1. Hugging Face (free model hub + inference API)

**Why we use it**: Host trained models (free, unlimited public models). Call hosted LLMs via Inference API (free tier with generous rate limit).

**Steps**:
1. Sign up at https://huggingface.co/join (free, no credit card)
2. Go to https://huggingface.co/settings/tokens
3. Create a token with `read` scope (for downloading models) and `write` if you'll push trained models
4. Paste into `.env` as `HUGGINGFACE_TOKEN=hf_...`

**Free limits**: Unlimited public model storage. Inference API: ~30,000 chars/month on free tier (enough for thousands of small queries).

## 2. Google Colab (free GPU training)

**Why we use it**: Train PyTorch models on a free Tesla T4 GPU.

**Steps**:
1. Sign in at https://colab.research.google.com with any Google account
2. No setup needed — Colab notebooks live in your Google Drive
3. To enable GPU: `Runtime` → `Change runtime type` → `T4 GPU`

**Free limits**: ~12 hour sessions, idle disconnection after ~90 min of inactivity. Plenty for training small-to-medium models. Daily GPU quotas exist but reset.

**Workflow**: Notebooks in `ml_training/` are designed to:
1. Open in Colab
2. Pull dataset (from Kaggle or HF Datasets)
3. Train with GPU
4. Push model back to HF Hub
5. Service code downloads from HF Hub on startup

## 3. Ollama (local LLM, free, optional)

**Why we use it**: Run open-source LLMs locally for the chat assistant when you can't / don't want to use HF Inference API.

**Steps**:
1. Download from https://ollama.com/download (Windows installer available)
2. Install — it sets up a local API at `http://localhost:11434`
3. Pull a small model: `ollama pull phi3:mini` (~2.3 GB, runs on your i5-12400 CPU)
4. Test: `ollama run phi3:mini`

**RAM cost**: phi3:mini uses ~3 GB of RAM. Llama 3.2 3B uses ~4 GB. Within your 16 GB budget. Don't try to run Llama 3 70B — it won't fit.

**Speed on your CPU**: phi3:mini ≈ 10-15 tokens/sec on i5-12400. Usable for chat.

## 4. Supabase (free Postgres, optional for production)

**Why we use it**: When SQLite isn't enough — switch the DATABASE_URL and the app keeps working.

**Steps**:
1. Sign up at https://supabase.com (free, no card)
2. Create a project
3. Settings → Database → connection string
4. Set `DATABASE_URL` in `.env` to that connection string
5. Uncomment `psycopg2-binary` in `requirements.txt` and reinstall

**Free limits**: 500 MB database, 2 GB bandwidth/month, paused after 1 week of inactivity (auto-resumes on first request).

## 5. Hugging Face Spaces (free deployment)

**Why we use it**: Deploy the whole app publicly, for free, forever.

**Steps**:
1. Create a Space at https://huggingface.co/new-space
2. Choose "Docker" SDK
3. Push your repo to the Space's git remote
4. The Dockerfile in this repo will build automatically

**Free limits**: CPU-only space, 16 GB RAM, ~2 vCPUs. Public spaces are free forever. Goes to sleep after 48 hours of inactivity (wakes on first request, ~30s cold start).

## 6. Kaggle (free datasets)

**Why we use it**: Most medical datasets we'll train on are mirrored on Kaggle (Pima Indians, Heart Disease UCI, Chest X-Ray Images, etc.).

**Steps**:
1. Sign up at https://www.kaggle.com (free)
2. Settings → Account → Create new API token (downloads `kaggle.json`)
3. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<user>\.kaggle\` (Windows)
4. Use `kaggle datasets download -d <user>/<dataset>` in scripts

Datasets that don't need Kaggle (already mirrored as raw CSV on GitHub) can be downloaded directly — see `scripts/train_diabetes_model.py` for an example.

## What we deliberately don't use

- **OpenAI / Anthropic / paid Gemini APIs** — not free permanently.
- **AWS / GCP / Azure paid tiers** — even free tiers eventually require credit cards or expire after 12 months.
- **Pinecone / Weaviate Cloud** for vector storage — free tiers are limited; ChromaDB embedded gives us enough for Phase 4.
