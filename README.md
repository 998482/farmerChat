# FasalGuru Backend — Feature 1: Dockerized FastAPI + Krishi Agent (RAG)

## Folder structure

```
fasalguru-backend/
├── app/
│   ├── main.py                     # FastAPI app entrypoint, mounts routers, loads vector store at startup
│   ├── core/
│   │   ├── config.py                # env vars / settings (Pydantic BaseSettings)
│   │   ├── vector_store.py          # PDF ingestion, Chroma build/load, E5 query/passage prefixing
│   │   └── llm_chain.py             # RAG chain: Groq primary -> Gemini fallback
│   ├── routers/
│   │   ├── health.py                # GET /health
│   │   └── krishi_agent.py          # POST /krishi-agent/ask
│   ├── models/
│   │   └── schemas.py               # Pydantic request/response models
│   └── ingestion/
│       └── ingest_docs.py           # Run this whenever you add/update PDFs: PDFs -> Chroma
├── data/
│   └── icar_kvk_docs/                # Put your ICAR/KVK PDFs here (gitignored, not code)
├── chroma_db/                        # Persisted vectorstore (created on first ingest, gitignored)
├── requirements.txt                  # Pinned, verified via real `uv pip install` (see report below)
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── .env                              # Your real keys go here (gitignored)
└── .env.example                      # Template
```

## Run locally (uv)

```bash
cd fasalguru-backend
uv venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
# open .env, fill in GROQ_API_KEY and GEMINI_API_KEY (EMBEDDING_MODEL is already set correctly)
# put your ICAR/KVK PDFs in data/icar_kvk_docs/
python -m app.ingestion.ingest_docs      # builds chroma_db/ — first run downloads the embedding
                                          # model (~1GB) and may take a few minutes
uvicorn app.main:app --reload --port 8000
```

Test:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/krishi-agent/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "चने में पीली पत्ती क्यों हो रही है?", "crop": "chickpea", "district": "sitapur"}'
```

Re-running `python -m app.ingestion.ingest_docs` is safe — it will NOT duplicate chunks
(see "Idempotency" in the verification report below).

## Run with Docker

```bash
docker compose up --build
```

---

## Verification report (2026-09-14)

This backend was inspected and tested inside a sandboxed environment. **That sandbox's
network only allows PyPI/GitHub/npm-type domains — `huggingface.co`, `api.groq.com`, and
Gemini's API are all blocked there (confirmed: both return HTTP 403).** So the tests below
split into two groups: things verified for real in the sandbox, and things that need to be
re-run on your machine (which has open internet) before you trust them fully.

### Environment
```
Python: 3.12.3
uv: 0.11.7
OS: Linux (sandbox) — your machine may differ, re-verify uv/python versions there
```

### Models (as configured)
```
Embedding model: intfloat/multilingual-e5-large  (local, HuggingFace, E5 query/passage prefixing applied)
Embedding dimension: 1024 (per intfloat's published model card — NOT independently verified
                             in-sandbox since the model can't be downloaded here)
LLM primary: Groq (llama-3.3-70b-versatile)
LLM fallback: Gemini (gemini-2.5-flash)
```

### What was actually executed and verified in-sandbox

| # | Test | Result |
|---|------|--------|
| 1 | `uv pip install -r requirements.txt` from a clean venv, 2x independently | PASS — no dependency conflicts |
| 2 | Import every module (`config`, `vector_store`, `llm_chain`, `schemas`, `routers`, `main`) against the real installed packages | PASS |
| 3 | Real PDF → PyPDFLoader → RecursiveCharacterTextSplitter, using 3 generated test PDFs (wheat/chickpea/potato advisories) | PASS — text extracted correctly, 3 usable chunks, 0 empty chunks, metadata (`source`, `page`) attached correctly |
| 4 | E5 prefix logic — mocked the embedding model's `encode` call and inspected exactly what string reached it | PASS — documents get `"passage: "` exactly once, queries (tested with a real Hindi sentence) get `"query: "` exactly once, no double-prefixing, no swapped prefixes |
| 5 | **Idempotent ingestion** — ran `build_vector_store_from_pdfs()` 3x on the same PDFs against a real (but fake-embedded) Chroma store | **Found bug, fixed, re-verified: 3 → 3 → 3 chunks.** Before the fix it was 3 → 6 → 9 (see Problems Found) |
| 6 | Persistence/restart — reopened the same Chroma persist directory in a **fresh Python process** with no re-ingestion call | PASS — collection count unchanged, `similarity_search()` returned results with correct `source`/`page` metadata |
| 7 | `GET /health` via FastAPI `TestClient` | PASS — returns `vector_store_loaded: false` before ingestion, as expected |
| 8 | `POST /krishi-agent/ask` with no vector store loaded | PASS — clean `503` with a readable message, not a traceback |
| 9 | `POST /krishi-agent/ask` with an empty `query` | PASS — clean `422` (Pydantic `min_length=1` validation) |
| 10 | `POST /krishi-agent/ask` full flow with retrieval mocked (real docs) and Groq call mocked (real response object) | PASS — `200`, `provider_used: "groq"`, `sources: ["wheat_advisory.pdf"]` |
| 11 | Same, but Groq's mocked call raises a 429 | PASS — automatically fell back to Gemini, `provider_used: "gemini"`, answer still returned, `200` |
| 12 | Both providers' API keys empty/missing | **Found bug, fixed, re-verified** — was a cryptic downstream error; now raises a clear `ValueError` per provider, still attempts fallback, final client response is a clean `502` with a readable message (server log has the full trace, client doesn't) |
| 13 | `.env` file loading via `pydantic-settings` | PASS — values load correctly with correct types (e.g. `RETRIEVER_TOP_K` loads as `int`, not `str`) |
| 14 | Embedding model loaded once (singleton), not per-request | PASS by inspection — `get_embeddings()` uses a module-level cache; `load_vector_store()` is only called once, at FastAPI startup |

### What could NOT be tested in-sandbox (needs your machine + real keys)

* **Actual semantic retrieval quality** — whether a real Hindi query about wheat irrigation
  actually retrieves the right chunk out of real ICAR PDFs. The sandbox test above proves the
  *mechanics* work (persistence, metadata, no duplication) using a fake deterministic embedding,
  not real semantic similarity — a fake hash-based embedding has no notion of meaning, so it
  returned unrelated chunks in the demo run, which is expected and not a bug.
* **Real Groq calls** — model availability, actual answer quality/latency, real rate-limit behavior.
* **Real Gemini fallback** — same, for the fallback path.
* **Out-of-domain query behavior** ("Who is the PM of Canada?") — the system prompt already
  instructs the LLM to say the answer isn't in the ICAR/KVK context, but whether Groq/Gemini
  actually follow that instruction can only be confirmed with a live call.

### Suggested test script for your machine

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/krishi-agent/ask -H "Content-Type: application/json" \
  -d '{"query": "गेहूं में सिंचाई कब करनी चाहिए?"}'

curl -X POST http://localhost:8000/krishi-agent/ask -H "Content-Type: application/json" \
  -d '{"query": "What are the critical irrigation stages for wheat?"}'

curl -X POST http://localhost:8000/krishi-agent/ask -H "Content-Type: application/json" \
  -d '{"query": "आलू की फसल में कौन से रोग हो सकते हैं?"}'

curl -X POST http://localhost:8000/krishi-agent/ask -H "Content-Type: application/json" \
  -d '{"query": "Who is the Prime Minister of Canada?"}'
```
Paste the actual JSON responses back and I'll tell you if retrieval/answers look right.

### Problems found, fixed, and verified this round

1. **Duplicate chunks on re-ingestion.** `build_vector_store_from_pdfs()` called
   `Chroma.from_documents()` with no explicit IDs — Chroma auto-generates a random ID per call,
   so re-running ingestion (e.g. after adding one new PDF) re-inserted every existing chunk too.
   Reproduced: 3 chunks → re-run → 6 → re-run → 9. **Fix:** deterministic ID =
   `sha256(source_filename + page_number + chunk_text)`, passed as `ids=` to
   `Chroma.from_documents()`. Verified 3 → 3 → 3 across 3 runs. *File: `app/core/vector_store.py`.*
   **Known limitation:** if a PDF's content is edited, its chunks get new IDs and old ones are
   NOT automatically removed — delete `chroma_db/` and re-ingest fully if you replace PDFs.

2. **Missing API key produced a cryptic error instead of a clear one**, and didn't reliably
   trigger the Gemini fallback. **Fix:** `get_llm()` now raises a clear `ValueError` naming the
   missing env var; `answer_query()` now catches `ValueError` alongside the existing
   `HTTPStatusError`/`RuntimeError`, so a missing Groq key still falls back to Gemini instead of
   crashing outright. *File: `app/core/llm_chain.py`.*

3. **Empty/whitespace-only chunks and unreadable PDFs weren't handled** — a corrupt or
   image-only (scanned) PDF would have thrown an unhandled exception and killed the whole
   ingestion run, and a rare empty chunk would waste an embedding call on nothing.
   **Fix:** per-PDF `try/except` (logs a warning and skips that file, continues with the rest),
   plus a filter that drops whitespace-only chunks before embedding. *File: `app/core/vector_store.py`.*

4. **`requirements.txt` had a real, reproducible dependency conflict**
   (`langchain==0.3.4` required `langchain-core>=0.3.12,<0.4.0`, but the file pinned
   `langchain-core==0.3.10`) — `uv pip install` failed outright from a clean environment.
   **Fix:** re-resolved with `uv`, verified every import against the real installed packages,
   pinned the actual resolved versions (see `requirements.txt` — note this pulled in
   `langchain==1.4.0`, a major version jump from what was originally pinned; all imports used
   in this codebase were re-verified against it and work).

### Dependency notes

* `sentence-transformers` pulls in `torch` with full NVIDIA CUDA dependencies even on a
  machine with no GPU — this is normal PyPI behavior for `torch` on Linux, not a
  misconfiguration. Expect a large first-time download (~3-4GB). It runs fine on CPU.
* `langchain_community.document_loaders.PyPDFLoader` (used for PDF loading) currently emits a
  deprecation warning — `langchain-community` is being sunset in favor of standalone
  integration packages. It still works correctly (verified above) — not urgent, but worth
  migrating later if LangChain removes it.
* No Mistral packages anywhere in the dependency tree — confirmed via `grep`.

### Known limitations (carried forward, not fixed this round — by design)

* Crop/district filtering is "soft" (folded into the query text for semantic search), not a
  hard Chroma metadata filter — see the note in `vector_store.py` for why.
* Editing a PDF's content in place doesn't remove its old chunks from Chroma automatically
  (see Problem 1 above) — delete `chroma_db/` and re-ingest for a full rebuild after content edits.
* Actual answer quality, real rate-limit handling, and semantic retrieval accuracy are untested
  here (sandbox network restriction) — please run the test script above and share the results.
