# Hugging Face Space Deployment Plan

**Goal:** publish a portfolio-safe, API-first Hugging Face Space for this hybrid search and recommendation project.

**Target outcome:** a public URL where reviewers can open API docs, call `/search`, see hybrid BM25 + semantic retrieval on a small non-confidential catalog, and understand that the production version originally used private ClickHouse data and internal model artifacts.

**Recommended approach:** inference-only Docker Space with a frozen public sample catalog and precomputed semantic artifacts.

---

## 1. Public Demo Scope

### Keep

- `GET /health`
- `GET /search`
- Hybrid ranking: BM25 lexical search + semantic retrieval + score fusion
- Small public retail catalog
- Precomputed product embeddings or vector index
- Query-time semantic retrieval
- Swagger/OpenAPI docs
- Minimal API-first demo page or docs landing section
- Clear disclosure that original data and production artifacts were confidential

### Cut From Public Space

- ClickHouse runtime dependency
- Training API service
- Model retraining jobs
- `/reload` model/data endpoint
- Live customer/member personalization
- RFM segment lookup backed by private customer data
- Background job monitor
- Runtime writes to checkpoints/jobs folders
- Large private `.pt`, `.npy`, FAISS, or catalog artifacts

### Why This Scope Works

This keeps the technical heart of the project visible while removing the parts that cannot safely or reliably run in a public portfolio deployment. The demo still proves search engineering, semantic retrieval, API design, and deployment judgment.

---

## 2. Target Architecture

```text
Reviewer / Browser
        |
        v
Hugging Face Space URL
        |
        v
FastAPI app on port 7860
        |
        +--> /health
        +--> /search
        +--> /docs
        |
        v
Offline public runtime assets
        |
        +--> sample catalog
        +--> BM25 index built at startup
        +--> precomputed product embeddings/index
        +--> lightweight query encoder or embedding fallback
```

The Space should boot without any private network, database, credentials, or training workflow.

---

## 3. Phase Plan

### Phase 0: Documentation And Scope Lock

**Purpose:** make the public deployment boundaries explicit before touching runtime code.

**Tasks:**

- Add this deployment plan under `docs/`.
- Add a short public-demo note to the root `README.md`.
- Decide the public dataset source and size.
- Decide whether the first public semantic layer uses FAISS or exact cosine over a small embedding matrix.

**Acceptance criteria:**

- The repo explains that private data is not included.
- The public demo scope is clear.
- No one reviewing the repo expects ClickHouse credentials or internal artifacts.

---

### Phase 1: Public Dataset And Artifacts

**Purpose:** replace confidential runtime inputs with safe, reproducible public inputs.

**Create:**

- `search-engine-service/public_demo/data/products.csv`
- `search-engine-service/public_demo/artifacts/`
- `search-engine-service/scripts/build_public_demo_artifacts.py`

**Alternative package layout:**

If imports become awkward because `search-engine-service` contains a hyphen, create a clean demo package at the repo root:

- `demo_search/api.py`
- `demo_search/loaders.py`
- `demo_search/retrieval.py`
- `demo_search/schemas.py`

This package can still reuse selected modules from `search-engine-service/inference/`, or it can copy only the public-safe retrieval logic if that keeps the Space simpler.

**Dataset requirements:**

- 200 to 2,000 public or synthetic retail products.
- Columns compatible with existing code:
  - `sku_id`
  - `sku_name`
  - `division_name`
  - `dept_name`
  - `class_name`
  - `subclass_name`
  - `group_name`
- Include enough examples for conceptual queries:
  - `sarapan`
  - `cemilan`
  - `kopi`
  - `susu bayi`
  - `shampoo`
  - `obat nyamuk`
  - `makanan kucing`

**Artifact options:**

- Preferred first version: precomputed embeddings stored as `.npy`, searched by exact cosine.
- Upgrade option: add FAISS index after the Space is stable.

**Git/artifact note:**

The current `.gitignore` excludes common model/runtime artifact types such as `.npy`, `.pt`, tokenizer files, and runtime outputs. For the Space repo, use one of these approaches:

- Keep small sanitized `.csv`, `.json`, and `.npy` demo artifacts under an explicit allowlist.
- Use Git LFS for larger public demo artifacts.
- Store artifacts in a separate Hugging Face Dataset or Model repo and download them during build, not during request handling.

**Acceptance criteria:**

- Artifacts can be rebuilt locally from public files only.
- No internal SKU, transaction, customer, or segment data is present.
- Demo queries produce sensible top results.

---

### Phase 2: Offline Inference App

**Purpose:** create a public FastAPI entrypoint that does not depend on ClickHouse, training, or private checkpoints.

**Create:**

- `search-engine-service/api_public.py`
- `search-engine-service/public_demo/loader.py`
- `search-engine-service/public_demo/semantic.py`

If the clean package layout is chosen, use:

- `demo_search/api.py`
- `demo_search/loaders.py`
- `demo_search/retrieval.py`
- `demo_search/schemas.py`

**Reuse:**

- `search-engine-service/inference/bm25.py`
- `search-engine-service/inference/hybrid_search.py` if compatible after adapting semantic search output
- `search-engine-service/inference/post_processing.py`

**Avoid in public runtime:**

- `search-engine-service/api_training.py`
- `search-engine-service/training/`
- `search-engine-service/inference/segment_fusion.py`
- ClickHouse-backed `load_products`

**Public endpoints:**

- `GET /`
- `GET /health`
- `GET /search?query=...&top_k=10`
- `GET /examples`
- `GET /docs`

**Recommended `/search` response additions:**

- `query`
- `results`
- `latency_ms`
- `demo_mode: true`
- `retrieval_mode: "bm25+semantic"`
- `notes` explaining that the dataset is public and sanitized

**Acceptance criteria:**

- App boots with no `.env`.
- App boots with no ClickHouse.
- App searches the bundled public catalog.
- App returns useful JSON for curated example queries.

---

### Phase 3: API-First Showcase Experience

**Purpose:** make the Space understandable for both technical reviewers and HR/recruiters.

**Recommended UX:**

- Keep Swagger/OpenAPI as the main technical interface.
- Add a minimal root page at `/` with:
  - project title
  - one-sentence public demo disclosure
  - links to `/docs`, `/health`, and example `/search` calls
  - three curated example queries
- Do not build a large marketing landing page for the first deploy.

**Example root page copy:**

```text
Public portfolio demo of a hybrid retail search API.
The original system used private retail data; this Space runs the same retrieval pattern on a small public sample catalog.
```

**Acceptance criteria:**

- A non-technical reviewer can understand what to click within 30 seconds.
- A technical reviewer can inspect `/docs` and try `/search`.
- The page states what is live and what is precomputed.

---

### Phase 4: Hugging Face Space Packaging

**Purpose:** make the app deployable as a Docker Space.

**Create:**

- `Dockerfile`
- `.dockerignore`
- `space-requirements.txt` or a trimmed public requirements file

**Hugging Face Space metadata:**

The Space `README.md` should include:

```yaml
---
title: Hybrid Retail Search API
emoji: Search
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
```

If keeping this repo as the source repo, add the metadata to the Space-specific README used by Hugging Face.

**Runtime command:**

```bash
uvicorn api_public:app --host 0.0.0.0 --port 7860
```

**Dependency strategy:**

- Start with the smallest possible dependency set.
- Include FastAPI, Uvicorn, Pandas, NumPy, rank-bm25, SymSpell, and optionally scikit-learn.
- Add FAISS only if needed after the first successful deploy.
- Avoid PyTorch/Transformers in the first public runtime unless the query encoder truly needs them.
- If using `sentence-transformers`, pin the model and cache/download it at build time so startup behavior is predictable.

**Acceptance criteria:**

- Docker build succeeds locally or in HF Space logs.
- The app listens on port `7860`.
- `/health` responds after cold start.
- `/docs` loads.
- `/search` returns results.

---

### Phase 5: Verification And Portfolio Polish

**Purpose:** make the deployed URL reliable enough to send to recruiters.

**Tests/checks:**

- `GET /health` returns `status: healthy`.
- `GET /search?query=sarapan&top_k=5` returns breakfast-like products.
- `GET /search?query=sampho&top_k=5` returns hair-care-like products if typo recovery is included.
- `GET /search?query=kopi&top_k=5` returns coffee-like products.
- Cold start is acceptable for a free Space.
- No private hostnames, credentials, or internal dataset names appear in public responses.

**Docs polish:**

- Root `README.md` links to the Space.
- Public demo section explains:
  - original data was confidential
  - public demo uses a safe sample catalog
  - semantic artifacts are prepared ahead of deployment
  - ranking still happens live per query

**Acceptance criteria:**

- The Space has a public URL.
- The repo has clear local run instructions for the public demo.
- A reviewer can understand the difference between original production architecture and public demo architecture.

---

## 4. Proposed Implementation Order

1. Write public demo dataset and artifact builder.
2. Create offline public loader.
3. Create simple semantic search over precomputed embeddings.
4. Create `api_public.py`.
5. Add basic tests for loader and `/search`.
6. Add Docker packaging.
7. Run locally.
8. Push to Hugging Face Space.
9. Verify deployed endpoints.
10. Update README with deployed URL.

---

## 5. Risks And Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Space cold start is slow | Reviewers may abandon the demo | Keep runtime dependencies small and avoid training/model loading at startup |
| Dataset looks too small | Demo may feel toy-like | Curate enough products and example queries to show behavior clearly |
| Semantic layer appears fake | Technical credibility drops | Explain that artifacts are precomputed but retrieval/ranking is live |
| ClickHouse dependency remains | Space fails to boot | Use separate `api_public.py` and public loader |
| Public endpoint exposes mutating operations | Operational risk | Do not expose `/reload`, training routes, or job routes |
| Heavy ML dependencies break build | Space deploy becomes fragile | Start with lightweight embeddings and add heavier models later only if needed |

---

## 6. Definition Of Done

The deployment project is complete when:

- A Hugging Face Space public URL is accessible.
- `/docs`, `/health`, and `/search` work on the deployed Space.
- The demo uses no confidential data or private infrastructure.
- The README explains the public demo scope honestly.
- At least five curated example queries produce useful results.
- The code path used by the Space can run locally from public assets only.

---

## 7. References

- Hugging Face Spaces Overview: https://huggingface.co/docs/hub/en/spaces-overview
- Hugging Face Docker Spaces: https://huggingface.co/docs/hub/main/en/spaces-sdks-docker
- Hugging Face Docker FastAPI demo: https://huggingface.co/docs/hub/spaces-sdks-docker-first-demo
