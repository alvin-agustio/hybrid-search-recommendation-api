# Hugging Face Space Verification Checklist

Use this after implementation, before sharing the Space URL publicly.

## 1. Local Tests

- [ ] Run the project test suite for the public demo path.
- [ ] Confirm loader tests use only public demo data and artifacts.
- [ ] Confirm `/search` tests cover at least one curated query, one typo query if typo recovery is enabled, and empty or invalid query handling.
- [ ] Confirm tests do not require `.env`, ClickHouse, private checkpoints, or private network access.

## 2. Local Uvicorn Smoke

- [ ] Start the public app locally with Uvicorn on port `7860`.
- [ ] `GET /health` returns a healthy status.
- [ ] `GET /docs` loads the OpenAPI UI.
- [ ] `GET /` explains that this is a public sanitized demo.
- [ ] `GET /examples` returns curated sample queries if the endpoint is implemented.
- [ ] `GET /search?query=sarapan&top_k=5` returns sensible breakfast-like results.
- [ ] `GET /search?query=kopi&top_k=5` returns sensible coffee-like results.
- [ ] `GET /search?query=sampho&top_k=5` returns hair-care-like results if typo recovery is implemented.
- [ ] Search responses include `demo_mode: true` or equivalent public-demo disclosure metadata.

## 3. Docker Build And Run

- [ ] Build the Docker image from a clean checkout using only committed public files.
- [ ] Confirm the image starts without `.env`, database credentials, private volumes, or local-only paths.
- [ ] Confirm the container listens on `0.0.0.0:7860`.
- [ ] Confirm `/health`, `/docs`, and `/search` work through the mapped host port.
- [ ] Confirm startup logs do not show ClickHouse connection attempts, training jobs, private artifact loading, or missing secret warnings.
- [ ] Confirm image size and cold-start time are acceptable for a Hugging Face Docker Space.

## 4. Hugging Face Space Endpoint Checks

- [ ] Open the deployed Space URL in a browser and confirm the app loads after cold start.
- [ ] Open `/docs` on the deployed Space and confirm the OpenAPI UI is usable.
- [ ] Call `/health` on the deployed Space and confirm it reports healthy.
- [ ] Call `/search?query=sarapan&top_k=5` and verify relevant results are returned.
- [ ] Call `/search?query=kopi&top_k=5` and verify relevant results are returned.
- [ ] Call `/search?query=sampho&top_k=5` if typo recovery is implemented and verify relevant results are returned.
- [ ] Confirm deployed responses include latency or retrieval metadata if implemented.
- [ ] Confirm no mutating, training, reload, job-monitor, or private personalization endpoints are exposed in `/docs`.

## 5. Confidentiality Checks

- [ ] Search the repository for private hostnames, credentials, tokens, connection strings, and internal dataset names before publishing.
- [ ] Confirm bundled data contains only public, synthetic, or explicitly sanitized product records.
- [ ] Confirm no customer, member, transaction, segment, or production SKU data is included.
- [ ] Confirm no private `.pt`, `.npy`, FAISS, tokenizer, checkpoint, or model artifact is bundled.
- [ ] Confirm public responses, logs, docs, and error messages do not reveal private infrastructure or internal paths.
- [ ] Confirm the README or Space landing page clearly states that the demo uses public sanitized data and precomputed public artifacts.

