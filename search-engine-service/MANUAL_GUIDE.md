# MANUAL GUIDE - RetailCo Personalized Search API

This API does not ship with a standalone user interface. It is accessed through HTTP requests such as `curl`, Postman, Swagger UI, or an application frontend.

This guide explains how to start the services and how to call the main endpoints.

---

## 1. Setup and Run the Services

### A. Run the Search API (Inference)
Start the search service on port `8000`. On a cold start, the system needs a few seconds to fetch product data and load the AI models into memory.

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Interactive API documentation (Swagger UI):** `http://localhost:8000/docs`

### B. Run the Training API
Start the training service in a separate terminal on port `8001`. This service is responsible for retraining and model lifecycle operations.

```bash
uvicorn api_training:app --host 0.0.0.0 --port 8001
```

---

## 2. How to Use the Search Endpoints

All product retrieval requests are served through `GET /search` on port `8000`.

### A. Standard Search (Non-Personalized)
Use this when the user is anonymous or no member profile is available.

**Request:**

```text
GET /search?query=susu%20bayi&top_k=10
```

**Response behavior:**
Returns up to 10 products ranked by hybrid relevance.

### B. Personalized Search
Pass a valid member ID to boost results according to the customer's RFM-driven segment behavior.

**Request:**

```text
GET /search?query=kopi&member_id=00000000000&top_k=10
```

**Response differences:**
- The `results` array contains product objects.
- `segment_score` becomes greater than `0` for products lifted by segment preferences.
- `boosted: true` appears on products that benefited from reranking.

### C. Member Segment Lookup
Use this endpoint when a downstream system only needs to know which segment a member belongs to.

**Request:**

```text
GET /member/00000000000
```

**Response example:**

```json
{
  "member_id": "00000000000",
  "segment": "Big Spenders",
  "found": true
}
```

---

## 3. How to Use the Training API

The training service runs only on port `8001` and is intended for scheduled automation through tools such as Airflow or Cron.

### Trigger a Full Retraining Run (Stage 1 and Stage 2)
The following example retrains the models using one year of data.

```bash
curl -X POST "http://localhost:8001/jobs/train/full" \
     -H "Content-Type: application/json" \
     -d '{"start_date":"2023-01-01", "end_date":"2023-12-31"}'
```

### Check Training Status
Training can take hours, so the API immediately returns a `job_id`. Use that ID to poll for progress.

```bash
curl "http://localhost:8001/jobs/status/{job_id}"
```

The status will move from `running` to `completed` once the process finishes.

### Reload the Search Service with the Latest Model
Once training is complete, trigger a model reload on the search API so it swaps the active model without downtime.

```bash
curl -X POST "http://localhost:8000/reload"
```

---

## 4. Basic Troubleshooting

| Issue | Prevention and Resolution |
|------|----------------------------|
| **`Pipeline not initialized` or the server fails to start** | Check network connectivity and verify the ClickHouse settings in `.env`. The API refuses to start if the primary database is unavailable during startup. |
| **Search or model loading is too slow** | Check server RAM and CPU usage. FAISS and PyTorch need stable memory headroom, typically at least 4 to 8 GB of free RAM. |
| **`Access to the path is denied` during training** | Ensure the process has write access to `runtime/` and `runtime/checkpoints`. On Windows, adjust folder permissions if needed. |
| **Queries often return zero results** | The query may be severely misspelled or too extreme. If the same pattern happens repeatedly, review the `QUERY_ALIASES` mapping in `api.py` and consider expanding the normalization rules. |
