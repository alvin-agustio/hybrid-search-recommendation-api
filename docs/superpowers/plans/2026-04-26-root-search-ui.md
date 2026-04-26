# Root Search UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the plain root landing page with a minimal interactive search tester that calls `/search` and renders balanced result cards.

**Architecture:** Keep the UI self-contained inside the existing FastAPI root route in `demo_search/api.py`. The page will render server-side HTML with embedded CSS and JavaScript, call the existing `/search` endpoint with `fetch`, and update the DOM without full page reloads. Existing API endpoints remain unchanged.

**Tech Stack:** FastAPI, plain HTML/CSS/JavaScript, pytest, Playwright verification

---

## File Structure

- Modify: `demo_search/api.py`
  Responsibility: render the root UI shell, client-side fetch logic, states, and result cards
- Modify: `tests/test_demo_search.py`
  Responsibility: validate the root page now exposes the search UI markers and interactive shell
- Modify: `README.md`
  Responsibility: keep public demo instructions aligned with the root UI behavior if needed

---

### Task 1: Add Root UI Regression Tests

**Files:**
- Modify: `tests/test_demo_search.py`

- [ ] **Step 1: Write failing tests for the root UI shell**

Add tests that assert:

```python
def test_root_page_contains_search_ui():
    from demo_search.api import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert 'id="search-form"' in html
    assert 'id="query-input"' in html
    assert 'id="results"' in html
    assert "Try an example query" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_demo_search.py::test_root_page_contains_search_ui -q
```

Expected: FAIL because the current root page only contains static links and does not have the new UI markers.

- [ ] **Step 3: Keep existing endpoint tests intact**

Do not remove the existing tests for `/search`, `/favicon.ico`, or result filtering.

- [ ] **Step 4: Re-run the full demo test file after implementation**

Run:

```bash
python -m pytest tests/test_demo_search.py -q
```

Expected: PASS after the UI implementation lands.

---

### Task 2: Implement Minimal Root Search UI

**Files:**
- Modify: `demo_search/api.py`
- Test: `tests/test_demo_search.py`

- [ ] **Step 1: Keep API contract unchanged**

Do not change:

```python
@app.get("/health")
@app.get("/examples")
@app.get("/search")
```

The new UI should call the existing `/search` endpoint.

- [ ] **Step 2: Replace the current root page HTML**

Implement a root page with:

```text
- title
- short context note
- link row: /docs, /health
- search form
- clickable example chips/buttons
- status area
- results area
```

The returned HTML must include stable markers:

```html
<form id="search-form">
  <input id="query-input" ...>
  <button id="search-button" ...>Search</button>
</form>
<div id="status"></div>
<section id="results"></section>
```

- [ ] **Step 3: Add client-side fetch behavior**

Implement minimal inline JavaScript that:

```javascript
async function runSearch(query) {
  const response = await fetch(`/search?query=${encodeURIComponent(query)}&top_k=5`);
  const payload = await response.json();
  // update status + results
}
```

Requirements:

- submit without page reload
- disable button while loading
- render empty state when `results.length === 0`
- render error message when response is not OK
- clicking an example query should fill the input and trigger search

- [ ] **Step 4: Render balanced result cards**

Each card must show:

```text
sku_name
source
final_score
bm25_score
semantic_score
explanation
```

Use compact metadata rows rather than raw JSON.

- [ ] **Step 5: Keep the page minimal**

Do not add:

- extra routes
- frontend framework
- analytics
- pagination
- large decorative hero sections

---

### Task 3: Align Public Demo Copy

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Check whether root page description in README still matches behavior**

The README should not imply the root page is only a plain landing page if it now functions as an interactive search tester.

- [ ] **Step 2: Adjust wording only if needed**

Keep the existing deploy/run instructions, but ensure the public demo description matches the new root behavior.

---

### Task 4: Verify Locally And In Browser

**Files:**
- Modify: none

- [ ] **Step 1: Run demo tests**

Run:

```bash
python -m pytest tests/test_demo_search.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Smoke test the root UI with a local FastAPI client**

Confirm the HTML contains:

```text
search form
query input
results container
example query copy
```

- [ ] **Step 3: Verify browser flow with Playwright**

Check:

- root page loads
- entering `sarapan` shows cards
- clicking example query triggers search
- `/docs` still loads

- [ ] **Step 4: Commit**

```bash
git add demo_search/api.py tests/test_demo_search.py README.md
git commit -m "feat: add interactive root search UI"
```

---

## Self-Review

- Spec coverage: covered root UI shell, example queries, balanced cards, states, and preservation of `/docs`
- Placeholder scan: no TODO/TBD placeholders left
- Type consistency: all referenced markers and files are consistent with the current `demo_search` structure
