# Root Search UI Design

## Context

The Hugging Face Space is already live and healthy, but the root page at `/` is still a plain HTML landing page. The live API under `/search`, `/health`, `/examples`, and `/docs` works, yet the first impression is still closer to a stub than a usable public demo.

The goal of this design is to turn the root page into a minimal search tester that helps both non-technical reviewers and technical reviewers understand the project quickly, while keeping `/docs` intact for Swagger-first exploration.

## Problem Statement

The current root page asks the user to click links manually and read JSON. That is enough for a functional API demo, but not strong enough for a polished portfolio experience. The Space needs a lightweight, reliable interface that:

- lets users test the API immediately
- explains what kind of queries work well
- presents results as readable product cards rather than raw JSON
- still feels like an API/product demo, not a marketing landing page

## Design Goal

Replace the current root page with a search-first minimal UI that submits to the existing `/search` endpoint and renders the returned results as balanced cards.

"Balanced" here means:

- product-first enough for HR and casual reviewers
- technical enough for engineers to see scores and source labels

## User Experience

### Primary flow

1. User opens the Space root `/`.
2. User sees a single clean page with:
   - page title
   - one-line context statement
   - inline links to `/docs` and `/health`
   - one search input
   - one submit button
   - several clickable example queries
3. User enters a query or clicks an example query.
4. The page calls `/search` with the chosen query and `top_k=5`.
5. Returned products are shown as cards below the form.

### Result card content

Each result card shows:

- `sku_name`
- `source`
- `final_score`
- `bm25_score`
- `semantic_score`
- `explanation`

### Page states

The root page must support these states:

- idle state before first search
- loading state during request
- success state with result cards
- empty state when `results` is empty
- error state for invalid query or failed fetch

## Visual Direction

The page should stay intentionally minimal and tool-like.

### Layout

- Single-column layout
- Constrained readable content width
- Clear vertical rhythm
- No card-heavy marketing hero
- No large decorative graphics

### Style

- Light background
- Strong heading
- Muted support copy
- Calm blue/teal accents consistent with the current Space identity
- Thin borders and modest radius
- Clean result cards with compact metadata rows

### Interaction

- Search button disabled while loading
- Example queries clickable
- Query field updates when example is chosen
- Results replace previous results on each new request

## Technical Design

### Backend scope

Keep the existing backend contract unchanged:

- `GET /search`
- `GET /health`
- `GET /examples`
- `GET /docs`

No API contract change is required for this phase.

### Frontend delivery model

The UI will still be served by the existing FastAPI root route in `demo_search/api.py`, but the returned HTML should become a richer self-contained page with:

- embedded CSS
- embedded JavaScript
- form submission via `fetch('/search?...')`
- dynamic DOM rendering for cards and states

This keeps deployment simple and avoids adding a frontend build system.

### Root page responsibilities

The root route should:

- render the search UI shell
- preload example queries from the static list already mirrored in `/examples`
- handle fetch calls client-side
- render response data without page reload

### Error handling

Client-side UI should handle:

- `400` invalid short query
- `422` invalid parameter shapes
- network or server failure

Displayed error messages should be plain and short.

## File Scope

### Modify

- `demo_search/api.py`

### Reuse

- existing `/search`, `/health`, and `/examples` routes

### Optional

- tests in `tests/test_demo_search.py` to validate the root page contains the search UI markers

## Acceptance Criteria

The design is complete when:

- Opening `/` shows a working search form instead of the plain link list
- Entering `sarapan` renders cards on the page
- Clicking an example query runs the search without a full page reload
- Cards show balanced metadata: product name, source, and scores
- `/docs` remains available and unchanged
- The root page still feels minimal and readable on desktop and mobile

## Non-Goals

This phase does not include:

- authentication
- saving query history
- charts or analytics
- sorting/filtering controls
- pagination
- moving data loading to a separate frontend framework

## Implementation Notes

- Keep the page self-contained inside FastAPI for speed and simplicity
- Avoid adding JS frameworks
- Prefer semantic HTML and small helper functions over clever abstractions
- Preserve the public-demo explanation so the Space remains honest about private vs public data
