from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from .retrieval import HybridDemoSearch


BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "data" / "products.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"

searcher = HybridDemoSearch.from_paths(CATALOG_PATH, ARTIFACT_DIR)

app = FastAPI(
    title="Hybrid Retail Search API - Public Demo",
    description=(
        "Portfolio-safe FastAPI demo for hybrid BM25 + semantic retail search. "
        "Uses a public sample catalog and precomputed semantic artifacts."
    ),
    version="1.0.0-public-demo",
)


@app.get("/", response_class=HTMLResponse)
def root():
    examples = ["sarapan", "kopi", "sampho", "makanan kucing"]
    links = "".join(
        f'<li><a href="/search?query={query}&top_k=5">{query}</a></li>'
        for query in examples
    )
    return f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Hybrid Retail Search API</title>
        <style>
          body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 40px; line-height: 1.5; max-width: 840px; }}
          code, a {{ color: #075985; }}
          .note {{ border-left: 4px solid #0f766e; padding-left: 16px; color: #334155; }}
        </style>
      </head>
      <body>
        <h1>Hybrid Retail Search API</h1>
        <p class="note">Public portfolio demo. The original system used private retail data; this Space runs the same retrieval pattern on a small public sample catalog.</p>
        <p>Open <a href="/docs">/docs</a> for Swagger, check <a href="/health">/health</a>, or try an example search:</p>
        <ul>{links}</ul>
      </body>
    </html>
    """


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "demo_mode": True,
        "products_loaded": len(searcher.products),
        "retrieval_mode": "bm25+semantic",
    }


@app.get("/examples")
def examples() -> List[dict]:
    return [
        {"query": "sarapan", "why": "Conceptual breakfast search"},
        {"query": "kopi", "why": "Exact category and product intent"},
        {"query": "sampho", "why": "Typo-like hair care query"},
        {"query": "makanan kucing", "why": "Pet food semantic query"},
        {"query": "obat nyamuk", "why": "Household problem-oriented query"},
    ]


@app.get("/search")
def search(
    query: str = Query(..., min_length=1, max_length=120),
    top_k: int = Query(default=10, ge=1, le=25),
):
    try:
        response = searcher.search(query=query, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        **asdict(response),
        "results": [asdict(result) for result in response.results],
    }

