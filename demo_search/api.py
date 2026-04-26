from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from .retrieval import HybridDemoSearch


BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "data" / "products.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"

searcher = HybridDemoSearch.from_paths(CATALOG_PATH, ARTIFACT_DIR)
FAVICON_SVG = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="12" fill="#075985"/><path d="M18 20h28v6H35v8h10v6H35v14h-8V20z" fill="#f8fafc"/></svg>"""
EXAMPLE_QUERIES = [
    {"query": "sarapan", "why": "Conceptual breakfast search"},
    {"query": "kopi", "why": "Exact category and product intent"},
    {"query": "sampho", "why": "Typo-like hair care query"},
    {"query": "makanan kucing", "why": "Pet food semantic query"},
    {"query": "obat nyamuk", "why": "Household problem-oriented query"},
]

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
    example_buttons = "".join(
        f'<button class="example-chip" type="button" data-query="{item["query"]}">{item["query"]}</button>'
        for item in EXAMPLE_QUERIES
    )
    return f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Hybrid Retail Search API</title>
        <style>
          :root {{
            color-scheme: light;
            --bg: #f8fafc;
            --surface: #ffffff;
            --surface-alt: #f1f5f9;
            --border: #d7e0ea;
            --text: #0f172a;
            --muted: #475569;
            --accent: #0f766e;
            --accent-soft: #d7f3ee;
            --link: #075985;
            --shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
          }}
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            background:
              linear-gradient(180deg, #f7fbff 0%, #f8fafc 42%, #eef6f5 100%);
            color: var(--text);
            font-family: "Segoe UI", "Aptos", ui-sans-serif, system-ui, sans-serif;
          }}
          main {{
            width: min(920px, calc(100vw - 32px));
            margin: 0 auto;
            padding: 48px 0 72px;
          }}
          h1 {{
            margin: 0 0 12px;
            font-size: clamp(2.2rem, 4vw, 3.6rem);
            line-height: 1;
          }}
          p {{
            margin: 0;
            line-height: 1.6;
          }}
          a {{
            color: var(--link);
            text-decoration: none;
          }}
          a:hover {{
            text-decoration: underline;
          }}
          .hero {{
            padding: 0 0 28px;
          }}
          .eyebrow {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 16px;
            color: var(--muted);
            font-size: 0.9rem;
            letter-spacing: 0;
          }}
          .eyebrow::before {{
            content: "";
            width: 26px;
            height: 2px;
            background: linear-gradient(90deg, #0f766e, #0ea5e9);
            border-radius: 999px;
          }}
          .subcopy {{
            max-width: 760px;
            color: var(--muted);
            font-size: 1.02rem;
          }}
          .links {{
            display: flex;
            flex-wrap: wrap;
            gap: 14px;
            margin-top: 18px;
            color: var(--muted);
            font-size: 0.95rem;
          }}
          .panel {{
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(215, 224, 234, 0.95);
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
            border-radius: 8px;
            padding: 22px;
          }}
          #search-form {{
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 12px;
            margin-top: 18px;
          }}
          #query-input {{
            width: 100%;
            padding: 15px 16px;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--surface);
            color: var(--text);
            font: inherit;
            outline: none;
            transition: border-color 140ms ease, box-shadow 140ms ease;
          }}
          #query-input:focus {{
            border-color: #0ea5e9;
            box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.12);
          }}
          #search-button {{
            padding: 15px 20px;
            border: 0;
            border-radius: 8px;
            background: linear-gradient(135deg, #0f766e 0%, #0ea5e9 100%);
            color: white;
            font: inherit;
            font-weight: 600;
            cursor: pointer;
            min-width: 120px;
            transition: transform 140ms ease, opacity 140ms ease;
          }}
          #search-button:hover:enabled {{
            transform: translateY(-1px);
          }}
          #search-button:disabled {{
            opacity: 0.7;
            cursor: wait;
          }}
          .examples {{
            margin-top: 18px;
          }}
          .examples h2,
          .results-head h2 {{
            margin: 0 0 10px;
            font-size: 0.88rem;
            text-transform: uppercase;
            color: var(--muted);
            letter-spacing: 0.08em;
          }}
          .example-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
          }}
          .example-chip {{
            padding: 10px 14px;
            border: 1px solid var(--border);
            border-radius: 999px;
            background: var(--surface);
            color: var(--text);
            font: inherit;
            cursor: pointer;
            transition: background 140ms ease, border-color 140ms ease, transform 140ms ease;
          }}
          .example-chip:hover {{
            background: var(--surface-alt);
            border-color: #aac4d6;
            transform: translateY(-1px);
          }}
          #status {{
            min-height: 24px;
            margin-top: 16px;
            color: var(--muted);
            font-size: 0.94rem;
          }}
          .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: var(--surface-alt);
            border: 1px solid var(--border);
          }}
          .status-pill::before {{
            content: "";
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent);
          }}
          .status-pill.error {{
            background: #fff1f2;
            border-color: #fecdd3;
            color: #9f1239;
          }}
          .status-pill.error::before {{
            background: #e11d48;
          }}
          #results {{
            display: grid;
            gap: 14px;
            margin-top: 20px;
          }}
          .results-head {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 14px;
            margin-top: 10px;
          }}
          .results-meta {{
            color: var(--muted);
            font-size: 0.95rem;
          }}
          .card {{
            border: 1px solid var(--border);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.92);
            padding: 18px;
            box-shadow: 0 14px 36px rgba(15, 23, 42, 0.05);
          }}
          .card-top {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 14px;
          }}
          .card h3 {{
            margin: 0;
            font-size: 1.08rem;
            line-height: 1.35;
          }}
          .badge {{
            display: inline-flex;
            align-items: center;
            padding: 6px 10px;
            border-radius: 999px;
            background: var(--accent-soft);
            color: #0f766e;
            font-size: 0.82rem;
            font-weight: 600;
            white-space: nowrap;
          }}
          .metrics {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin-top: 14px;
          }}
          .metric {{
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--surface-alt);
            padding: 10px 12px;
          }}
          .metric-label {{
            display: block;
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 4px;
          }}
          .metric-value {{
            font-size: 0.98rem;
            font-weight: 600;
          }}
          .explanation {{
            margin-top: 14px;
            color: var(--muted);
            font-size: 0.95rem;
          }}
          .empty {{
            border: 1px dashed #cbd5e1;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.68);
            padding: 18px;
            color: var(--muted);
          }}
          @media (max-width: 720px) {{
            main {{
              width: min(100vw - 24px, 920px);
              padding: 28px 0 48px;
            }}
            #search-form {{
              grid-template-columns: 1fr;
            }}
            #search-button {{
              width: 100%;
            }}
            .card-top,
            .results-head {{
              flex-direction: column;
              align-items: flex-start;
            }}
            .metrics {{
              grid-template-columns: 1fr;
            }}
          }}
        </style>
      </head>
      <body>
        <main>
          <section class="hero">
            <div class="eyebrow">Public portfolio demo</div>
            <h1>Hybrid Retail Search API</h1>
            <p class="subcopy">Search the demo catalog with a lightweight hybrid stack: lexical matching, semantic retrieval, and live score fusion. The production system used private retail data; this Space runs a sanitized public sample.</p>
            <div class="links">
              <a href="/docs">Open API docs</a>
            </div>
          </section>

          <section class="panel">
            <form id="search-form">
              <input id="query-input" name="query" type="text" placeholder="Try sarapan, kopi, sampho, makanan kucing..." autocomplete="off" />
              <button id="search-button" type="submit">Search</button>
            </form>

            <div class="examples">
              <h2>Try an example query</h2>
              <div class="example-list">
                {example_buttons}
              </div>
            </div>

            <div id="status"></div>

            <section id="results" aria-live="polite"></section>
          </section>
        </main>

        <script>
          const form = document.getElementById("search-form");
          const input = document.getElementById("query-input");
          const button = document.getElementById("search-button");
          const results = document.getElementById("results");
          const status = document.getElementById("status");
          const chips = Array.from(document.querySelectorAll(".example-chip"));

          function escapeHtml(value) {{
            return String(value)
              .replaceAll("&", "&amp;")
              .replaceAll("<", "&lt;")
              .replaceAll(">", "&gt;")
              .replaceAll('"', "&quot;")
              .replaceAll("'", "&#39;");
          }}

          function setStatus(message, type = "info") {{
            const className = type === "error" ? "status-pill error" : "status-pill";
            status.innerHTML = `<span class="${{className}}">${{escapeHtml(message)}}</span>`;
          }}

          function renderEmpty(message) {{
            results.innerHTML = `<div class="empty">${{escapeHtml(message)}}</div>`;
          }}

          function renderResults(payload) {{
            if (!payload.results.length) {{
              renderEmpty("No products matched this query in the public demo catalog.");
              setStatus(`No results for "${{payload.query}}"`, "info");
              return;
            }}

            const cards = payload.results.map((item) => `
              <article class="card">
                <div class="card-top">
                  <div>
                    <h3>${{escapeHtml(item.sku_name)}}</h3>
                  </div>
                  <span class="badge">${{escapeHtml(item.source)}}</span>
                </div>
                <div class="metrics">
                  <div class="metric">
                    <span class="metric-label">Final score</span>
                    <span class="metric-value">${{item.final_score}}</span>
                  </div>
                  <div class="metric">
                    <span class="metric-label">BM25</span>
                    <span class="metric-value">${{item.bm25_score}}</span>
                  </div>
                  <div class="metric">
                    <span class="metric-label">Semantic</span>
                    <span class="metric-value">${{item.semantic_score}}</span>
                  </div>
                </div>
                <p class="explanation">${{escapeHtml(item.explanation)}}</p>
              </article>
            `).join("");

            results.innerHTML = `
              <div class="results-head">
                <div>
                  <h2>Results</h2>
                  <div class="results-meta">${{payload.results.length}} shown from ${{payload.total_found}} matches</div>
                </div>
                <div class="results-meta">${{payload.latency_ms}} ms</div>
              </div>
              ${{cards}}
            `;
            setStatus(`Showing results for "${{payload.query}}"`, "info");
          }}

          async function runSearch(query) {{
            const trimmed = query.trim();
            if (!trimmed) {{
              setStatus("Enter a query before searching.", "error");
              renderEmpty("Search results will appear here.");
              return;
            }}
            if (trimmed.length < 2) {{
              setStatus("Query must contain at least 2 characters.", "error");
              renderEmpty("Use at least 2 characters to run a search.");
              return;
            }}

            button.disabled = true;
            setStatus(`Searching for "${{trimmed}}"...`, "info");

            try {{
              const response = await fetch('/search?query=' + encodeURIComponent(trimmed) + '&top_k=5');
              const payload = await response.json();

              if (!response.ok) {{
                const message = payload.detail || "Search request failed.";
                setStatus(typeof message === "string" ? message : "Search request failed.", "error");
                renderEmpty("Please adjust the query and try again.");
                return;
              }}

              renderResults(payload);
            }} catch (error) {{
              setStatus("Unable to reach the search API right now.", "error");
              renderEmpty("Please retry in a moment.");
            }} finally {{
              button.disabled = false;
            }}
          }}

          form.addEventListener("submit", async (event) => {{
            event.preventDefault();
            await runSearch(input.value);
          }});

          chips.forEach((chip) => {{
            chip.addEventListener("click", async () => {{
              input.value = chip.dataset.query || "";
              input.focus();
              await runSearch(input.value);
            }});
          }});
        </script>
      </body>
    </html>
    """


@app.get("/favicon.ico")
def favicon():
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


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
    return EXAMPLE_QUERIES


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
