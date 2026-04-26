from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient


def test_load_products_reads_public_catalog():
    from demo_search.loaders import load_products

    products = load_products(Path("demo_search/data/products.csv"))

    assert len(products) >= 40
    assert products[0].sku_id
    assert products[0].sku_name
    assert products[0].class_name


def test_artifact_builder_creates_normalized_embeddings(tmp_path):
    from demo_search.artifacts import build_artifacts
    from demo_search.loaders import load_products

    products = load_products(Path("demo_search/data/products.csv"))
    artifacts = build_artifacts(products, tmp_path)

    assert artifacts.embeddings_path.exists()
    assert artifacts.metadata_path.exists()

    embeddings = np.load(artifacts.embeddings_path)
    norms = np.linalg.norm(embeddings, axis=1)

    assert embeddings.shape[0] == len(products)
    assert embeddings.shape[1] >= 16
    assert np.allclose(norms, 1.0)


def test_hybrid_retriever_handles_conceptual_query():
    from demo_search.retrieval import HybridDemoSearch

    searcher = HybridDemoSearch.from_paths(
        catalog_path=Path("demo_search/data/products.csv"),
        artifact_dir=Path("demo_search/artifacts"),
    )

    response = searcher.search("sarapan", top_k=5)
    names = " ".join(result.sku_name.lower() for result in response.results)

    assert response.retrieval_mode == "bm25+semantic"
    assert response.results
    assert any(word in names for word in ["oat", "sereal", "cereal", "roti", "granola"])


def test_fastapi_search_endpoint_returns_demo_metadata():
    from demo_search.api import app

    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    response = client.get("/search", params={"query": "kopi", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["demo_mode"] is True
    assert body["retrieval_mode"] == "bm25+semantic"
    assert body["total_found"] > 0
    assert len(body["results"]) <= 3


def test_root_page_contains_search_ui():
    from demo_search.api import app

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert 'id="search-form"' in html
    assert 'id="query-input"' in html
    assert 'id="search-button"' in html
    assert 'id="results"' in html
    assert "Try an example query" in html
    assert "fetch('/search" in html


def test_root_page_validates_short_query_client_side():
    from demo_search.api import app

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "trimmed.length < 2" in response.text


def test_search_filters_zero_score_results():
    from demo_search.retrieval import HybridDemoSearch

    searcher = HybridDemoSearch.from_paths(
        catalog_path=Path("demo_search/data/products.csv"),
        artifact_dir=Path("demo_search/artifacts"),
    )

    response = searcher.search("obat nyamuk", top_k=5)

    assert response.results
    assert all(result.final_score > 0 for result in response.results)


def test_favicon_route_exists():
    from demo_search.api import app

    client = TestClient(app)
    response = client.get("/favicon.ico")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")
