"""
FastAPI Search Engine API with RFM Personalization.
Migrated from app_personalized.py (Streamlit).

Run: uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

# Fix OpenMP conflict on Windows (must be before other imports)
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime

# Add project root to path (must be before local imports)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import CLICKHOUSE_CONFIG
from training.loader import load_products
from inference.bm25 import BM25Baseline
from inference.hybrid_search import HybridSearchPipeline
from inference.segment_fusion import SegmentFusion
from inference.semantic_search import DirectSemanticSearcher
from models.search_model import SearchModel

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 200
MAX_TOP_K = 100

# Query aliases normalize common retail intents before retrieval.
QUERY_ALIASES = {
    "susu formula": "susu bayi dan anak",
    "formula anak": "susu bayi dan anak",
    "penyegar": "larutan",
    "cuci piring": "sabun cuci piring",
}


def sanitize_query(query: str) -> str:
    """Remove emoji, symbols, and non-printable characters from query."""
    import re

    # Keep only alphanumeric, spaces, and basic punctuation
    cleaned = re.sub(r"[^\w\s\-.,?!]+", "", query, flags=re.UNICODE)
    # Collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()

    for source_term, target_term in QUERY_ALIASES.items():
        if source_term in cleaned:
            cleaned = cleaned.replace(source_term, target_term)

    return cleaned if cleaned else query  # Fallback to original if empty after cleaning


def validate_member_id(member_id: str) -> bool:
    """Validate member ID is digits only."""
    return member_id.isdigit()


class ProductResult(BaseModel):
    """Single product in search results."""

    sku_id: int
    sku_name: str
    final_score: float
    search_score: float
    segment_score: float
    boosted: bool
    source: str


class SearchResponse(BaseModel):
    """Search response with results and metadata."""

    query: str
    member_id: Optional[str]
    segment: Optional[str]
    product_class: Optional[str]
    results: List[ProductResult]
    total_found: int
    latency_ms: float


class MemberResponse(BaseModel):
    """Member segment lookup response."""

    member_id: str
    segment: Optional[str]
    found: bool


# ============================================================================
# APP STATE (replaces global variables)
# ============================================================================


class AppState:
    """Application state container."""

    def __init__(self):
        self.pipeline = None
        self.fusion = None
        self.df = None
        self.sku_lookup = {}
        self.start_time = time.time()


app_state = AppState()


# ============================================================================
# STARTUP / SHUTDOWN (Lifespan)
# ============================================================================


def build_search_components(_df) -> HybridSearchPipeline:
    """Build HybridSearchPipeline with proper SKU alignment (same as app.py)."""
    import json

    # BM25
    bm25 = BM25Baseline()
    bm25.build_index(_df)

    # Load embeddings and models
    checkpoint_dir = "runtime/checkpoints/production_stage1"  # Stable production path
    embeddings_path = f"{checkpoint_dir}/product_embeddings.npy"
    semantic_searcher = None
    embeddings = None

    if os.path.exists(embeddings_path):
        raw_embeddings = np.load(embeddings_path)
        logger.info("Raw embeddings shape: %s", raw_embeddings.shape)

        # Load category vocab from checkpoint (consistent with training)
        vocab_path = f"{checkpoint_dir}/category_vocab.json"
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                category_vocab = json.load(f)
            logger.info("Loaded category vocab from checkpoint")
        else:
            # Fallback: build vocab using same function as training
            from training.loader import get_category_vocab

            category_vocab = get_category_vocab(_df)
            logger.warning("Category vocab not found in checkpoint, rebuilding")

        # Calculate vocab sizes (consistent with training)
        vocab_sizes = {cat: len(v) for cat, v in category_vocab.items()}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        search_model = SearchModel(
            category_vocab_sizes=vocab_sizes, use_dora=True, embedding_dim=256
        )

        # Load model weights
        model_path = f"{checkpoint_dir}/best_model.pt"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            search_model.load_state_dict(state_dict, strict=False)

        search_model.to(device)
        search_model.eval()
        search_model.set_category_vocab(category_vocab)

        # Load embedding metadata for SKU-based alignment
        metadata_path = f"{checkpoint_dir}/embedding_metadata.json"

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            embedding_skus = metadata.get("sku_order", [])
            sku_to_emb_idx = {int(sku): idx for idx, sku in enumerate(embedding_skus)}

            # Find missing products
            df_skus = _df["sku_id"].tolist()
            missing_indices = []
            aligned_embeddings = []

            for i, sku in enumerate(df_skus):
                sku_int = int(sku)
                if sku_int in sku_to_emb_idx:
                    aligned_embeddings.append(raw_embeddings[sku_to_emb_idx[sku_int]])
                else:
                    aligned_embeddings.append(None)  # Placeholder
                    missing_indices.append(i)

            # Encode missing products with Stage 1 model
            if missing_indices:
                logger.info(
                    "Encoding %d missing products with Stage 1...",
                    len(missing_indices),
                )
                missing_df = _df.iloc[missing_indices]

                category_values = {
                    cat: missing_df[cat].fillna("UNKNOWN").tolist()
                    for cat in [
                        "division_name",
                        "dept_name",
                        "class_name",
                        "subclass_name",
                        "group_name",
                    ]
                }

                with torch.no_grad():
                    missing_embeddings = search_model.encode_products(
                        product_names=missing_df["sku_name"].fillna("").tolist(),
                        category_values=category_values,
                        batch_size=32,
                        device=device,
                        show_progress=False,
                    )

                # Fill in missing embeddings
                for j, idx in enumerate(missing_indices):
                    aligned_embeddings[idx] = missing_embeddings[j]

                logger.info("Encoded %d missing products", len(missing_indices))

            embeddings = np.array(aligned_embeddings, dtype=np.float32)
            matched_count = len(df_skus) - len(missing_indices)
            logger.info(
                "Final embeddings: %d pre-computed, %d newly encoded",
                matched_count,
                len(missing_indices),
            )
        else:
            # Fallback: use raw embeddings directly (may cause index errors if mismatch)
            logger.warning("No metadata file, using raw embeddings directly")
            embeddings = raw_embeddings.astype(np.float32)

        # Load Stage 2 Query Encoder (aligned with app_personalized)
        from models.enhanced_query_encoder import EnhancedQueryEncoder

        query_encoder_path = (
            "runtime/checkpoints/production_stage2"  # Stable production path
        )

        if os.path.exists(query_encoder_path):
            logger.info("Loading Stage 2 Query Encoder...")
            query_encoder, _, _ = EnhancedQueryEncoder.load(
                query_encoder_path, device=device
            )
            logger.info("EnhancedQueryEncoder loaded")
        else:
            # Fallback to Stage 1
            logger.warning("Stage 2 not found, using Stage 1 fallback")

            class SimpleWrapper:
                def __init__(self, model):
                    self.model = model

                def encode(self, query, device=None):
                    return self.model.encode_query(query, device=device)

                def to(self, device):
                    self.model.to(device)
                    return self

                def eval(self):
                    self.model.eval()

            query_encoder = SimpleWrapper(search_model)

        semantic_searcher = DirectSemanticSearcher(
            encoder=query_encoder,
            embeddings=embeddings,
            sku_ids=_df["sku_id"].tolist(),
            device=device,
        )

    # Create pipeline with reranking enabled
    _pipeline = HybridSearchPipeline(
        bm25=bm25,
        semantic_searcher=semantic_searcher,
    )

    # Store df reference for result lookup
    _pipeline.df = _df

    return _pipeline


def build_segment_fusion() -> SegmentFusion:
    """Build SegmentFusion with direct ClickHouse connection (no cache)."""
    _fusion = SegmentFusion()

    logger.info("Fetching segment data from ClickHouse...")
    try:
        _fusion.refresh_from_clickhouse(
            host=CLICKHOUSE_CONFIG["host"],
            port=CLICKHOUSE_CONFIG["port"],
            username=CLICKHOUSE_CONFIG["user"],
            password=CLICKHOUSE_CONFIG["password"],
            database="default",
        )
        logger.info("Segment data loaded from ClickHouse")
    except Exception as exc:  # noqa: BLE001
        logger.warning("ClickHouse connection failed: %s", exc)
        logger.warning("Personalization will be disabled")

    return _fusion


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup and shutdown events with caching."""
    from datetime import timedelta

    logger.info("Starting API server...")

    # Dynamic window: last 365 days
    end_date_obj = datetime.now()
    start_date_obj = end_date_obj - timedelta(days=365)
    end_date = end_date_obj.strftime("%Y-%m-%d")
    start_date = start_date_obj.strftime("%Y-%m-%d")

    logger.info("Product window: %s to %s (last 365 days)", start_date, end_date)

    # Load products from ClickHouse (no cache)
    logger.info(
        "Fetching products from ClickHouse (%s to %s)...",
        start_date,
        end_date,
    )
    app_state.df = load_products(start_date=start_date, end_date=end_date)
    logger.info("Loaded %s products from ClickHouse", f"{len(app_state.df):,}")

    logger.info("Building SKU lookup...")
    app_state.sku_lookup = {p.sku_id: p for p in app_state.df.itertuples()}
    logger.info("Built lookup for %s SKUs", f"{len(app_state.sku_lookup):,}")

    logger.info("Building search pipeline...")
    app_state.pipeline = build_search_components(app_state.df)
    logger.info("Pipeline ready")

    logger.info("Loading segment data...")
    app_state.fusion = build_segment_fusion()
    if app_state.fusion.preferences:
        logger.info(
            "Segments: %d, Members: %d",
            len(app_state.fusion.preferences),
            len(app_state.fusion.member_segments),
        )
    else:
        logger.warning("No segments loaded - personalization disabled")

    logger.info("API ready!")
    yield

    logger.info("Shutting down...")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="RetailCo Personalized Search API",
    description="Hybrid Search (BM25 + Semantic) with RFM Segmentation",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ENDPOINTS
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "products_loaded": len(app_state.df) if app_state.df is not None else 0,
        "segments_loaded": len(app_state.fusion.preferences) if app_state.fusion else 0,
        "members_loaded": (
            len(app_state.fusion.member_segments) if app_state.fusion else 0
        ),
    }


@app.get("/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., min_length=1, max_length=MAX_QUERY_LENGTH),
    member_id: Optional[str] = None,
    top_k: int = Query(default=10, ge=1, le=MAX_TOP_K),
):
    """
    Main search endpoint with optional personalization.

    - Without member_id: Returns base search results
    - With member_id: Returns personalized results using HYBRID preferences

    Args:
        query: Search query (1-200 chars)
        member_id: Optional member ID for personalization
        top_k: Number of results (1-100)
    """

    if app_state.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Sanitize query - remove emoji and special characters
    clean_query = sanitize_query(query)
    if not clean_query:
        raise HTTPException(
            status_code=400, detail="Query contains only invalid characters"
        )

    # Base search - returns SearchResult wrapper (with products, latency)
    search_result = app_state.pipeline.search(
        query=clean_query,
        top_k=top_k * 2,
    )
    products_list = search_result.products

    # Recover low-confidence queries with explicit spelling correction.
    # This keeps the API responsive for typo-heavy queries without forcing
    # spelling correction into the main retrieval path.
    if not products_list or products_list[0].get("fused_score", 0) < 0.60:
        logger.info(
            "Triggering SymSpell rescue for low-confidence query: %s",
            clean_query,
        )
        # Call explicit correct_spelling from underlying BM25 instance
        rescued_query = app_state.pipeline.bm25.correct_spelling(clean_query)
        if rescued_query != clean_query:
            logger.info("Rescued query: %s -> %s", clean_query, rescued_query)
            search_result = app_state.pipeline.search(
                query=rescued_query,
                top_k=top_k * 2,
            )
            products_list = search_result.products

    # 404 if no results found
    if not products_list:
        raise HTTPException(status_code=404, detail="No results found")

    # Hardcoded weights (no longer exposed in API)
    search_weight = 0.5  # Balance between search and segment
    subclass_weight = 0.7  # Balance between subclass and class

    segment = None
    product_context = None
    personalized_results = None
    segment_weight = 1.0 - search_weight
    class_weight = 1.0 - subclass_weight

    # Personalization with HYBRID preferences (if member_id provided)
    if member_id and app_state.fusion:
        segment = app_state.fusion.get_member_segment(member_id)
        if segment:
            # Get class/subclass contexts from top-3 results
            top_results = products_list[:3] if products_list else []
            # Lookup product info from df for class/subclass
            contexts = []
            for r in top_results:
                sku_id = r["sku_id"]
                product_row = app_state.df[app_state.df["sku_id"] == sku_id]
                if not product_row.empty:
                    class_name = product_row["class_name"].iloc[0]
                    subclass_name = (
                        product_row["subclass_name"].iloc[0]
                        if "subclass_name" in product_row.columns
                        else ""
                    )
                    if class_name:
                        contexts.append((class_name, subclass_name))

            # Aggregate HYBRID preferences
            segment_prefs = {}
            for class_name, subclass_name in contexts:
                hybrid_prefs = app_state.fusion.get_hybrid_preferences(
                    segment_id=segment,
                    class_name=class_name,
                    subclass_name=subclass_name or "",
                    subclass_weight=subclass_weight,
                    class_weight=class_weight,
                )
                # Merge: keep highest score if duplicate SKU
                for sku, score in hybrid_prefs.items():
                    if sku not in segment_prefs or score > segment_prefs[sku]:
                        segment_prefs[sku] = score

            product_context = (
                ", ".join(set(f"{c}/{s}" for c, s in contexts if c))
                if contexts
                else None
            )

            if segment_prefs:
                app_state.fusion.search_weight = search_weight
                app_state.fusion.segment_weight = segment_weight
                search_results = [
                    (r["sku_id"], r["fused_score"]) for r in products_list
                ]
                personalized_results = app_state.fusion.fuse(
                    search_results, segment_prefs, top_k
                )

    # Build response
    products = []
    if personalized_results:
        for res in personalized_results[:top_k]:
            product = app_state.sku_lookup.get(res.sku_id)
            if product:
                products.append(
                    ProductResult(
                        sku_id=res.sku_id,
                        sku_name=product.sku_name[:100] if product.sku_name else "N/A",
                        final_score=round(res.final_score, 4),
                        search_score=round(res.search_score, 4),
                        segment_score=round(res.segment_score, 4),
                        boosted=res.segment_score > 0,
                        source=res.source,
                    )
                )
    else:
        for r in products_list[:top_k]:
            # Products list is already dict with sku_name included
            products.append(
                ProductResult(
                    sku_id=r["sku_id"],
                    sku_name=r["sku_name"][:100] if r["sku_name"] else "N/A",
                    final_score=round(r["fused_score"], 4),
                    search_score=round(r["bm25_score"], 4),
                    segment_score=0,
                    boosted=False,
                    source="hybrid",
                )
            )

    # Use latency from pipeline
    latency = search_result.latency_ms

    return SearchResponse(
        query=query,
        member_id=member_id,
        segment=segment,
        product_class=product_context,
        results=products,
        total_found=len(products_list),
        latency_ms=round(latency, 2),
    )


@app.get("/member/{member_id}", response_model=MemberResponse)
async def get_member(member_id: str):
    """Check member segment."""
    # Validate member ID format (digits only)
    if not validate_member_id(member_id):
        raise HTTPException(status_code=400, detail="Member ID must be digits only")

    if app_state.fusion is None:
        raise HTTPException(status_code=503, detail="Segment data not loaded")

    segment = app_state.fusion.get_member_segment(member_id)
    return MemberResponse(
        member_id=member_id,
        segment=segment,
        found=segment is not None,
    )


@app.post("/reload")
async def reload_models_and_data():
    """Reload models and inference data without restarting API.

    Use this after training completes to load new models.
    """
    from datetime import timedelta

    start_time = time.time()
    logger.info("Reload requested - starting...")

    try:
        # Step 1: Reload inference data
        end_date_obj = datetime.now()
        start_date_obj = end_date_obj - timedelta(days=365)
        end_date = end_date_obj.strftime("%Y-%m-%d")
        start_date = start_date_obj.strftime("%Y-%m-%d")

        logger.info("Reloading products: %s to %s", start_date, end_date)
        new_df = load_products(start_date=start_date, end_date=end_date)
        logger.info("Loaded %s products", f"{len(new_df):,}")

        # Step 2: Rebuild search pipeline (loads new models)
        logger.info("Rebuilding search pipeline with new models...")
        new_pipeline = build_search_components(new_df)
        logger.info("Pipeline rebuilt")

        # Step 3: Reload segment data
        logger.info("Reloading segment data...")
        new_fusion = build_segment_fusion()
        logger.info("Segment data reloaded")

        # Step 4: Atomic swap (replace old with new)
        app_state.df = new_df
        app_state.sku_lookup = {p.sku_id: p for p in new_df.itertuples()}
        app_state.pipeline = new_pipeline
        app_state.fusion = new_fusion

        duration = time.time() - start_time
        logger.info("Reload complete in %.1fs", duration)

        return {
            "status": "success",
            "message": "Models and data reloaded successfully",
            "products_loaded": len(new_df),
            "duration_seconds": round(duration, 1),
        }

    except Exception as exc:  # noqa: BLE001
        logger.error("Reload failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Reload failed: {str(exc)}",
        ) from exc


# ============================================================================
# RUN (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
