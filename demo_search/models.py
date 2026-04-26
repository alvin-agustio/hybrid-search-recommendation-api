from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Product:
    sku_id: int
    sku_name: str
    division_name: str
    dept_name: str
    class_name: str
    subclass_name: str
    group_name: str


@dataclass(frozen=True)
class SearchResult:
    sku_id: int
    sku_name: str
    final_score: float
    bm25_score: float
    semantic_score: float
    source: str
    explanation: str


@dataclass(frozen=True)
class SearchResponse:
    query: str
    demo_mode: bool
    retrieval_mode: str
    results: List[SearchResult]
    total_found: int
    latency_ms: float
    notes: str

