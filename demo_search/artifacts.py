from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .models import Product


EMBEDDING_DIM = 2048

CONCEPT_TERMS = {
    "sarapan": ["sereal", "cereal", "oat", "granola", "roti", "selai", "susu", "kopi"],
    "cemilan": ["snack", "keripik", "biskuit", "wafer", "kacang", "popcorn", "cokelat"],
    "kopi": ["kopi", "coffee", "latte", "espresso", "kapal", "torabika"],
    "susu bayi": ["susu", "bayi", "anak", "formula", "dancow", "bebelac"],
    "shampoo": ["shampoo", "conditioner", "rambut", "hair"],
    "sampho": ["shampoo", "conditioner", "rambut", "hair"],
    "obat nyamuk": ["nyamuk", "aerosol", "repellent", "mosquito"],
    "makanan kucing": ["kucing", "cat", "pet", "whiskas", "tuna", "salmon"],
}


@dataclass(frozen=True)
class ArtifactPaths:
    embeddings_path: Path
    metadata_path: Path


def tokenize(text: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) > 1]


def text_for_product(product: Product) -> str:
    return " ".join(
        [
            product.sku_name,
            product.division_name,
            product.dept_name,
            product.class_name,
            product.subclass_name,
            product.group_name,
        ]
    )


def expand_query(query: str) -> str:
    lowered = query.lower()
    expanded = []
    for concept, related_terms in CONCEPT_TERMS.items():
        if concept in lowered:
            expanded.extend(related_terms)
    if expanded:
        return " ".join(expanded)
    expanded = [query]
    return " ".join(expanded)


def build_vocabulary(texts: Iterable[str]) -> dict[str, int]:
    tokens = set()
    for text in texts:
        tokens.update(tokenize(expand_query(text)))
    for concept, related_terms in CONCEPT_TERMS.items():
        tokens.update(tokenize(concept))
        tokens.update(related_terms)
    return {token: index for index, token in enumerate(sorted(tokens))}


def embed_text(
    text: str,
    dim: int = EMBEDDING_DIM,
    vocabulary: dict[str, int] | None = None,
) -> np.ndarray:
    if vocabulary is not None:
        dim = max(len(vocabulary), 1)
    vector = np.zeros(dim, dtype=np.float32)
    for token in tokenize(expand_query(text)):
        if vocabulary is None:
            index = hash(token) % dim
        else:
            if token not in vocabulary:
                continue
            index = vocabulary[token]
        vector[index] += 1.0

    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def build_product_embeddings(
    products: Iterable[Product],
    vocabulary: dict[str, int] | None = None,
) -> np.ndarray:
    embeddings = [
        embed_text(text_for_product(product), vocabulary=vocabulary)
        for product in products
    ]
    return np.vstack(embeddings).astype(np.float32)


def build_artifacts(products: List[Product], output_dir: Path) -> ArtifactPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "product_embeddings.npy"
    metadata_path = output_dir / "embedding_metadata.json"

    vocabulary = build_vocabulary(text_for_product(product) for product in products)
    embeddings = build_product_embeddings(products, vocabulary=vocabulary)
    np.save(embeddings_path, embeddings)

    metadata = {
        "embedding_dim": len(vocabulary),
        "sku_order": [product.sku_id for product in products],
        "strategy": "public_demo_vocab_vectorizer",
        "vocabulary": vocabulary,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return ArtifactPaths(embeddings_path=embeddings_path, metadata_path=metadata_path)
