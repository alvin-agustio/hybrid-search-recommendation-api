from __future__ import annotations

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from demo_search.artifacts import build_artifacts  # noqa: E402
from demo_search.loaders import load_products  # noqa: E402


def main() -> None:
    catalog_path = BASE_DIR / "demo_search" / "data" / "products.csv"
    artifact_dir = BASE_DIR / "demo_search" / "artifacts"
    products = load_products(catalog_path)
    artifacts = build_artifacts(products, artifact_dir)
    print(f"Wrote {artifacts.embeddings_path}")
    print(f"Wrote {artifacts.metadata_path}")


if __name__ == "__main__":
    main()
