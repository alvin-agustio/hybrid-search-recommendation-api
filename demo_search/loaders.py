from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from .models import Product


REQUIRED_COLUMNS = [
    "sku_id",
    "sku_name",
    "division_name",
    "dept_name",
    "class_name",
    "subclass_name",
    "group_name",
]


def load_products(path: Path) -> List[Product]:
    """Load the public demo catalog from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Public catalog not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Catalog missing required columns: {missing}")

        products = [
            Product(
                sku_id=int(row["sku_id"]),
                sku_name=row["sku_name"].strip(),
                division_name=row["division_name"].strip(),
                dept_name=row["dept_name"].strip(),
                class_name=row["class_name"].strip(),
                subclass_name=row["subclass_name"].strip(),
                group_name=row["group_name"].strip(),
            )
            for row in reader
        ]

    if not products:
        raise ValueError(f"Public catalog is empty: {path}")

    return products

