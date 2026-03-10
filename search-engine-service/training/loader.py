"""Data loader for ClickHouse product data with caching."""

import logging
import pandas as pd
import clickhouse_connect
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLICKHOUSE_CONFIG

logger = logging.getLogger(__name__)

# Constants
TRANSACTION_TABLE = "commerce_db.`transactions`"
CATEGORY_COLUMNS = [
    "division_name",
    "dept_name",
    "class_name",
    "subclass_name",
    "group_name",
]


def get_client():
    """Create ClickHouse client connection."""
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_CONFIG["host"],
        port=CLICKHOUSE_CONFIG["port"],
        username=CLICKHOUSE_CONFIG["user"],
        password=CLICKHOUSE_CONFIG["password"],
    )


def load_products(start_date: str, end_date: str, cache_path: str = None) -> pd.DataFrame:
    """
    Load distinct products with categories from ClickHouse or cache.

    Priority: ClickHouse First → Parquet Fallback

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        cache_path: Optional path to cached parquet file (fallback if ClickHouse fails)

    Returns:
        DataFrame with product data
    """
    # Try ClickHouse first
    try:
        logger.info("Querying ClickHouse for products (%s to %s)...", start_date, end_date)
        client = get_client()

        # Build query for distinct products with categories
        query = f"""
        SELECT
            sku_id,
            sku_name,
            division_name,
            dept_name,
            class_name,
            subclass_name,
            group_name
        FROM {TRANSACTION_TABLE}
        WHERE transaction_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY sku_id, sku_name, division_name, dept_name, class_name, subclass_name, group_name
        ORDER BY sku_id
        """

        # Execute query and convert to DataFrame
        result = client.query(query)
        df = pd.DataFrame(result.result_rows, columns=result.column_names)

        # Clean up text columns (fill nulls, strip whitespace)
        text_columns = ["sku_name"] + CATEGORY_COLUMNS
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.strip()

        logger.info("Loaded %d products from ClickHouse", len(df))
        return df

    except Exception as e:
        logger.warning("ClickHouse failed: %s. Trying cache...", e)

        # Fallback to parquet cache
        if cache_path and os.path.exists(cache_path):
            logger.info("Loading products from cache: %s", cache_path)
            try:
                df = pd.read_parquet(cache_path)
                logger.info("Loaded %d products from cache", len(df))

                # Clean up text columns
                text_columns = ["sku_name"] + CATEGORY_COLUMNS
                for col in text_columns:
                    if col in df.columns:
                        df[col] = df[col].fillna("").astype(str).str.strip()

                return df
            except Exception as ce:
                logger.error("Failed to load from cache: %s", ce)
                raise RuntimeError(
                    f"Cannot load products: Both ClickHouse and cache failed.\n"
                    f"  ClickHouse error: {e}\n"
                    f"  Cache error: {ce}\n"
                    f"  Cache path: {cache_path}"
                )

        # Both failed
        raise RuntimeError(
            f"Cannot load products: Both ClickHouse and cache failed.\n"
            f"  ClickHouse error: {e}\n"
            f"  Cache path: {cache_path}\n\n"
            f"To create cache, run:\n"
            f"  python training/cache_products.py --start_date {start_date} --end_date {end_date} --output {cache_path}"
        )


def get_category_vocab(df: pd.DataFrame) -> dict:
    """Build category vocabulary with 1-based indexing (0 reserved for padding/unknown)."""
    vocab = {}

    # Build vocab for each category column
    for col in CATEGORY_COLUMNS:
        if col in df.columns:
            # Get unique values
            unique_values = df[col].unique().tolist()

            # Map to 1-based IDs (0 reserved for unknown)
            vocab[col] = {val: idx + 1 for idx, val in enumerate(unique_values)}

    return vocab
