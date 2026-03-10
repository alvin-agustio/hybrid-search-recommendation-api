"""
Main configuration file for application.
Manages database connections.
"""

import os
from dotenv import load_dotenv
from typing import Literal, Optional, Dict, Any

# Database connection settings
load_dotenv()

# Load ClickHouse config if available, otherwise use dummy values for parquet mode
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST")
CLICKHOUSE_PORT = os.getenv("CLICKHOUSE_PORT", "8123")

if CLICKHOUSE_HOST:
    # ClickHouse is available - use it
    CLICKHOUSE_CONFIG = {
        "host": CLICKHOUSE_HOST,
        "port": int(CLICKHOUSE_PORT or "8123"),
        "user": os.getenv("CLICKHOUSE_USER", "default"),
        "password": os.getenv("CLICKHOUSE_PASSWORD", "default"),
    }
else:
    # VM mode: No ClickHouse - use default dummy config
    CLICKHOUSE_CONFIG = {
        "host": "localhost",
        "port": 8123,
        "user": "default",
        "password": "default",
    }

# Simple validation for ClickHouse mode
if not CLICKHOUSE_HOST:
    missing = [k for k in CLICKHOUSE_CONFIG if not CLICKHOUSE_CONFIG[k]]
    if missing:
        raise ValueError(f"Missing ClickHouse config keys: {missing}")
else:
    # For parquet-only mode, we skip validation
    pass
