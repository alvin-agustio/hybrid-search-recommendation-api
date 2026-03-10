"""Segment-based personalization fusion with RFM cache."""

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import clickhouse_connect

logger = logging.getLogger(__name__)


@dataclass
class PersonalizedResult:
    """Personalized search result with score breakdown."""

    sku_id: int
    search_score: float  # Raw BM25/fused score
    normalized_search: float  # Normalized 0-1
    segment_score: float  # Already 0-1 (log-decay)
    final_score: float  # Weighted combination
    source: str
    personalized: bool


class SegmentFusion:
    """Weighted fusion + RFM cache management."""

    def __init__(
        self,
        search_weight: float = 0.7,
        segment_weight: float = 0.3,
    ):
        """Initialize fusion with weights."""
        # Store fusion weights
        self.search_weight = search_weight
        self.segment_weight = segment_weight

        # Initialize in-memory cache
        # Format: {segment: {level_key: {sku_id: score}}}
        self.preferences: Dict[str, Dict[str, Dict[int, float]]] = {}
        self.member_segments: Dict[str, str] = {}

    def _get_last_month_dates(self) -> Tuple[str, str, str]:
        """Calculate date range for last fully completed month."""
        # Get today's date
        today = datetime.now().date()

        # Calculate first day of this month
        this_month_first = today.replace(day=1)

        # Calculate last day of last month (day before this month)
        last_month_last = this_month_first - timedelta(days=1)

        # Calculate first day of last month
        last_month_first = last_month_last.replace(day=1)

        # Return (start_date, end_date, month_key)
        return (
            last_month_first.strftime("%Y-%m-%d"),
            last_month_last.strftime("%Y-%m-%d"),
            last_month_first.strftime("%Y-%m-%d"),
        )

    def get_member_segment(self, member_id: str) -> Optional[str]:
        """Get RFM segment for member."""
        return self.member_segments.get(member_id)

    def get_hybrid_preferences(
        self,
        segment_id: str,
        class_name: str,
        subclass_name: str,
        subclass_weight: float = 0.7,
        class_weight: float = 0.3,
        min_subclass_count: int = 5,
    ) -> Dict[int, float]:
        """Get hybrid preferences: 70% subclass + 30% class."""
        # Check if segment exists
        if segment_id not in self.preferences:
            return {}

        # Build keys for both levels
        subclass_key = f"SUBCLASS:{subclass_name}"
        class_key = f"CLASS:{class_name}"

        # Get preferences for both levels
        subclass_prefs = self.preferences[segment_id].get(subclass_key, {})
        class_prefs = self.preferences[segment_id].get(class_key, {})

        # Check if subclass has enough items
        if len(subclass_prefs) >= min_subclass_count:
            # Weighted combination: 70% subclass + 30% class
            combined = {}
            all_skus = set(subclass_prefs.keys()) | set(class_prefs.keys())

            # Calculate weighted score for each SKU
            for sku in all_skus:
                sub_score = subclass_prefs.get(sku, 0) * subclass_weight
                cls_score = class_prefs.get(sku, 0) * class_weight
                combined[sku] = sub_score + cls_score

            return combined
        else:
            # Fallback to class only (subclass too sparse)
            return class_prefs

    def fuse(
        self,
        search_results: List[Tuple[int, float]],
        segment_preferences: Dict[int, float],
        top_k: int = 20,
    ) -> List[PersonalizedResult]:
        """Fuse search results with segment preferences."""
        # Return empty if no search results
        if not search_results:
            return []

        # Extract scores for normalization
        scores = [score for _, score in search_results]
        min_score = min(scores)
        max_score = max(scores)

        results = []

        # Process each search result
        for sku_id, search_score in search_results:
            # Normalize search score to [0, 1]
            if max_score == min_score:
                # All scores equal: give neutral normalized value
                norm_search = 0.5
            else:
                # Min-max normalization to [0, 1]
                norm_search = (search_score - min_score) / (max_score - min_score)

            # Get segment score for this SKU
            segment_score = segment_preferences.get(sku_id, 0.0)

            # Calculate final weighted score
            final_score = (
                self.search_weight * norm_search + self.segment_weight * segment_score
            )

            # Determine source label
            source = "search+segment" if segment_score > 0 else "search"
            personalized = segment_score > 0

            # Add to results
            results.append(
                PersonalizedResult(
                    sku_id=sku_id,
                    search_score=search_score,
                    normalized_search=norm_search,
                    segment_score=segment_score,
                    final_score=final_score,
                    source=source,
                    personalized=personalized,
                )
            )

        # Sort by final score (highest first)
        results.sort(key=lambda x: x.final_score, reverse=True)

        # Return top-k results
        return results[:top_k]

    def refresh_from_clickhouse(
        self,
        host: str = "localhost",
        port: int = 8123,
        username: str = "default",
        password: str = "",
        database: str = "default",
    ) -> Tuple[int, int]:
        """Refresh cache from ClickHouse."""
        import time

        start_time = time.time()

        try:
            # Connect to ClickHouse
            client = clickhouse_connect.get_client(
                host=host,
                port=port,
                username=username,
                password=password,
                database=database,
            )

            logger.info("Starting ClickHouse refresh...")

            # Calculate date range for last full month
            start_date, end_date, month_key = self._get_last_month_dates()
            logger.info("Target Period: %s to %s", start_date, end_date)
            logger.info("Month Key: %s", month_key)
            logger.info("Mode: Full Data (no sampling)")

            # Build ClickHouse query for segment preferences
            segment_query = f"""
            WITH recent_transactions AS (
                SELECT customer_id, sku_id, class_name, subclass_name
                FROM commerce_db.`transactions`
                WHERE transaction_date BETWEEN '{start_date}' AND '{end_date}'
                    AND substring(cast(sku_id as String), 1, 2) NOT IN ('41', '42')
                    AND trxtype IN (1, 21)
                    AND trxflag != 5
            ),
            class_prefs AS (
                SELECT
                    thk.rfm_segment,
                    rt.class_name,
                    '' as subclass_name,
                    rt.sku_id,
                    COUNT(*) as purchase_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY thk.rfm_segment, rt.class_name
                        ORDER BY COUNT(*) DESC
                    ) AS rn
                FROM recent_transactions rt
                JOIN analytics_db.customer_core_kpi thk
                    ON rt.customer_id = thk.customer_id
                    AND thk.month = '{month_key}'
                GROUP BY thk.rfm_segment, rt.class_name, rt.sku_id
            ),
            subclass_prefs AS (
                SELECT
                    thk.rfm_segment,
                    rt.class_name,
                    rt.subclass_name,
                    rt.sku_id,
                    COUNT(*) as purchase_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY thk.rfm_segment, rt.subclass_name
                        ORDER BY COUNT(*) DESC
                    ) AS rn
                FROM recent_transactions rt
                JOIN analytics_db.customer_core_kpi thk
                    ON rt.customer_id = thk.customer_id
                    AND thk.month = '{month_key}'
                GROUP BY thk.rfm_segment, rt.class_name, rt.subclass_name, rt.sku_id
            )
            SELECT rfm_segment, class_name, subclass_name, sku_id, rn
            FROM class_prefs WHERE rn <= 50
            UNION ALL
            SELECT rfm_segment, class_name, subclass_name, sku_id, rn
            FROM subclass_prefs WHERE rn <= 50
            ORDER BY rfm_segment, class_name, subclass_name, rn
            """

            # Execute segment preferences query
            logger.info("Executing segment preferences query...")
            query_start = time.time()
            result = client.query(segment_query)
            query_time = time.time() - query_start
            logger.info("Query completed in %.2fs", query_time)

            # Build preferences dict per class and subclass
            self.preferences = {}
            class_count = 0
            subclass_count = 0

            # Process each row from query result
            for row in result.result_rows:
                segment, class_name, subclass_name, sku_id, rank = row

                # Create segment entry if new
                if segment not in self.preferences:
                    self.preferences[segment] = {}

                # Determine level key (class or subclass)
                if subclass_name and subclass_name != "":
                    level_key = f"SUBCLASS:{subclass_name}"
                    subclass_count += 1
                else:
                    level_key = f"CLASS:{class_name}"
                    class_count += 1

                # Create level entry if new
                if level_key not in self.preferences[segment]:
                    self.preferences[segment][level_key] = {}

                # Calculate log-decay score: rank 1 = 1.0, rank 50 = 0.18
                self.preferences[segment][level_key][int(sku_id)] = 1.0 / math.log2(
                    rank + 1
                )

            logger.info(
                "Processed %d preferences (class: %d, subclass: %d)",
                len(result.result_rows),
                class_count,
                subclass_count,
            )

            # Query member segments
            member_query = f"""
            SELECT customer_id, rfm_segment 
            FROM analytics_db.customer_core_kpi 
            WHERE month = '{month_key}'
            """

            # Execute member segments query
            logger.info("Executing member segments query...")
            member_start = time.time()
            member_result = client.query(member_query)
            member_time = time.time() - member_start
            logger.info("Query completed in %.2fs", member_time)

            # Build member segments dict
            self.member_segments = {
                str(row[0]): row[1] for row in member_result.result_rows
            }

            logger.info("Processed %d member segments", len(member_result.result_rows))

            # Log completion stats
            total_time = time.time() - start_time
            logger.info("Refresh completed in %.2fs total", total_time)
            logger.info(
                "Performance: %.0f prefs/sec",
                len(result.result_rows) / query_time,
            )

            # Return counts
            return len(self.preferences), len(self.member_segments)

        except Exception as e:
            logger.error("ClickHouse error: %s", e)
            return 0, 0

    def refresh_from_parquet(self, parquet_dir: str) -> tuple:
        """
        Load segment preferences and member segments from local parquet files.
        Use this when ClickHouse is unavailable.

        Expects files exported by training/export_parquet.py:
            {parquet_dir}/segment_preferences.parquet
            {parquet_dir}/member_segments.parquet
        """
        import math
        import pandas as pd

        pref_path = os.path.join(parquet_dir, "segment_preferences.parquet")
        member_path = os.path.join(parquet_dir, "member_segments.parquet")

        if not os.path.exists(pref_path):
            raise FileNotFoundError(
                f"Segment preferences not found: {pref_path}\n"
                f"Run: python training/export_parquet.py"
            )
        if not os.path.exists(member_path):
            raise FileNotFoundError(
                f"Member segments not found: {member_path}\n"
                f"Run: python training/export_parquet.py"
            )

        # Load segment preferences
        prefs_df = pd.read_parquet(pref_path)
        self.preferences = {}
        for _, row in prefs_df.iterrows():
            segment = str(row["rfm_segment"])
            subclass_name = str(row["subclass_name"]) if row["subclass_name"] else ""
            class_name = str(row["class_name"])
            sku_id = int(row["sku_id"])
            score = float(row["score"])

            if segment not in self.preferences:
                self.preferences[segment] = {}

            level_key = f"SUBCLASS:{subclass_name}" if subclass_name else f"CLASS:{class_name}"
            if level_key not in self.preferences[segment]:
                self.preferences[segment][level_key] = {}

            self.preferences[segment][level_key][sku_id] = score

        # Load member segments
        members_df = pd.read_parquet(member_path)
        self.member_segments = {
            str(row["customer_id"]): row["rfm_segment"]
            for _, row in members_df.iterrows()
        }

        n_prefs = sum(len(v) for seg in self.preferences.values() for v in seg.values())
        logger.info(
            "Loaded from parquet: %d preferences, %d members",
            n_prefs, len(self.member_segments)
        )
        return len(self.preferences), len(self.member_segments)

