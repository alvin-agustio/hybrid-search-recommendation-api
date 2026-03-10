"""
BM25 Baseline Search Implementation.

Uses rank_bm25 library for traditional keyword-based search with SymSpell typo correction.
"""

import re
import os
import pickle
from typing import List, Tuple

import pandas as pd
from rank_bm25 import BM25Okapi
from symspellpy import SymSpell, Verbosity


class BM25Baseline:
    """
    BM25 baseline search using rank_bm25 library and SymSpell.

    Provides traditional keyword-based search as a baseline
    and for hybrid search fusion.
    """

    def __init__(self):
        """Initialize BM25Baseline."""
        self.bm25 = None
        self.sku_ids = []
        self.product_names = []
        self.tokenized_corpus = []

        # Initialize SymSpell
        self.symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self.spell_ready = False

    def build_index(self, df: pd.DataFrame, text_column: str = "sku_name"):
        """Build BM25 index and SymSpell dictionary from product dataframe."""
        # Extract SKU IDs and product names
        self.sku_ids = df["sku_id"].tolist()
        self.product_names = df[text_column].tolist()

        # Tokenize all product names (word only)
        self.tokenized_corpus = [self._tokenize(text) for text in self.product_names]

        # Build BM25 index from tokenized corpus
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Build SymSpell Dictionary on-the-fly from actual product names
        print("  Building SymSpell dictionary from product names...")
        for name in self.product_names:
            words = self._tokenize(name)
            for word in words:
                self.symspell.create_dictionary_entry(word, 1)
        self.spell_ready = True

        print(f"[OK] BM25 index built: {len(self.sku_ids)} products")
        print(
            f"[OK] SymSpell dictionary built: {len(self.symspell.words)} unique words"
        )

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into pure words.
        No more n-grams. SymSpell handles the typo tolerance.
        """
        text = str(text).lower()
        tokens = re.split(r"[^a-z0-9]+", text)
        return [t for t in tokens if len(t) > 1]

    def correct_spelling(self, query: str) -> str:
        """Correct typos in the query using SymSpell."""
        if not self.spell_ready:
            return query

        words = self._tokenize(query)
        corrected_words = []

        for word in words:
            # max_edit_distance=2 means it will fix up to 2 typos per word
            suggestions = self.symspell.lookup(
                word, Verbosity.CLOSEST, max_edit_distance=2
            )
            if suggestions:
                # Use the closest match found in the product catalog
                corrected_words.append(suggestions[0].term)
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    def get_word_frequencies(self, query: str) -> List[int]:
        """
        Get Document Frequency (DF) for each valid word in the query.
        Useful for Intent Detection (Conceptual vs Exact).
        We do NOT correct spelling here. We want to know if the EXACT word
        typed by the user exists in our catalog. If an exact word doesn't exist
        (DF=0), it strongly signals a conceptual query or extreme typo.
        """
        if self.bm25 is None:
            return []

        # We deliberately DO NOT correct the spelling here.
        # We need the original query's document frequency.
        words = self._tokenize(query)
        freqs = []

        # Count document frequency manually since rank_bm25 API doesn't expose total doc frequency easily
        for word in words:
            count = sum(1 for doc in self.tokenized_corpus if word in doc)
            freqs.append(count)

        return freqs

    def search(
        self, query: str, top_k: int = 20, use_symspell: bool = False
    ) -> List[Tuple[int, float]]:
        """Search for products matching query."""
        # Check if index is built
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")

        if use_symspell:
            # 1. Correct spelling first
            corrected_query = self.correct_spelling(query)
            # 2. Tokenize the corrected query
            query_tokens = self._tokenize(corrected_query)
        else:
            # Tokenize original query without spelling correction
            query_tokens = self._tokenize(query)

        # Return empty if no valid tokens
        if not query_tokens:
            return []

        # Calculate BM25 scores for all products
        scores = self.bm25.get_scores(query_tokens)

        # Get indices of top-k highest scores
        top_indices = scores.argsort()[::-1][:top_k]

        # Build results with SKU IDs and scores (filter zero scores)
        results = [
            (self.sku_ids[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]

        return results

    def save(self, output_dir: str):
        """Save BM25 index and SymSpell dictionary."""
        os.makedirs(output_dir, exist_ok=True)

        # Save BM25
        with open(os.path.join(output_dir, "bm25_index.pkl"), "wb") as f:
            pickle.dump(self, f)

        # Optional: could also save symspell dictionary uniquely if it gets large,
        # but since it's pickled within `self`, it's already saved above.

    @classmethod
    def load(cls, input_dir: str) -> "BM25Baseline":
        """Load BM25 index from disk."""
        with open(os.path.join(input_dir, "bm25_index.pkl"), "rb") as f:
            bm25 = pickle.load(f)
            return bm25
