"""
Query Augmentation for Stage 2 Training.

This module provides augmenters for generating query variations:
- CharacterNoiseAugmenter: Character-level noise for typo robustness
- SemanticQueryAugmenter: Semantic-level augmentation (word dropout, shuffle, truncate)
"""

import random
import re
from typing import List, Optional

# ============================================================================
# CHARACTER NOISE AUGMENTER
# ============================================================================


class CharacterNoiseAugmenter:
    """Character-level noise augmenter for typo robustness (5% noise rate)."""

    def __init__(self, noise_prob: float = 0.05, seed: Optional[int] = None):
        """Initialize character noise augmenter."""
        self.noise_prob = noise_prob
        if seed is not None:
            random.seed(seed)

        # Common character substitutions for Indonesian/English typos
        self.substitutions = {
            "a": ["e", "o"],
            "e": ["a", "i"],
            "i": ["e", "y"],
            "o": ["a", "u"],
            "u": ["o", "i"],
            "s": ["z", "c"],
            "c": ["s", "k"],
            "k": ["c", "g"],
            "p": ["b"],
            "b": ["p"],
            "t": ["d"],
            "d": ["t"],
            "n": ["m"],
            "m": ["n"],
        }
        self.insertions = ["h", "y", "i", "u", "a", "e"]

    def augment(self, text: str) -> str:
        """Apply character-level noise to text."""
        if not text or not text.strip():
            return text

        chars = list(text.strip().lower())
        result = []

        for char in chars:
            if not char.isalpha() or random.random() >= self.noise_prob:
                result.append(char)
                continue

            noise_type = random.choice(["substitute", "delete", "insert", "duplicate"])

            if noise_type == "substitute":
                result.append(
                    self.substitutions.get(char, [char])[0]
                    if char in self.substitutions
                    else char
                )
            elif noise_type == "delete":
                pass  # Skip character
            elif noise_type == "insert":
                result.append(random.choice(self.insertions))
                result.append(char)
            else:  # duplicate
                result.append(char)
                result.append(char)

        return "".join(result)


# ============================================================================
# SEMANTIC QUERY AUGMENTER
# ============================================================================


class SemanticQueryAugmenter:
    """Semantic-level augmentation simulating real user search behavior."""

    # Common units and quantities to strip
    UNITS = r"\b\d+(?:\.\d+)?\s*(?:kg|g|ml|l|liter|litre|pcs|pc|pack|sachet|btl|bottle|can|box|gr|gram|ons|oz)\b"

    # Common size indicators
    SIZES = r"\b(?:mini|small|medium|large|xl|xxl|jumbo|family|travel|sample)\b"

    def __init__(self, seed: Optional[int] = None):
        """Initialize semantic query augmenter with optimal probabilities."""
        # Hardcoded optimal probabilities (proven in production)
        self.word_dropout_prob = 0.3
        self.shuffle_prob = 0.2
        self.truncate_prob = 0.25
        self.strip_numbers_prob = 0.4

        if seed is not None:
            random.seed(seed)

    def augment(self, text: str, n_augments: int = 1) -> List[str]:
        """Generate multiple augmented query variations from SKU name."""
        results = []

        for _ in range(n_augments):
            augmented = self._single_augment(text)
            if augmented and augmented.strip():
                results.append(augmented)

        # Always include a clean, minimal version
        minimal = self._extract_minimal(text)
        if minimal and minimal not in results:
            results.append(minimal)

        return results if results else [text.lower()]

    def _single_augment(self, text: str) -> str:
        """Apply a single random augmentation."""
        text = text.lower().strip()

        # Choose augmentation strategy randomly
        strategies = []

        if random.random() < self.word_dropout_prob:
            strategies.append("dropout")
        if random.random() < self.shuffle_prob:
            strategies.append("shuffle")
        if random.random() < self.truncate_prob:
            strategies.append("truncate")
        if random.random() < self.strip_numbers_prob:
            strategies.append("strip_numbers")

        # If no strategy selected, apply at least one
        if not strategies:
            strategies = [random.choice(["dropout", "truncate", "strip_numbers"])]

        result = text

        for strategy in strategies:
            if strategy == "dropout":
                result = self._word_dropout(result)
            elif strategy == "shuffle":
                result = self._shuffle_words(result)
            elif strategy == "truncate":
                result = self._truncate(result)
            elif strategy == "strip_numbers":
                result = self._strip_numbers(result)

        return result.strip()

    def _word_dropout(self, text: str) -> str:
        """Randomly drop words from text."""
        words = text.split()
        if len(words) <= 2:
            return text

        # Keep at least 1-2 words
        keep_count = max(1, int(len(words) * (1 - self.word_dropout_prob)))
        keep_count = min(keep_count, len(words) - 1)  # Drop at least 1

        keep_indices = sorted(random.sample(range(len(words)), keep_count))
        kept_words = [words[i] for i in keep_indices]

        return " ".join(kept_words)

    def _shuffle_words(self, text: str) -> str:
        """Shuffle word order."""
        words = text.split()
        if len(words) <= 1:
            return text

        random.shuffle(words)
        return " ".join(words)

    def _truncate(self, text: str) -> str:
        """Keep only first N words."""
        words = text.split()
        if len(words) <= 2:
            return text

        # Keep 1-3 words
        keep_count = random.randint(1, min(3, len(words) - 1))
        return " ".join(words[:keep_count])

    def _strip_numbers(self, text: str) -> str:
        """Remove numbers, quantities, and units."""
        # Remove unit patterns
        result = re.sub(self.UNITS, "", text, flags=re.IGNORECASE)
        # Remove size indicators
        result = re.sub(self.SIZES, "", result, flags=re.IGNORECASE)
        # Remove standalone numbers
        result = re.sub(r"\b\d+(?:\.\d+)?\b", "", result)
        # Clean up extra spaces
        result = re.sub(r"\s+", " ", result).strip()
        return result

    def _extract_minimal(self, text: str) -> str:
        """Extract minimal searchable query (usually brand + product type)."""
        text = text.lower().strip()

        # Strip numbers and units
        clean = self._strip_numbers(text)

        # Keep only first 2-3 significant words
        words = clean.split()

        # Filter out very short words (likely noise)
        significant = [w for w in words if len(w) > 2]

        if len(significant) >= 2:
            return " ".join(significant[:2])
        elif significant:
            return significant[0]
        else:
            return words[0] if words else text
