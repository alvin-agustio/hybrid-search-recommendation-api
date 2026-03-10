# Search Pipeline Evaluation Report V5.4

**Date:** February 24, 2026  
**Summary:** Objective evaluation of the Hybrid Search V5.4 pipeline. This round of testing focused on both direct keyword retrieval and conceptual search behavior to measure how far the model has moved beyond purely lexical matching.

**Average search latency:** ~98.9 ms, which is responsive for a hybrid retrieval plus reranking pipeline.

---

## 1. Pure Conceptual Search

This section evaluates how well the semantic layer interprets intent and synonyms without explicit brand terms.

### Query: `sarapan`
**Top 3 results:**
1. SEREAL GANDUM COKELAT 390G
2. BISKUIT OATMEAL
3. MINUMAN OAT COKELAT 1 LT

**Status:** Very strong  
**Latency:** ~82.1 ms

**Analysis:** This is one of the strongest outcomes in V5.4. The single-word query `sarapan` maps correctly to breakfast-adjacent products such as cereal and oatmeal even though the token itself does not appear in product names. This demonstrates genuine conceptual retrieval rather than surface keyword matching.

### Query: `cemilan`
**Top 3 results:**
1. KERIPIK PISANG COKELAT 115GR
2. SNACK GURIH PANGSIT 250GR
3. KERUPUK RINGAN ANEKA RASA

**Status:** Very strong  
**Latency:** ~76.2 ms

**Analysis:** Similar to the previous case, the model correctly maps `cemilan` to chips and snack products. This is a strong signal that the semantic representation captures purchase intent well.

### Query: `makanan kucing`
**Top 3 results:**
1. ANEKA MAKANAN
2. MAKANAN KUCING KALENG TUNA 400GR
3. MAKANAN KUCING DEWASA SALMON 1.1KG

**Status:** Acceptable  
**Latency:** ~95.6 ms

**Analysis:** The encoder successfully links `kucing` to cat-food products and surfaces specialized pet-food items. The first result is still generic, so the behavior is useful but not yet perfect.

---

## 2. Direct Attribute Search

This section measures how accurately the system handles exact, attribute-driven search behavior.

### Query: `mi instan goreng`
**Top 3 results:**
1. MI INSTAN GORENG RASA IKAN
2. MI INSTAN GORENG SPESIAL
3. MI INSTAN GORENG VARIAN KLASIK

**Status:** Good  
**Latency:** ~97.4 ms

**Analysis:** No major issue here. The hybrid combination of BM25 and semantic retrieval finds the intended product family reliably.

### Query: `obat nyamuk`
**Top 3 results:**
1. KOTAK OBAT RABBIT
2. AEROSOL ANTI NYAMUK 600ML
3. AEROSOL ANTI NYAMUK 200ML

**Status:** Good  
**Latency:** ~104.0 ms

**Analysis:** The hybrid approach is reasonably reliable. The vector model bridges `nyamuk` toward the household insecticide category, although the first result shows room for reranking improvement.

---

## 3. Typo Tolerance

This section checks how robust the retrieval stack is when users misspell common retail queries.

### Query: `sampho conditioner`
**Top 3 results:**
1. CONDITIONER KERATIN 300ML
2. SHAMPOO ANTI DANDRUFF 70ML
3. CONDITIONER ANTI LEPEK 290ML

**Status:** Good  
**Latency:** ~100.6 ms

**Analysis:** The system recovers well from the malformed token `sampho` and still centers the results on hair-care products. This indicates the typo-tolerance path is working as intended.

---

