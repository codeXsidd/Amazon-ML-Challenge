# Amazon ML Challenge 2026 — Business Entity Resolution

**Team:** flying ninjas  
**Evaluated Competition Score:** 0.617 (Macro $F_{0.5}$)  

---

## 📌 Project Overview
This repository contains the complete end-to-end Machine Learning pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution**. 

The goal is to match business entities across three heterogeneous, noisy data sources:
- **Source 1:** Reference deduplicated dataset (1.73M entities in test).
- **Source 2 & Source 3:** Noisy secondary sources with abbreviations, missing address fields, variations, and transliterations across the US, India, and France.

---

## 🚀 Architecture & Key Features

1. **Scalable Multi-Pass Inverted-Index Blocking:**
   - Evaluates entities across 5 complementary blocking rules:
     - Exact normalized name (`nn`)
     - Sorted name tokens (`sn`)
     - Compact whitespace-stripped names (`cn`)
     - Token overlap via inverted index ($\ge 2$ shared tokens, frequency $\le 1000$, length $\ge 3$)
     - Long distinctive tokens ($\ge 1$ shared token, length $\ge 5$, frequency $\le 500$)
   - Reduces the Cartesian search space ($>10^{12}$ pairs) by **99.998%** while maintaining strong recall.

2. **26 Handcrafted Similarity Features:**
   - C-accelerated string kernels using **RapidFuzz** (`fuzz.ratio`, `fuzz.token_sort_ratio`, `fuzz.token_set_ratio`).
   - Character 3-gram Jaccard and structural token overlap coefficients.
   - Address string similarity, length ratios, and missing address indicators.
   - Cross-feature interactions (`name_ratio` $\times$ `addr_ratio`) and blocking provenance metadata.

3. **High-Precision LightGBM Matching Engine:**
   - 400 trees with `is_unbalance=True`, max depth 7, 63 leaves.
   - Optimized specifically for the precision-heavy **Macro $F_{0.5}$** metric on an entity-level validation split.
   - Calibrated decision threshold: **0.92**.

---

## 📂 Repository Structure

```
.
├── README.md                              # Repository overview
├── .gitignore                             # Git ignore rules for datasets and large TSVs
└── student_resource/
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/                       # Complete pipeline source code
    │       │   ├── run_pipeline_v5.py     # Main reproducible end-to-end script
    │       │   ├── blocking.py            # Blocking logic and index builders
    │       │   ├── features.py            # Feature extraction
    │       │   ├── matcher.py             # LightGBM classifier wrapper
    │       │   ├── normalization.py       # Text preprocessors
    │       │   ├── data_loader.py         # TSV streaming loaders
    │       │   └── evaluation.py          # Entity-level macro F0.5 evaluator
    │       ├── README.md                  # Detailed code documentation
    │       └── requirements.txt           # Pinned dependencies
    ├── utils/
    │   └── validate_submission.py         # Official submission validation script
    └── Documentation_template.md          # Official technical solution report
```

---

## ⚡ Quick Start & Reproduction

### 1. Install Dependencies
```bash
pip install -r student_resource/code/business_entity_resolution/requirements.txt
```

### 2. Run Pipeline End-to-End
```bash
cd student_resource
python code/business_entity_resolution/src/run_pipeline_v5.py
```
This runs preprocessing, blocking, training, validation, test inference, and outputs:
- `output/matching_results.tsv` (Leaderboard upload)
- `output/candidate_pairs.tsv` (Candidate blocking set)

### 3. Validate Submission
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
