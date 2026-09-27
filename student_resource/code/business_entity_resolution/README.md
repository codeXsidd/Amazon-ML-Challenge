# Business Entity Resolution Pipeline

## Overview

Matches business entities from Source 1 against Source 2 and Source 3 using
multi-strategy blocking, feature engineering with RapidFuzz string similarity,
and a LightGBM binary classifier optimized for F0.5 (precision-heavy).

**Validated performance:**
- F0.5 (macro): **0.6663** on validation set
- Precision: 0.9353, Recall: 0.4714
- Optimal threshold: 0.92
- Test S1 entities: 1,732,544 | With matches: 1,374,239 | Singletons: 358,305

## Requirements

- Python 3.10+
- Dependencies: `pip install -r requirements.txt`

## Quick Start

From the `student_resource/` directory:

```bash
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/run_pipeline_v5.py
```

This runs the full pipeline end-to-end and writes:
- `output/matching_results.tsv` — final entity matches (upload to leaderboard)
- `output/candidate_pairs.tsv` — blocking candidate set

Expected runtime: ~2 hours on 16 GB RAM / 14 CPU cores.

## Pipeline Architecture

### 1. Data Loading & Normalization
- Loads Source 1, 2, 3 per country partition (memory-efficient)
- Normalizes names: lowercase, replace `&` with `and`, remove punctuation,
  strip legal suffixes (Inc, Corp, LLC, Ltd, Pvt, GmbH, SA, SARL, SAS, etc.), collapse whitespace
- Normalizes addresses: lowercase, remove punctuation, collapse whitespace
- Creates derived fields: sorted name tokens, compact name (no spaces)
- Country treated as open-set string — France (unseen in training) handled correctly

### 2. Blocking (Candidate Generation)
Five blocking strategies combined (union of all candidates):

| Strategy | Key | Description |
|----------|-----|-------------|
| Exact name (E) | `nn` | Normalized name merge |
| Sorted name (S) | `sn` | Sorted-token name merge |
| Compact name (C) | `cn` | Whitespace-removed name merge (≥5 chars) |
| Token overlap (T) | tokens | ≥2 shared name tokens via inverted index (freq ≤1000, len ≥3) |
| Long token (L) | tokens | ≥1 shared name token ≥5 chars (freq ≤500) |

Candidates capped at 200 per Source 1 entity.
Candidate recall (validation): ~52.7% | Avg candidates per S1: ~36.8

### 3. Feature Engineering (26 features)
- **Name similarity (10):** exact match, sorted match, fuzz.ratio, token_sort_ratio,
  token_set_ratio, Jaccard, overlap coefficient, length ratio, compact ratio,
  full-name ratio (with legal suffixes)
- **Address similarity (7):** exact match, fuzz.ratio, token_sort_ratio, Jaccard,
  overlap coefficient, length ratio, missing-address indicator
- **Context (3):** country match, name×address interaction, source indicator (S2 vs S3)
- **Blocking meta (6):** block count, per-strategy indicators (E, S, C, T, L)

All string similarities computed via RapidFuzz (C-optimized, fast).

### 4. LightGBM Classifier
- Binary classifier with `is_unbalance=True`
- 400 trees, learning rate 0.08, 63 leaves, max depth 7
- Training sample: up to 30K S1 per country partition
- Ground truth positives injected into training pairs

### 5. Threshold Optimization
- Sweeps 19 thresholds from 0.20 to 0.98 on validation set
- Selects threshold maximizing macro-averaged F0.5
- **Optimal threshold: 0.92** (precision-heavy F0.5 metric)

### 6. Test Inference
- Processes each country partition sequentially (France, India, US)
- Features computed in 2M-pair chunks to control memory
- Writes both output TSV files in correct format

## Project Structure

```
student_resource/
├── dataset/
│   ├── train/              # train_source1/2/3.tsv, train_ground_truth.tsv
│   └── test/               # test_source1/2/3.tsv
├── output/
│   ├── matching_results.tsv    # Final matches (upload to leaderboard)
│   └── candidate_pairs.tsv     # Blocking candidate set
├── utils/
│   └── validate_submission.py  # Official validator
└── code/business_entity_resolution/
    ├── src/
    │   ├── run_pipeline_v5.py # Main entry point — complete self-contained pipeline
    │   ├── __init__.py
    │   ├── config.py          # paths and hyperparameters
    │   ├── normalization.py   # name/address normalization
    │   ├── data_loader.py     # data loading utilities
    │   ├── blocking.py        # inverted index blocking engine
    │   ├── features.py        # feature computation
    │   ├── matcher.py         # LightGBM wrapper
    │   ├── evaluation.py      # F0.5 metrics
    │   ├── pipeline.py        # modular pipeline (reference)
    │   ├── inference.py       # inference entry point
    │   └── utils.py           # output validation
    ├── requirements.txt
    └── README.md
```

The main entry point is `code/business_entity_resolution/src/run_pipeline_v5.py`,
a complete self-contained pipeline (no imports from other `src/` modules needed).

## Reproducing Results

```bash
# Step 1: Install dependencies
cd student_resource/
pip install -r code/business_entity_resolution/requirements.txt

# Step 2: Run the complete pipeline
python code/business_entity_resolution/src/run_pipeline_v5.py

# Step 3: Validate the submission files
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
# Expected: PASS

# Step 4 (optional): Create submission ZIP
python -c "
import zipfile, os
with zipfile.ZipFile('business_entity_resolution_submission.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('output/matching_results.tsv')
    zf.write('output/candidate_pairs.tsv')
    for root, dirs, files in os.walk('code/business_entity_resolution'):
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.venv', 'venv']]
        for file in files:
            if not file.endswith(('.pyc',)):
                path = os.path.join(root, file)
                zf.write(path)
    zf.write('Documentation_template.md')
"
```

## Constraints Met

- No external data, APIs, or services used
- All libraries MIT/Apache 2.0 / BSD-3-Clause licensed
- LightGBM model ≪ 8B parameters (400 trees, 63 leaves each)
- Country treated as open set (France handled without retraining or special-casing)
- Full coverage: every test S1 entity appears in output exactly once

## Dependencies

| Package | Version | License |
|---------|---------|---------|
| pandas | ≥2.0.0 | BSD-3-Clause |
| numpy | ≥1.24.0 | BSD-3-Clause |
| scikit-learn | ≥1.4.0 | BSD-3-Clause |
| lightgbm | ≥4.0.0 | MIT |
| rapidfuzz | ≥3.0.0 | MIT |
| tqdm | ≥4.60.0 | MIT |
| scipy | ≥1.10.0 | BSD-3-Clause |
