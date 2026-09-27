import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'dataset', 'test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

TRAIN_S1 = os.path.join(TRAIN_DIR, 'train_source1.tsv')
TRAIN_S2 = os.path.join(TRAIN_DIR, 'train_source2.tsv')
TRAIN_S3 = os.path.join(TRAIN_DIR, 'train_source3.tsv')
TRAIN_GT = os.path.join(TRAIN_DIR, 'train_ground_truth.tsv')

TEST_S1 = os.path.join(TEST_DIR, 'test_source1.tsv')
TEST_S2 = os.path.join(TEST_DIR, 'test_source2.tsv')
TEST_S3 = os.path.join(TEST_DIR, 'test_source3.tsv')

MATCHING_RESULTS = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')

RANDOM_SEED = 42
VAL_FRACTION = 0.2
TRAIN_SAMPLE_SIZE = 30_000

TOKEN_FREQ_CAP_PASS1 = 1000
TOKEN_FREQ_CAP_PASS2 = 500
MIN_TOKEN_LEN_PASS1 = 3
MIN_TOKEN_LEN_PASS2 = 5
MIN_SHARED_TOKENS_PASS1 = 2
MIN_SHARED_TOKENS_PASS2 = 1
PER_S1_CAP_PASS1 = 150
PER_S1_CAP_PASS2 = 30
MAX_CANDIDATES_PER_S1 = 200

LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'n_estimators': 400,
    'learning_rate': 0.08,
    'num_leaves': 63,
    'max_depth': 7,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'is_unbalance': True,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
}

THRESHOLD_CANDIDATES = [
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
    0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98
]

FEATURE_CHUNK_SIZE = 2_000_000
