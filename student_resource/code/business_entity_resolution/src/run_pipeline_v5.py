#!/usr/bin/env python3
"""
Business Entity Resolution - Pipeline v5
Adds token inverted index blocking for dramatically better recall.
"""
import os, sys, time, gc, re
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from rapidfuzz import fuzz
import lightgbm as lgb

def log(msg): print(msg, flush=True)

# File lives in code/business_entity_resolution/src/ → go up 3 levels to student_resource/
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAIN_DIR = os.path.join(BASE, 'dataset', 'train')
TEST_DIR  = os.path.join(BASE, 'dataset', 'test')
OUTPUT_DIR = os.path.join(BASE, 'output')
SEED = 42

# ─── Fast vectorized normalization ───────────────────────────

LEGAL_RE = re.compile(
    r'\b(?:inc|incorporated|corp|corporation|llc|ltd|limited|co|company|'
    r'plc|pllc|llp|pvt|private|pub|public|gmbh|sa|sarl|sas|srl|ag|bv|nv|'
    r'trust|foundation|association|assn|com|org|net)\b')

def norm_name_col(series):
    s = series.fillna('').str.lower()
    s = s.str.replace('&', ' and ', regex=False)
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    s = s.str.replace(LEGAL_RE, ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def norm_name_full_col(series):
    s = series.fillna('').str.lower()
    s = s.str.replace('&', ' and ', regex=False)
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def norm_addr_col(series):
    s = series.fillna('').str.lower()
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def preprocess(df):
    df = df.copy()
    df['business_name'] = df['business_name'].fillna('')
    df['business_address'] = df['business_address'].fillna('')
    df['country'] = df['country'].fillna('')
    df['nn'] = norm_name_col(df['business_name'])
    df['nnf'] = norm_name_full_col(df['business_name'])
    df['sn'] = df['nn'].apply(lambda x: ' '.join(sorted(x.split())) if x else '')
    df['cn'] = df['nn'].str.replace(r'\s+', '', regex=True)
    df['na'] = norm_addr_col(df['business_address'])
    df['cl'] = df['country'].str.lower().str.strip()
    return df

# ─── Blocking: merge-based + token inverted index ────────────

def blocking_merge_only(s1_df, s2s3_df):
    """Exact name + sorted name + compact name blocking via merge."""
    pairs_list = []

    # Block 1: exact normalized name
    s1_nn = s1_df[s1_df['nn'] != ''][['entity_id', 'nn']]
    s2_nn = s2s3_df[s2s3_df['nn'] != ''][['entity_id', 'nn']]
    if len(s1_nn) > 0 and len(s2_nn) > 0:
        m = s1_nn.merge(s2_nn, on='nn', suffixes=('_s1', '_s2s3'))
        if len(m) > 0:
            m['block'] = 'E'
            pairs_list.append(m[['entity_id_s1', 'entity_id_s2s3', 'block']])
    del s1_nn, s2_nn; gc.collect()

    # Block 2: sorted name
    s1_sn = s1_df[s1_df['sn'] != ''][['entity_id', 'sn']]
    s2_sn = s2s3_df[s2s3_df['sn'] != ''][['entity_id', 'sn']]
    if len(s1_sn) > 0 and len(s2_sn) > 0:
        m = s1_sn.merge(s2_sn, on='sn', suffixes=('_s1', '_s2s3'))
        if len(m) > 0:
            m['block'] = 'S'
            pairs_list.append(m[['entity_id_s1', 'entity_id_s2s3', 'block']])
    del s1_sn, s2_sn; gc.collect()

    # Block 3: compact name (no spaces, ≥5 chars)
    s1_cn = s1_df[(s1_df['cn'] != '') & (s1_df['cn'].str.len() >= 5)][['entity_id', 'cn']]
    s2_cn = s2s3_df[(s2s3_df['cn'] != '') & (s2s3_df['cn'].str.len() >= 5)][['entity_id', 'cn']]
    if len(s1_cn) > 0 and len(s2_cn) > 0:
        m = s1_cn.merge(s2_cn, on='cn', suffixes=('_s1', '_s2s3'))
        if len(m) > 0:
            m['block'] = 'C'
            pairs_list.append(m[['entity_id_s1', 'entity_id_s2s3', 'block']])
    del s1_cn, s2_cn; gc.collect()

    return pairs_list


def blocking_token(s1_df, s2s3_df, min_shared=2, max_freq=3000, min_tok_len=3, per_s1_cap=200):
    """Token inverted index blocking. For each S1 entity, find S2S3 entities
    sharing ≥min_shared name tokens. Uses numpy for speed."""
    t0 = time.time()

    # Build inverted index: token -> list of S2S3 integer indices
    idx = defaultdict(list)
    s2s3_eids = s2s3_df['entity_id'].values
    s2s3_nns = s2s3_df['nn'].values
    for j in range(len(s2s3_eids)):
        nn = s2s3_nns[j]
        if nn:
            for tok in nn.split():
                if len(tok) >= min_tok_len:
                    idx[tok].append(j)

    # Filter high-frequency tokens and convert to numpy arrays
    filtered_idx = {}
    for tok, arr in idx.items():
        if len(arr) <= max_freq:
            filtered_idx[tok] = np.array(arr, dtype=np.int32)
    del idx; gc.collect()
    log(f"    Token index: {len(filtered_idx):,} tokens (filtered ≤{max_freq}, len≥{min_tok_len})")

    # For each S1 entity, find candidates with ≥min_shared shared tokens
    s1_eids = s1_df['entity_id'].values
    s1_nns = s1_df['nn'].values

    all_s1_idx = []
    all_s2s3_idx = []

    for i in range(len(s1_eids)):
        nn = s1_nns[i]
        if not nn:
            continue
        toks = [t for t in nn.split() if len(t) >= min_tok_len and t in filtered_idx]
        if len(toks) < min_shared:
            continue

        # Concatenate all candidate indices from each token
        arrays = [filtered_idx[t] for t in toks]
        all_cands = np.concatenate(arrays)
        if len(all_cands) == 0:
            continue

        # Count occurrences using numpy
        cand_ids, counts = np.unique(all_cands, return_counts=True)
        mask = counts >= min_shared
        matches = cand_ids[mask]
        match_counts = counts[mask]

        if len(matches) > per_s1_cap:
            top_idx = np.argsort(-match_counts)[:per_s1_cap]
            matches = matches[top_idx]

        if len(matches) > 0:
            all_s1_idx.extend([i] * len(matches))
            all_s2s3_idx.extend(matches.tolist())

    log(f"    Token blocking (min={min_shared}): {len(all_s1_idx):,} pairs in {time.time()-t0:.0f}s")

    if not all_s1_idx:
        return pd.DataFrame(columns=['entity_id_s1', 'entity_id_s2s3', 'block'])

    return pd.DataFrame({
        'entity_id_s1': s1_eids[all_s1_idx],
        'entity_id_s2s3': s2s3_eids[all_s2s3_idx],
        'block': 'T'
    })


def blocking_combined(s1_df, s2s3_df):
    """All blocking strategies combined: merge-based + token inverted index."""
    pairs_list = blocking_merge_only(s1_df, s2s3_df)

    # Token blocking pass 1: ≥2 shared tokens (catches multi-word partial matches)
    t_pairs1 = blocking_token(s1_df, s2s3_df, min_shared=2, max_freq=1000, min_tok_len=3, per_s1_cap=150)
    if len(t_pairs1) > 0:
        pairs_list.append(t_pairs1[['entity_id_s1', 'entity_id_s2s3', 'block']])
    del t_pairs1; gc.collect()

    # Token blocking pass 2: ≥1 shared long token (catches short-name entities)
    t_pairs2 = blocking_token(s1_df, s2s3_df, min_shared=1, max_freq=500, min_tok_len=5, per_s1_cap=30)
    if len(t_pairs2) > 0:
        t_pairs2['block'] = 'L'
        pairs_list.append(t_pairs2[['entity_id_s1', 'entity_id_s2s3', 'block']])
    del t_pairs2; gc.collect()

    if not pairs_list:
        return pd.DataFrame(columns=['entity_id_s1', 'entity_id_s2s3', 'block', 'n_blocks'])

    all_pairs = pd.concat(pairs_list, ignore_index=True)
    del pairs_list; gc.collect()

    # Deduplicate and aggregate block info
    dedup = all_pairs.drop_duplicates(subset=['entity_id_s1', 'entity_id_s2s3', 'block'])
    agg = dedup.groupby(['entity_id_s1', 'entity_id_s2s3']).agg(
        block=('block', lambda x: ','.join(sorted(x))),
        n_blocks=('block', 'count')
    ).reset_index()
    del all_pairs, dedup; gc.collect()

    # Cap candidates per S1 at 200
    cpc = agg.groupby('entity_id_s1').cumcount()
    agg = agg[cpc < 200].reset_index(drop=True)

    return agg

# ─── Features ─────────────────────────────────────────────────

FEAT_NAMES = [
    'name_exact', 'name_sorted_exact', 'name_ratio', 'name_tsort', 'name_tset',
    'name_jaccard', 'name_overlap', 'name_lr', 'compact_ratio', 'name_full_ratio',
    'addr_exact', 'addr_ratio', 'addr_tsort', 'addr_jaccard', 'addr_overlap',
    'addr_lr', 'addr_miss',
    'country_eq', 'nx_addr', 'is_s2',
    'n_blocks', 'blk_E', 'blk_S', 'blk_C', 'blk_T', 'blk_L',
]

def _jac(s1, s2):
    t1 = set(s1.split()) if s1 else set()
    t2 = set(s2.split()) if s2 else set()
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)

def _ov(s1, s2):
    t1 = set(s1.split()) if s1 else set()
    t2 = set(s2.split()) if s2 else set()
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / min(len(t1), len(t2))

def _lr(a, b):
    if not a or not b: return 0.0
    return min(len(a), len(b)) / max(len(a), len(b))

def compute_features_batch(pairs_df, s1_lk, s2s3_lk):
    n = len(pairs_df)
    X = np.zeros((n, len(FEAT_NAMES)), dtype=np.float32)
    s1_ids = pairs_df['entity_id_s1'].values
    s2s3_ids = pairs_df['entity_id_s2s3'].values
    blocks = pairs_df['block'].values
    nblks = pairs_df['n_blocks'].values

    for i in range(n):
        r1 = s1_lk.get(s1_ids[i])
        r2 = s2s3_lk.get(s2s3_ids[i])
        if r1 is None or r2 is None:
            continue
        nn1, nnf1, sn1, cn1, na1, cl1 = r1
        nn2, nnf2, sn2, cn2, na2, cl2 = r2
        blk = blocks[i]; eid2 = s2s3_ids[i]

        nr = fuzz.ratio(nn1, nn2)/100 if nn1 and nn2 else 0.0
        nts = fuzz.token_sort_ratio(nn1, nn2)/100 if nn1 and nn2 else 0.0
        ntset = fuzz.token_set_ratio(nn1, nn2)/100 if nn1 and nn2 else 0.0
        cr = fuzz.ratio(cn1, cn2)/100 if cn1 and cn2 else 0.0
        nfr = fuzz.ratio(nnf1, nnf2)/100 if nnf1 and nnf2 else 0.0
        ar = fuzz.ratio(na1, na2)/100 if na1 and na2 else 0.0
        ats = fuzz.token_sort_ratio(na1, na2)/100 if na1 and na2 else 0.0

        X[i] = [
            1.0 if (nn1 != '' and nn2 != '' and nn1 == nn2) else 0.0,
            1.0 if (sn1 != '' and sn2 != '' and sn1 == sn2) else 0.0,
            nr, nts, ntset,
            _jac(nn1, nn2), _ov(nn1, nn2), _lr(nn1, nn2), cr, nfr,
            1.0 if (na1 != '' and na2 != '' and na1 == na2) else 0.0,
            ar, ats, _jac(na1, na2), _ov(na1, na2),
            _lr(na1, na2),
            1.0 if (na1 == '' or na2 == '') else 0.0,
            1.0 if (cl1 == cl2 and cl1 != '') else 0.0,
            nts * ats,
            1.0 if eid2[:2] == 'S2' else 0.0,
            float(nblks[i]),
            1.0 if 'E' in blk else 0.0,
            1.0 if 'S' in blk else 0.0,
            1.0 if 'C' in blk else 0.0,
            1.0 if 'T' in blk else 0.0,
            1.0 if 'L' in blk else 0.0,
        ]
    return X

def make_lookup(df):
    d = {}
    eids = df['entity_id'].values
    nns = df['nn'].values; nnfs = df['nnf'].values; sns = df['sn'].values
    cns = df['cn'].values; nas = df['na'].values; cls = df['cl'].values
    for i in range(len(eids)):
        d[eids[i]] = (nns[i], nnfs[i], sns[i], cns[i], nas[i], cls[i])
    return d

# ─── Evaluation ───────────────────────────────────────────────

def ef05(pred, true):
    if not true and not pred: return 1.0
    if not true or not pred: return 0.0
    tp = len(pred & true); fp = len(pred - true); fn = len(true - pred)
    p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
    return 1.25*p*r/(0.25*p+r) if p+r else 0.0

def evaluate(preds, gt):
    scores = []; tp=fp=fn=0; st=sc=0
    for s1, true in gt.items():
        pred = preds.get(s1, set())
        scores.append(ef05(pred, true))
        tp += len(pred & true); fp += len(pred - true); fn += len(true - pred)
        if not true: st += 1; sc += (1 if not pred else 0)
    return dict(f05=np.mean(scores), precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn) if tp+fn else 0, tp=tp, fp=fp, fn=fn,
                singleton_acc=sc/st if st else 1.0, n=len(gt))

# ─── I/O helpers ──────────────────────────────────────────────

def load_gt():
    gt_raw = pd.read_csv(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'), sep='\t', dtype=str)
    gt_raw['matched_entity_ids'] = gt_raw['matched_entity_ids'].fillna('')
    keys = gt_raw['source1_entity_id'].values
    vals = gt_raw['matched_entity_ids'].values
    gt = {}
    for i in range(len(keys)):
        m = vals[i]
        gt[keys[i]] = set(m.split(',')) if m else set()
    return gt

def load_source_country(data_dir, prefix, src_num, country):
    path = os.path.join(data_dir, f'{prefix}_source{src_num}.tsv')
    df = pd.read_csv(path, sep='\t', dtype=str)
    return df[df['country'] == country].reset_index(drop=True)

def load_and_prep(data_dir, prefix, country):
    t0 = time.time()
    s1 = load_source_country(data_dir, prefix, 1, country)
    s2 = load_source_country(data_dir, prefix, 2, country)
    s3 = load_source_country(data_dir, prefix, 3, country)
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3; gc.collect()
    s1 = preprocess(s1)
    s2s3 = preprocess(s2s3)
    log(f"  Loaded: S1={len(s1):,}, S2S3={len(s2s3):,} in {time.time()-t0:.0f}s")
    return s1, s2s3

# ─── Main ─────────────────────────────────────────────────────

def main():
    log(f"Business Entity Resolution Pipeline v5 (token blocking)")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    ts = time.time()

    # ── PHASE 1: Train & Validate ──
    log("\n" + "="*60)
    log("PHASE 1: TRAINING & VALIDATION")
    log("="*60)

    gt = load_gt()
    log(f"Ground truth: {len(gt):,}")

    s1_meta = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source1.tsv'), sep='\t',
                           dtype=str, usecols=['entity_id', 'country'])
    np.random.seed(SEED)
    all_ids = s1_meta['entity_id'].values.copy()
    np.random.shuffle(all_ids)
    vn = int(len(all_ids) * 0.2)
    val_set = set(all_ids[:vn])
    train_set = set(all_ids[vn:])
    countries_train = sorted(s1_meta['country'].unique())
    del s1_meta; gc.collect()
    log(f"Train: {len(train_set):,}, Val: {len(val_set):,}, Countries: {countries_train}")

    train_X_all, train_y_all = [], []
    val_X_all, val_y_all, val_s1_all, val_c_all = [], [], [], []
    val_cand_all = {}
    val_sampled_eids = set()

    TRAIN_SAMPLE = 30_000
    VAL_SAMPLE = 30_000

    for country in countries_train:
        log(f"\n--- {country} ---")
        s1, s2s3 = load_and_prep(TRAIN_DIR, 'train', country)

        s1_tr = s1[s1['entity_id'].isin(train_set)]
        s1_vl = s1[s1['entity_id'].isin(val_set)]
        if len(s1_tr) > TRAIN_SAMPLE:
            s1_tr = s1_tr.sample(TRAIN_SAMPLE, random_state=SEED)
        if len(s1_vl) > VAL_SAMPLE:
            s1_vl = s1_vl.sample(VAL_SAMPLE, random_state=SEED)
        val_sampled_eids.update(s1_vl['entity_id'].values)
        log(f"  Train S1: {len(s1_tr):,}, Val S1: {len(s1_vl):,}")

        # Training pairs with combined blocking
        log(f"  Blocking train...")
        t = time.time()
        tr_pairs = blocking_combined(s1_tr, s2s3)
        log(f"  Blocked: {len(tr_pairs):,} pairs in {time.time()-t:.0f}s")

        # Inject GT positives for training
        s2s3_set = set(s2s3['entity_id'].values)
        inject = []
        for eid in s1_tr['entity_id'].values:
            for mid in gt.get(eid, set()):
                if mid in s2s3_set:
                    inject.append({'entity_id_s1': eid, 'entity_id_s2s3': mid, 'block': 'G', 'n_blocks': 0})
        if inject:
            tr_pairs = pd.concat([tr_pairs, pd.DataFrame(inject)], ignore_index=True)
            tr_pairs = tr_pairs.drop_duplicates(subset=['entity_id_s1', 'entity_id_s2s3'], keep='first')

        s1_lk = make_lookup(s1_tr)
        s2s3_lk = make_lookup(s2s3)

        log(f"  Features train ({len(tr_pairs):,})...")
        t = time.time()
        X = compute_features_batch(tr_pairs, s1_lk, s2s3_lk)
        y = np.zeros(len(tr_pairs), dtype=np.int32)
        for i, (s1id, s2id) in enumerate(zip(tr_pairs['entity_id_s1'].values,
                                               tr_pairs['entity_id_s2s3'].values)):
            if s2id in gt.get(s1id, set()): y[i] = 1
        train_X_all.append(X); train_y_all.append(y)
        log(f"  Train: {len(y):,} pairs, {y.sum():,} pos in {time.time()-t:.0f}s")

        # Validation pairs
        log(f"  Blocking val...")
        t = time.time()
        vl_pairs = blocking_combined(s1_vl, s2s3)
        log(f"  Val blocked: {len(vl_pairs):,} pairs in {time.time()-t:.0f}s")

        s1_lk_vl = make_lookup(s1_vl)
        log(f"  Features val...")
        t = time.time()
        Xv = compute_features_batch(vl_pairs, s1_lk_vl, s2s3_lk)
        yv = np.zeros(len(vl_pairs), dtype=np.int32)
        vl_s1_ids = vl_pairs['entity_id_s1'].values
        vl_c_ids = vl_pairs['entity_id_s2s3'].values
        for i in range(len(yv)):
            if vl_c_ids[i] in gt.get(vl_s1_ids[i], set()): yv[i] = 1
        val_X_all.append(Xv); val_y_all.append(yv)
        val_s1_all.extend(vl_s1_ids.tolist())
        val_c_all.extend(vl_c_ids.tolist())
        for s1id, s2id in zip(vl_s1_ids, vl_c_ids):
            val_cand_all.setdefault(s1id, set()).add(s2id)
        log(f"  Val: {len(yv):,} pairs, {yv.sum():,} pos in {time.time()-t:.0f}s")

        del s1, s2s3, s1_lk, s2s3_lk, s1_lk_vl; gc.collect()

    Xtr = np.vstack(train_X_all); ytr = np.concatenate(train_y_all)
    Xvl = np.vstack(val_X_all);   yvl = np.concatenate(val_y_all)
    del train_X_all, train_y_all, val_X_all, val_y_all; gc.collect()
    log(f"\nTotal train: {len(ytr):,} ({ytr.sum():,.0f} pos), val: {len(yvl):,} ({yvl.sum():,.0f} pos)")

    # Train LightGBM
    log("\nTraining LightGBM...")
    model = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss', verbosity=-1,
        n_estimators=400, learning_rate=0.08, num_leaves=63, max_depth=7,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        is_unbalance=True,
        random_state=SEED, n_jobs=-1)
    model.fit(Xtr, ytr, feature_name=FEAT_NAMES)
    log("Feature importance (top 15):")
    for n, v in sorted(zip(FEAT_NAMES, model.feature_importances_), key=lambda x: -x[1])[:15]:
        log(f"  {n}: {v}")

    # Score val
    val_scores = model.predict_proba(Xvl)[:, 1]

    # Evaluate only on sampled val entities (not ALL val entities)
    val_gt = {}
    for eid in val_sampled_eids:
        val_gt[eid] = gt.get(eid, set())
        if eid not in val_cand_all:
            val_cand_all[eid] = set()

    # Candidate recall
    ch = ct = tc = 0
    for sid, trues in val_gt.items():
        cs = val_cand_all.get(sid, set())
        tc += len(cs)
        if trues: ch += len(trues & cs); ct += len(trues)
    can_recall = ch/ct if ct else 0.0
    avg_cands = tc/len(val_gt) if val_gt else 0.0
    log(f"\nCandidate recall: {can_recall:.4f}, Avg cands/S1: {avg_cands:.1f}")

    # Threshold optimization
    THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                  0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98]
    best_t, best_f = 0.5, -1
    log("\nThreshold optimization:")
    for t in THRESHOLDS:
        preds = {}
        for s1, cid, sc in zip(val_s1_all, val_c_all, val_scores):
            if sc >= t: preds.setdefault(s1, set()).add(cid)
        for s1 in val_gt: preds.setdefault(s1, set())
        m = evaluate(preds, val_gt)
        mk = " <--" if m['f05'] > best_f else ""
        if m['f05'] > best_f: best_f = m['f05']; best_t = t
        log(f"  T={t:.2f}: F0.5={m['f05']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
            f"TP={m['tp']} FP={m['fp']} FN={m['fn']}{mk}")

    # Final val
    preds = {}
    for s1, cid, sc in zip(val_s1_all, val_c_all, val_scores):
        if sc >= best_t: preds.setdefault(s1, set()).add(cid)
    for s1 in val_gt: preds.setdefault(s1, set())
    vm = evaluate(preds, val_gt)
    log(f"\nBest threshold: {best_t}")
    log(f"Val F0.5:      {vm['f05']:.4f}")
    log(f"Val Precision: {vm['precision']:.4f}")
    log(f"Val Recall:    {vm['recall']:.4f}")
    log(f"Singleton acc: {vm['singleton_acc']:.4f}")
    log(f"Phase 1: {(time.time()-ts)/60:.1f} min")

    del Xtr, ytr, Xvl, yvl, gt; gc.collect()

    # ── PHASE 2: Test Inference ──
    log(f"\n{'='*60}")
    log("PHASE 2: TEST INFERENCE")
    log(f"{'='*60}")
    t0 = time.time()

    s1_test_meta = pd.read_csv(os.path.join(TEST_DIR, 'test_source1.tsv'), sep='\t',
                                dtype=str, usecols=['entity_id', 'country'])
    test_s1_ids = s1_test_meta['entity_id'].tolist()
    test_countries = sorted(s1_test_meta['country'].unique())
    log(f"Test S1: {len(test_s1_ids):,}, Countries: {test_countries}")
    del s1_test_meta; gc.collect()

    all_results = {}
    all_candidates = {}

    for country in test_countries:
        log(f"\n--- {country} ---")
        s1c, s2s3c = load_and_prep(TEST_DIR, 'test', country)

        log(f"  Blocking...")
        t1 = time.time()
        pairs = blocking_combined(s1c, s2s3c)
        log(f"  Blocked: {len(pairs):,} pairs in {time.time()-t1:.0f}s")

        # Build candidate dict
        for s1id, s2id in zip(pairs['entity_id_s1'].values, pairs['entity_id_s2s3'].values):
            all_candidates.setdefault(s1id, set()).add(s2id)

        if len(pairs) > 0:
            s1_lk = make_lookup(s1c)
            s2s3_lk = make_lookup(s2s3c)

            CHUNK = 2_000_000
            for start in range(0, len(pairs), CHUNK):
                end = min(start + CHUNK, len(pairs))
                chunk = pairs.iloc[start:end]
                log(f"  Features+scoring {start:,}-{end:,}...")
                t1 = time.time()
                X = compute_features_batch(chunk, s1_lk, s2s3_lk)
                scores = model.predict_proba(X)[:, 1]
                for s1id, cid, sc in zip(chunk['entity_id_s1'].values,
                                          chunk['entity_id_s2s3'].values, scores):
                    if sc >= best_t:
                        all_results.setdefault(s1id, set()).add(cid)
                log(f"  Chunk done in {time.time()-t1:.0f}s")
                del X; gc.collect()

            del s1_lk, s2s3_lk

        del s1c, s2s3c, pairs; gc.collect()

    # Ensure every test S1 has an entry
    for eid in test_s1_ids:
        all_results.setdefault(eid, set())
        all_candidates.setdefault(eid, set())

    log(f"\nTest inference: {(time.time()-t0)/60:.1f} min")

    # ── PHASE 3: Write outputs ──
    log(f"\n{'='*60}")
    log("PHASE 3: WRITING OUTPUTS")
    log(f"{'='*60}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    nm = sum(1 for s in test_s1_ids if all_results.get(s))
    ns = sum(1 for s in test_s1_ids if not all_results.get(s))
    nl = sum(len(all_results.get(s, set())) for s in test_s1_ids)
    log(f"  S1: {len(test_s1_ids):,}, matched: {nm:,}, singletons: {ns:,}, links: {nl:,}")

    mp = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    with open(mp, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in test_s1_ids:
            ms = all_results.get(s, set())
            f.write(f"{s}\t{','.join(sorted(ms)) if ms else ''}\n")

    cp = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    with open(cp, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s in test_s1_ids:
            cs = all_candidates.get(s, set())
            f.write(f"{s}\t{','.join(sorted(cs)) if cs else ''}\n")

    bad = sum(1 for s in test_s1_ids if all_results.get(s, set()) - all_candidates.get(s, set()))
    log(f"  Matches ⊆ candidates: {'OK' if not bad else f'WARNING: {bad}'}")
    log(f"  Written: {mp}")
    log(f"  Written: {cp}")

    log(f"\n{'='*60}")
    log(f"COMPLETE in {(time.time()-ts)/60:.1f} min")
    log(f"Threshold: {best_t}")
    log(f"Val F0.5: {vm['f05']:.4f}, P: {vm['precision']:.4f}, R: {vm['recall']:.4f}")
    log(f"Candidate recall: {can_recall:.4f}, Avg cands/S1: {avg_cands:.1f}")
    log(f"{'='*60}")

if __name__ == '__main__':
    main()
