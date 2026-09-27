#!/usr/bin/env python3
"""
Business Entity Resolution Pipeline
Processes by country partition for memory efficiency.
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
    MATCHING_RESULTS, CANDIDATE_PAIRS, OUTPUT_DIR,
    RANDOM_SEED, VAL_FRACTION, TRAIN_SAMPLE_SIZE,
    THRESHOLD_CANDIDATES,
)
from src.data_loader import load_source, preprocess_source, load_ground_truth
from src.blocking import BlockingEngine
from src.features import compute_pair_features, FEATURE_NAMES
from src.matcher import EntityMatcher
from src.evaluation import evaluate_predictions, evaluate_candidates


def generate_pairs_for_partition(s1_df, s2s3_df, gt_dict=None, max_per_s1=100):
    """Generate candidate pairs and features for a country partition."""
    if len(s2s3_df) == 0:
        return [], [], [], [], {}

    print(f"    Building blocking indexes for {len(s2s3_df):,} S2/S3 entities...")
    engine = BlockingEngine(s2s3_df)

    s2s3_lookup = s2s3_df.set_index('entity_id').to_dict('index')

    all_features = []
    all_labels = []
    all_s1_ids = []
    all_cand_ids = []
    candidate_dict = {}

    for _, s1_row in tqdm(s1_df.iterrows(), total=len(s1_df), desc="    Blocking+Features"):
        s1_id = s1_row['entity_id']
        candidates, methods = engine.generate_candidates(s1_row)
        candidate_dict[s1_id] = candidates

        true_matches = gt_dict.get(s1_id, set()) if gt_dict else set()

        if gt_dict is not None and true_matches:
            for mid in true_matches:
                if mid in s2s3_lookup:
                    candidates.add(mid)
                    if mid not in methods:
                        methods[mid] = {'gt_inject'}

        for cand_id in candidates:
            if cand_id not in s2s3_lookup:
                continue

            cand_row = s2s3_lookup[cand_id]
            feat = compute_pair_features(s1_row, cand_row, methods.get(cand_id, set()))
            all_features.append(feat)
            all_s1_ids.append(s1_id)
            all_cand_ids.append(cand_id)

            if gt_dict is not None:
                label = 1 if cand_id in true_matches else 0
                all_labels.append(label)

    return all_features, all_labels, all_s1_ids, all_cand_ids, candidate_dict


def run_training_pipeline():
    """Train and validate the entity resolution model."""
    print("=" * 60)
    print("PHASE 1: TRAINING & VALIDATION")
    print("=" * 60)

    print("\nLoading training data...")
    t0 = time.time()
    s1_raw = load_source(TRAIN_S1)
    print(f"  S1: {len(s1_raw):,} entities")

    gt_dict = load_ground_truth(TRAIN_GT)
    print(f"  Ground truth: {len(gt_dict):,} entries")

    np.random.seed(RANDOM_SEED)
    all_s1_ids = s1_raw['entity_id'].unique()
    np.random.shuffle(all_s1_ids)

    val_size = int(len(all_s1_ids) * VAL_FRACTION)
    val_ids = set(all_s1_ids[:val_size])
    train_ids = set(all_s1_ids[val_size:])

    train_sample = set(np.random.choice(
        list(train_ids), size=min(TRAIN_SAMPLE_SIZE, len(train_ids)), replace=False
    ))

    print(f"  Train sample: {len(train_sample):,}")
    print(f"  Validation: {len(val_ids):,}")

    s1_train = s1_raw[s1_raw['entity_id'].isin(train_sample)]
    s1_val = s1_raw[s1_raw['entity_id'].isin(val_ids)]
    del s1_raw
    gc.collect()

    print(f"\nPreprocessing S1 train ({len(s1_train):,})...")
    s1_train = preprocess_source(s1_train)
    print(f"Preprocessing S1 val ({len(s1_val):,})...")
    s1_val = preprocess_source(s1_val)

    countries = sorted(set(s1_train['country'].unique()) | set(s1_val['country'].unique()))
    print(f"Countries: {countries}")

    train_gt = {k: v for k, v in gt_dict.items() if k in train_sample}
    val_gt = {k: v for k, v in gt_dict.items() if k in val_ids}
    del gt_dict
    gc.collect()

    all_train_features = []
    all_train_labels = []
    all_val_features = []
    all_val_labels = []
    all_val_s1_ids = []
    all_val_cand_ids = []
    val_candidate_dict = {}

    for country in countries:
        print(f"\n--- Processing country: {country} ---")

        s1_tr_c = s1_train[s1_train['country'] == country]
        s1_val_c = s1_val[s1_val['country'] == country]
        print(f"  Train S1: {len(s1_tr_c):,}, Val S1: {len(s1_val_c):,}")

        if len(s1_tr_c) == 0 and len(s1_val_c) == 0:
            continue

        print(f"  Loading S2 for {country}...")
        s2_raw = load_source(TRAIN_S2)
        s2_c = s2_raw[s2_raw['country'] == country]
        del s2_raw
        gc.collect()

        print(f"  Loading S3 for {country}...")
        s3_raw = load_source(TRAIN_S3)
        s3_c = s3_raw[s3_raw['country'] == country]
        del s3_raw
        gc.collect()

        s2s3 = pd.concat([s2_c, s3_c], ignore_index=True)
        del s2_c, s3_c
        gc.collect()
        print(f"  S2+S3 for {country}: {len(s2s3):,}")

        print(f"  Preprocessing S2+S3...")
        s2s3 = preprocess_source(s2s3)

        if len(s1_tr_c) > 0:
            print(f"\n  --- Training pairs for {country} ---")
            feats, labels, _, _, _ = generate_pairs_for_partition(
                s1_tr_c, s2s3, train_gt
            )
            all_train_features.extend(feats)
            all_train_labels.extend(labels)
            print(f"  Train pairs: {len(feats):,} (pos: {sum(labels):,})")

        if len(s1_val_c) > 0:
            print(f"\n  --- Validation pairs for {country} ---")
            feats, labels, s1_ids, cand_ids, cand_dict = generate_pairs_for_partition(
                s1_val_c, s2s3, val_gt
            )
            all_val_features.extend(feats)
            all_val_labels.extend(labels)
            all_val_s1_ids.extend(s1_ids)
            all_val_cand_ids.extend(cand_ids)
            val_candidate_dict.update(cand_dict)
            print(f"  Val pairs: {len(feats):,} (pos: {sum(labels):,})")

        del s2s3
        gc.collect()

    print(f"\n{'=' * 60}")
    print(f"Total train pairs: {len(all_train_features):,}")
    print(f"  Positive: {sum(all_train_labels):,}")
    print(f"  Negative: {len(all_train_labels) - sum(all_train_labels):,}")
    print(f"Total val pairs: {len(all_val_features):,}")
    print(f"  Positive: {sum(all_val_labels):,}")
    print(f"  Negative: {len(all_val_labels) - sum(all_val_labels):,}")

    X_train = np.array([[f[k] for k in FEATURE_NAMES] for f in all_train_features])
    y_train = np.array(all_train_labels)
    X_val = np.array([[f[k] for k in FEATURE_NAMES] for f in all_val_features])
    y_val = np.array(all_val_labels)

    del all_train_features, all_val_features
    gc.collect()

    print(f"\nTraining LightGBM model...")
    matcher = EntityMatcher()
    matcher.train(X_train, y_train)

    print("\nFeature importance (top 15):")
    imp = matcher.feature_importance()
    for name, score in list(imp.items())[:15]:
        print(f"  {name}: {score}")

    print(f"\nScoring validation pairs...")
    val_scores = matcher.predict_proba(X_val)

    print(f"\nOptimizing threshold for F0.5...")
    best_threshold, best_f05 = matcher.optimize_threshold(
        val_scores, all_val_s1_ids, all_val_cand_ids, val_gt
    )
    print(f"  Best threshold: {best_threshold}")
    print(f"  Best val F0.5: {best_f05:.4f}")

    print("\nDetailed threshold analysis:")
    for threshold in THRESHOLD_CANDIDATES:
        predictions = {}
        for s1_id, cand_id, score in zip(all_val_s1_ids, all_val_cand_ids, val_scores):
            if score >= threshold:
                if s1_id not in predictions:
                    predictions[s1_id] = set()
                predictions[s1_id].add(cand_id)
        for s1_id in val_gt:
            if s1_id not in predictions:
                predictions[s1_id] = set()

        metrics = evaluate_predictions(predictions, val_gt)
        marker = " <-- BEST" if threshold == best_threshold else ""
        print(f"  T={threshold:.2f}: F0.5={metrics['macro_f05']:.4f} "
              f"P={metrics['micro_precision']:.4f} R={metrics['micro_recall']:.4f} "
              f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}"
              f"{marker}")

    cand_metrics = evaluate_candidates(val_candidate_dict, val_gt)
    print(f"\nCandidate quality:")
    print(f"  Candidate recall: {cand_metrics['candidate_recall']:.4f}")
    print(f"  Avg candidates/S1: {cand_metrics['avg_candidates']:.1f}")

    final_predictions = {}
    for s1_id, cand_id, score in zip(all_val_s1_ids, all_val_cand_ids, val_scores):
        if score >= best_threshold:
            if s1_id not in final_predictions:
                final_predictions[s1_id] = set()
            final_predictions[s1_id].add(cand_id)
    for s1_id in val_gt:
        if s1_id not in final_predictions:
            final_predictions[s1_id] = set()

    final_metrics = evaluate_predictions(final_predictions, val_gt)
    print(f"\nFinal validation results:")
    print(f"  F0.5 (macro): {final_metrics['macro_f05']:.4f}")
    print(f"  Precision (micro): {final_metrics['micro_precision']:.4f}")
    print(f"  Recall (micro): {final_metrics['micro_recall']:.4f}")
    print(f"  TP: {final_metrics['tp']}")
    print(f"  FP: {final_metrics['fp']}")
    print(f"  FN: {final_metrics['fn']}")
    print(f"  Singleton accuracy: {final_metrics['singleton_accuracy']:.4f}")

    elapsed = time.time() - t0
    print(f"\nTraining phase completed in {elapsed/60:.1f} minutes")

    return matcher, final_metrics, cand_metrics


def run_test_inference(matcher):
    """Run inference on the test set by country partition."""
    print(f"\n{'=' * 60}")
    print("PHASE 2: TEST INFERENCE")
    print("=" * 60)

    t0 = time.time()

    print("\nLoading test S1...")
    s1_raw = load_source(TEST_S1)
    print(f"  Test S1: {len(s1_raw):,} entities")

    countries = sorted(s1_raw['country'].unique())
    print(f"  Countries: {countries}")

    all_results = {}
    all_candidates = {}

    for country in countries:
        print(f"\n--- Test inference: {country} ---")

        s1_c = s1_raw[s1_raw['country'] == country].copy()
        print(f"  S1 entities: {len(s1_c):,}")

        print(f"  Preprocessing S1...")
        s1_c = preprocess_source(s1_c)

        print(f"  Loading S2 for {country}...")
        s2_raw = load_source(TEST_S2)
        s2_c = s2_raw[s2_raw['country'] == country]
        del s2_raw
        gc.collect()

        print(f"  Loading S3 for {country}...")
        s3_raw = load_source(TEST_S3)
        s3_c = s3_raw[s3_raw['country'] == country]
        del s3_raw
        gc.collect()

        s2s3 = pd.concat([s2_c, s3_c], ignore_index=True)
        del s2_c, s3_c
        gc.collect()
        print(f"  S2+S3: {len(s2s3):,}")

        print(f"  Preprocessing S2+S3...")
        s2s3 = preprocess_source(s2s3)

        print(f"  Building blocking indexes...")
        engine = BlockingEngine(s2s3)
        s2s3_lookup = s2s3.set_index('entity_id').to_dict('index')

        batch_s1_ids = []
        batch_cand_ids = []
        batch_features = []

        for _, s1_row in tqdm(s1_c.iterrows(), total=len(s1_c), desc=f"  Blocking {country}"):
            s1_id = s1_row['entity_id']
            candidates, methods = engine.generate_candidates(s1_row)
            all_candidates[s1_id] = candidates

            for cand_id in candidates:
                if cand_id not in s2s3_lookup:
                    continue
                cand_row = s2s3_lookup[cand_id]
                feat = compute_pair_features(s1_row, cand_row, methods.get(cand_id, set()))
                batch_features.append(feat)
                batch_s1_ids.append(s1_id)
                batch_cand_ids.append(cand_id)

        if batch_features:
            print(f"  Scoring {len(batch_features):,} candidate pairs...")
            X = np.array([[f[k] for k in FEATURE_NAMES] for f in batch_features])
            scores = matcher.predict_proba(X)

            for s1_id, cand_id, score in zip(batch_s1_ids, batch_cand_ids, scores):
                if score >= matcher.threshold:
                    if s1_id not in all_results:
                        all_results[s1_id] = set()
                    all_results[s1_id].add(cand_id)

        del s2s3, s2s3_lookup, engine, batch_features
        gc.collect()
        print(f"  {country} done.")

    for _, row in s1_raw.iterrows():
        s1_id = row['entity_id']
        if s1_id not in all_results:
            all_results[s1_id] = set()
        if s1_id not in all_candidates:
            all_candidates[s1_id] = set()

    elapsed = time.time() - t0
    print(f"\nTest inference completed in {elapsed/60:.1f} minutes")

    return all_results, all_candidates, s1_raw


def write_output_files(results, candidates, s1_df):
    """Write matching_results.tsv and candidate_pairs.tsv."""
    print(f"\n{'=' * 60}")
    print("PHASE 3: WRITING OUTPUT FILES")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    s1_ids_ordered = s1_df['entity_id'].tolist()

    with_matches = sum(1 for s1_id in s1_ids_ordered if results.get(s1_id))
    singletons = sum(1 for s1_id in s1_ids_ordered if not results.get(s1_id))
    total_links = sum(len(results.get(s1_id, set())) for s1_id in s1_ids_ordered)

    print(f"  Total S1 entities: {len(s1_ids_ordered):,}")
    print(f"  With predicted matches: {with_matches:,}")
    print(f"  Singletons: {singletons:,}")
    print(f"  Total predicted links: {total_links:,}")

    print(f"\nWriting {MATCHING_RESULTS}...")
    with open(MATCHING_RESULTS, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids_ordered:
            matches = results.get(s1_id, set())
            match_str = ','.join(sorted(matches)) if matches else ''
            f.write(f"{s1_id}\t{match_str}\n")

    print(f"Writing {CANDIDATE_PAIRS}...")
    with open(CANDIDATE_PAIRS, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids_ordered:
            cands = candidates.get(s1_id, set())
            cand_str = ','.join(sorted(cands)) if cands else ''
            f.write(f"{s1_id}\t{cand_str}\n")

    print("Output files written successfully.")

    matched_not_in_candidates = 0
    for s1_id in s1_ids_ordered:
        matches = results.get(s1_id, set())
        cands = candidates.get(s1_id, set())
        if matches - cands:
            matched_not_in_candidates += 1
    if matched_not_in_candidates > 0:
        print(f"  WARNING: {matched_not_in_candidates} S1 entities have matches not in candidates!")
    else:
        print("  All matches are subset of candidates.")


def main():
    print("Business Entity Resolution Pipeline")
    print(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t_start = time.time()

    matcher, val_metrics, cand_metrics = run_training_pipeline()
    results, candidates, s1_test = run_test_inference(matcher)
    write_output_files(results, candidates, s1_test)

    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Validation F0.5: {val_metrics['macro_f05']:.4f}")
    print(f"Candidate recall: {cand_metrics['candidate_recall']:.4f}")
    print(f"Threshold: {matcher.threshold}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
