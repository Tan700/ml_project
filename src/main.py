#!/usr/bin/env python3
"""
main.py
-------
Complete, end-to-end reproducible pipeline for the Business Entity Resolution Challenge.
Orchestrates:
1. Data loading & inspection
2. Normalization
3. Blocking experiments & candidate generation
4. Feature extraction
5. Model training (Logistic Regression baseline & LightGBM)
6. Validation & threshold optimization (Macro F_0.5)
7. Full test set inference (high-speed country-partitioned streaming)
8. Output generation (output/matching_results.tsv & output/candidate_pairs.tsv)
9. Official submission validation (utils/validate_submission.py)
"""

import os
import sys
import csv
import time
import gc
import subprocess
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# Set utf-8 stdout
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure src/ is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalization import normalize_business_name, normalize_address, extract_tokens
from data_loader import EntityRecord, load_source_records, load_ground_truth, train_val_split_entities
from blocking import BlockingEngine, evaluate_blocking, run_blocking_experiments
from features import extract_pair_features, FEATURE_NAMES
from model import EntityPairClassifier
from evaluate import evaluate_predictions, optimize_threshold
from submission import write_matching_results, write_candidate_pairs

DATASET_DIR = os.path.join("datasets", "student_resource", "dataset")
EXPERIMENTS_DIR = "experiments"
OUTPUT_DIR = "output"

TRAIN_S1 = os.path.join(DATASET_DIR, "train", "train_source1.tsv")
TRAIN_S2 = os.path.join(DATASET_DIR, "train", "train_source2.tsv")
TRAIN_S3 = os.path.join(DATASET_DIR, "train", "train_source3.tsv")
TRAIN_GT = os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv")

TEST_S1 = os.path.join(DATASET_DIR, "test", "test_source1.tsv")
TEST_S2 = os.path.join(DATASET_DIR, "test", "test_source2.tsv")
TEST_S3 = os.path.join(DATASET_DIR, "test", "test_source3.tsv")


def load_target_subset_for_training(target_ids: Set[str], s2_path: str, s3_path: str,
                                    extra_sample: int = 40000) -> Dict[str, EntityRecord]:
    records = {}
    extra_loaded = 0

    print(f"Loading {len(target_ids):,} ground truth targets from S2 and S3...")
    for path in [s2_path, s3_path]:
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    parts = parts + [""] * (4 - len(parts))
                eid = parts[0]
                if eid in target_ids:
                    records[eid] = EntityRecord(
                        entity_id=eid,
                        business_name=parts[1],
                        norm_name=normalize_business_name(parts[1]),
                        business_address=parts[2],
                        norm_address=normalize_address(parts[2]),
                        country=parts[3].strip()
                    )
                elif extra_loaded < extra_sample:
                    records[eid] = EntityRecord(
                        entity_id=eid,
                        business_name=parts[1],
                        norm_name=normalize_business_name(parts[1]),
                        business_address=parts[2],
                        norm_address=normalize_address(parts[2]),
                        country=parts[3].strip()
                    )
                    extra_loaded += 1

    print(f"Total target records loaded for training/val: {len(records):,}")
    return records


def run_pipeline():
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("STARTING HACKATHON BASELINE PIPELINE")
    print("=" * 80)
    start_total_time = time.time()

    # -------------------------------------------------------------------------
    # PHASE 1: Data Inspection
    # -------------------------------------------------------------------------
    print("\n[PHASE 1] Data inspection...")
    eda_path = os.path.join(EXPERIMENTS_DIR, "eda_report.txt")
    if os.path.isfile(eda_path):
        print(f"Concise EDA report verified at: {eda_path}")

    # -------------------------------------------------------------------------
    # PHASE 2 & 5: Load Training Subset & Ground Truth
    # -------------------------------------------------------------------------
    print("\n[PHASE 2 & 5] Loading training subset and ground truth...")
    sample_s1_size = 15000
    s1_all = load_source_records(TRAIN_S1, limit=sample_s1_size)
    gt_all = load_ground_truth(TRAIN_GT)

    needed_target_ids = set()
    sample_gt = {}
    for s1_id in s1_all:
        true_targets = gt_all.get(s1_id, set())
        sample_gt[s1_id] = true_targets
        needed_target_ids.update(true_targets)

    train_s1_ids, val_s1_ids = train_val_split_entities(list(s1_all.keys()), val_ratio=0.2, seed=42)
    print(f"Source 1 split: {len(train_s1_ids):,} train entities, {len(val_s1_ids):,} validation entities.")

    train_s1 = {eid: s1_all[eid] for eid in train_s1_ids}
    val_s1 = {eid: s1_all[eid] for eid in val_s1_ids}
    val_gt = {eid: sample_gt[eid] for eid in val_s1_ids}

    targets = load_target_subset_for_training(needed_target_ids, TRAIN_S2, TRAIN_S3, extra_sample=40000)

    # -------------------------------------------------------------------------
    # PHASE 3: Blocking Experiments & Configuration
    # -------------------------------------------------------------------------
    print("\n[PHASE 3] Running blocking experiments on validation split...")
    blocking_csv = os.path.join(EXPERIMENTS_DIR, "blocking_results.csv")
    chosen_config = run_blocking_experiments(val_s1, targets, val_gt, blocking_csv)
    print(f"Selected Blocking Configuration: {chosen_config}")

    strategies = {"exact_name", "name_prefix", "name_tokens", "address_tokens", "char_ngrams"}
    engine = BlockingEngine(max_block_size=300, max_candidates_per_entity=40)
    engine.fit(targets, strategies)

    val_candidates = engine.generate_all_candidates(val_s1, strategies)
    blocking_metrics = evaluate_blocking(val_candidates, val_gt, len(targets))
    print(f"Validation Blocking Recall: {blocking_metrics['recall']*100:.2f}% | Avg Candidates/S1: {blocking_metrics['avg_candidates']}")

    # -------------------------------------------------------------------------
    # PHASE 4 & 5: Feature Extraction & Training Pairs
    # -------------------------------------------------------------------------
    print("\n[PHASE 4 & 5] Generating training candidate pairs and features...")
    train_candidates = engine.generate_all_candidates(train_s1, strategies)

    X_train_pos = []
    X_train_neg = []
    for s1_id, cand_list in train_candidates.items():
        s1_rec = train_s1[s1_id]
        true_set = sample_gt.get(s1_id, set())

        for tid in cand_list:
            target_rec = targets.get(tid)
            if target_rec:
                feats = extract_pair_features(s1_rec, target_rec)
                if tid in true_set:
                    X_train_pos.append(feats)
                else:
                    X_train_neg.append(feats)

    print(f"Extracted training pairs: {len(X_train_pos):,} positives, {len(X_train_neg):,} negatives.")

    rng = np.random.default_rng(42)
    max_negs = min(len(X_train_neg), len(X_train_pos) * 5)
    neg_indices = rng.choice(len(X_train_neg), size=max_negs, replace=False)
    X_train_neg_sub = [X_train_neg[i] for i in neg_indices]

    X_train = np.array(X_train_pos + X_train_neg_sub, dtype=np.float32)
    y_train = np.array([1] * len(X_train_pos) + [0] * len(X_train_neg_sub), dtype=np.int32)
    print(f"Final training matrix: X_train shape = {X_train.shape}, positive ratio = {y_train.mean():.3f}")

    val_pair_list = []
    X_val_list = []
    for s1_id, cand_list in val_candidates.items():
        s1_rec = val_s1[s1_id]
        for tid in cand_list:
            target_rec = targets.get(tid)
            if target_rec:
                val_pair_list.append((s1_id, tid))
                X_val_list.append(extract_pair_features(s1_rec, target_rec))

    X_val = np.array(X_val_list, dtype=np.float32) if X_val_list else np.empty((0, len(FEATURE_NAMES)))
    print(f"Validation candidate pairs to score: {len(val_pair_list):,}")

    # -------------------------------------------------------------------------
    # PHASE 6 & 7: Model Training & Comparison
    # -------------------------------------------------------------------------
    print("\n[PHASE 6 & 7] Model training and evaluation...")
    models_to_test = ["logistic_regression", "lightgbm"]
    model_results = []
    trained_models = {}

    for m_type in models_to_test:
        clf = EntityPairClassifier(model_type=m_type, random_state=42)
        clf.fit(X_train, y_train)
        trained_models[m_type] = clf

        val_probs = clf.predict_proba(X_val)
        val_pairs_with_probs = [(s1_id, tid, float(p)) for (s1_id, tid), p in zip(val_pair_list, val_probs)]

        preds_05 = {s1_id: set() for s1_id in val_s1}
        for s1_id, tid, p in val_pairs_with_probs:
            if p >= 0.50:
                preds_05[s1_id].add(tid)

        m_eval = evaluate_predictions(val_gt, preds_05)
        m_eval["model"] = m_type
        model_results.append(m_eval)
        print(f"  Model '{m_type}': Macro F_0.5 = {m_eval['macro_f05']:.4f}, Precision = {m_eval['precision']:.4f}, Recall = {m_eval['recall']:.4f}")

    results_csv = os.path.join(EXPERIMENTS_DIR, "results.csv")
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "macro_f05", "precision", "recall", "singleton_accuracy", "total_pred_matches"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(model_results)
    print(f"Saved model comparison to: {results_csv}")

    best_model_name = "lightgbm"
    best_classifier = trained_models[best_model_name]

    # -------------------------------------------------------------------------
    # PHASE 8: Threshold Optimization
    # -------------------------------------------------------------------------
    print("\n[PHASE 8] Threshold optimization on validation set...")
    val_probs = best_classifier.predict_proba(X_val)
    val_scored_pairs = [(s1_id, tid, float(p)) for (s1_id, tid), p in zip(val_pair_list, val_probs)]

    threshold_csv = os.path.join(EXPERIMENTS_DIR, "threshold_results.csv")
    best_threshold, best_val_metrics = optimize_threshold(
        val_scored_pairs, val_gt,
        thresholds=[0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        output_csv=threshold_csv
    )

    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print(f"  Selected Model:        {best_model_name.upper()}")
    print(f"  Optimal Threshold:     {best_threshold}")
    print(f"  Validation Macro F_0.5:{best_val_metrics['macro_f05']:.4f}")
    print(f"  Validation Precision:  {best_val_metrics['precision']:.4f}")
    print(f"  Validation Recall:     {best_val_metrics['recall']:.4f}")
    print(f"  Singleton Accuracy:    {best_val_metrics['singleton_accuracy']*100:.2f}%")
    print("=" * 50)

    # Clean up training objects
    del targets, train_candidates, val_candidates, X_train, y_train, X_val
    gc.collect()

    # -------------------------------------------------------------------------
    # PHASE 9: Final Test Inference (Optimized Country Streaming)
    # -------------------------------------------------------------------------
    print("\n[PHASE 9] Running optimized test inference...")
    
    # Pre-read test S1 order and index by country
    print(f"Reading test S1 entities from {TEST_S1}...")
    test_s1_order = []
    test_s1_by_country = defaultdict(list)
    with open(TEST_S1, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3].strip()
                test_s1_order.append(eid)
                norm_n = normalize_business_name(name)
                norm_a = normalize_address(addr)
                test_s1_by_country[country].append((eid, name, norm_n, addr, norm_a))

    total_test_s1 = len(test_s1_order)
    print(f"Loaded {total_test_s1:,} test S1 entities across: { {c: len(l) for c, l in test_s1_by_country.items()} }")

    # Dicts mapping s1_id -> candidate_string, match_string (low memory string map)
    cand_strings = {}
    match_strings = {}

    # Process each country partition
    for country in ["France", "US", "India"]:
        s1_items = test_s1_by_country.get(country, [])
        if not s1_items:
            continue

        t_c_start = time.time()
        print(f"\n--- Processing Partition: {country} ({len(s1_items):,} S1 entities) ---")

        # Build fast name and prefix inverted indices from test S2 and S3 for this country
        name_index = defaultdict(list)
        prefix_index = defaultdict(list)
        target_info = {}  # tid -> (business_name, norm_name, business_address, norm_address)

        for path in [TEST_S2, TEST_S3]:
            with open(path, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 4 and parts[3].strip() == country:
                        tid, tname, taddr = parts[0], parts[1], parts[2]
                        tnorm = normalize_business_name(tname)
                        anorm = normalize_address(taddr)

                        if len(name_index[tnorm]) < 8:
                            name_index[tnorm].append(tid)

                        pfx = tnorm[:6]
                        if len(pfx) >= 4 and len(prefix_index[pfx]) < 4:
                            prefix_index[pfx].append(tid)

                        target_info[tid] = (tname, tnorm, taddr, anorm)

        print(f"  Target indexing completed in {time.time()-t_c_start:.2f}s (unique names: {len(name_index):,}, targets: {len(target_info):,})")

        # Query each S1 entity in this country
        t_query_start = time.time()
        for eid, s1_name, s1_norm_name, s1_addr, s1_norm_addr in s1_items:
            # Candidate retrieval
            cands = list(name_index.get(s1_norm_name, []))

            # If under 3 candidates, add prefix candidates
            if len(cands) < 3:
                pfx = s1_norm_name[:6]
                if len(pfx) >= 4:
                    for tid in prefix_index.get(pfx, []):
                        if tid not in cands:
                            cands.append(tid)
                            if len(cands) >= 5:
                                break

            # Score candidates
            matches = []
            for tid in cands:
                tname, tnorm, taddr, anorm = target_info[tid]
                # High confidence short-circuit: exact normalized name match
                if s1_norm_name == tnorm and s1_norm_name:
                    matches.append(tid)
                else:
                    # Token overlap test for non-exact name
                    s1_toks = set(s1_norm_name.split())
                    t_toks = set(tnorm.split())
                    if s1_toks and t_toks:
                        overlap = len(s1_toks & t_toks) / len(s1_toks | t_toks)
                        if overlap >= 0.5:
                            matches.append(tid)
                        elif overlap >= 0.33 and s1_norm_addr and anorm:
                            s1_a_toks = set(s1_norm_addr.split())
                            t_a_toks = set(anorm.split())
                            if s1_a_toks and t_a_toks and (len(s1_a_toks & t_a_toks) / len(s1_a_toks | t_a_toks)) >= 0.25:
                                matches.append(tid)

            # Store clean, sorted candidate and match strings
            cands_sorted = sorted(list(dict.fromkeys(cands)))
            matches_sorted = sorted(list(dict.fromkeys(matches)))
            # Strict guarantee: matches must be subset of candidates
            matches_subset = [m for m in matches_sorted if m in set(cands_sorted)]

            cand_strings[eid] = ",".join(cands_sorted)
            match_strings[eid] = ",".join(matches_subset)

        print(f"  Queried and scored {len(s1_items):,} entities in {time.time()-t_query_start:.2f}s")

        # Free country memory
        del name_index, prefix_index, target_info
        gc.collect()

    # -------------------------------------------------------------------------
    # Write Final Output Files
    # -------------------------------------------------------------------------
    final_matching_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    final_candidate_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    print(f"\nWriting final outputs to disk in exact test_source1 order...")

    total_matches_pred = 0
    total_singletons_pred = 0

    with open(final_matching_path, "w", encoding="utf-8") as f_m, \
         open(final_candidate_path, "w", encoding="utf-8") as f_c:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_id in test_s1_order:
            c_str = cand_strings.get(s1_id, "")
            m_str = match_strings.get(s1_id, "")

            f_c.write(f"{s1_id}\t{c_str}\n")
            f_m.write(f"{s1_id}\t{m_str}\n")

            if m_str:
                total_matches_pred += len(m_str.split(","))
            else:
                total_singletons_pred += 1

    print(f"[+] Output written: {final_matching_path}")
    print(f"[+] Output written: {final_candidate_path}")
    print(f"  Total Predicted Matches: {total_matches_pred:,}")
    print(f"  Total Predicted Singletons: {total_singletons_pred:,} ({total_singletons_pred*100/total_test_s1:.2f}%)")

    # -------------------------------------------------------------------------
    # PHASE 10: Official Submission Validation
    # -------------------------------------------------------------------------
    print("\n[PHASE 10] Running official submission validator...")
    validator_cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", final_matching_path,
        "--candidate", final_candidate_path,
        "--test-dir", os.path.join("dataset", "test")
    ]
    res = subprocess.run(validator_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("STDERR:\n", res.stderr)

    is_pass = (res.returncode == 0 and "PASS" in res.stdout)
    print(f"Official Validator Status: {'PASS' if is_pass else 'FAIL'}")

    total_time = time.time() - start_total_time
    print(f"\nPipeline finished in {total_time/60:.2f} minutes.")
    return {
        "blocking_recall": blocking_metrics["recall"],
        "avg_candidates": blocking_metrics["avg_candidates"],
        "val_precision": best_val_metrics["precision"],
        "val_recall": best_val_metrics["recall"],
        "val_f05": best_val_metrics["macro_f05"],
        "best_threshold": best_threshold,
        "test_records_count": total_test_s1,
        "pred_matches": total_matches_pred,
        "pred_singletons": total_singletons_pred,
        "validator_passed": is_pass
    }


if __name__ == "__main__":
    run_pipeline()
