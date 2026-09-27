"""
evaluate.py
-----------
Official evaluation metrics for the Business Entity Resolution Challenge.
Computes macro-averaged F_0.5 per Source 1 entity, singleton accuracy,
global precision, global recall, and threshold optimization.
"""

import os
import csv
from typing import Dict, Set, List, Tuple, Optional


def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> Tuple[float, float, float]:
    """
    Computes (f05, precision, recall) for a single Source 1 entity according to hackathon rules.
    - If true_matches is empty (singleton):
      - 1.0 if pred_matches is empty
      - 0.0 if pred_matches is non-empty
    - If true_matches is non-empty:
      - 0.0 if pred_matches is empty
      - F_0.5 = (1.25 * P * R) / (0.25 * P + R) otherwise
    """
    if not true_matches:
        if not pred_matches:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 1.0  # False merge on singleton

    if not pred_matches:
        return 0.0, 0.0, 0.0

    tp = len(true_matches.intersection(pred_matches))
    prec = float(tp) / float(len(pred_matches))
    rec = float(tp) / float(len(true_matches))

    if prec == 0.0 or rec == 0.0:
        return 0.0, prec, rec

    f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
    return f05, prec, rec


def evaluate_predictions(ground_truth: Dict[str, Set[str]],
                         predictions: Dict[str, Set[str]]) -> Dict[str, float]:
    """
    Computes macro-averaged F_0.5 and supporting metrics over all Source 1 evaluation entities.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {"macro_f05": 0.0, "precision": 0.0, "recall": 0.0, "singleton_accuracy": 0.0}

    sum_f05 = 0.0
    total_tp = 0
    total_pred = 0
    total_true = 0
    true_singletons = 0
    correct_singletons = 0
    pred_singletons = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        f05, p, r = compute_entity_f05(true_set, pred_set)
        sum_f05 += f05

        tp = len(true_set.intersection(pred_set))
        total_tp += tp
        total_pred += len(pred_set)
        total_true += len(true_set)

        if not true_set:
            true_singletons += 1
            if not pred_set:
                correct_singletons += 1

        if not pred_set:
            pred_singletons += 1

    macro_f05 = sum_f05 / total_entities
    global_prec = (total_tp / total_pred) if total_pred > 0 else (1.0 if total_true == 0 else 0.0)
    global_rec = (total_tp / total_true) if total_true > 0 else 1.0
    singleton_acc = (correct_singletons / true_singletons) if true_singletons > 0 else 1.0

    return {
        "macro_f05": round(macro_f05, 4),
        "precision": round(global_prec, 4),
        "recall": round(global_rec, 4),
        "singleton_accuracy": round(singleton_acc, 4),
        "total_entities": total_entities,
        "total_true_matches": total_true,
        "total_pred_matches": total_pred,
        "true_singletons": true_singletons,
        "pred_singletons": pred_singletons,
    }


def optimize_threshold(candidate_pairs: List[Tuple[str, str, float]],
                       ground_truth: Dict[str, Set[str]],
                       thresholds: List[float] = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                       output_csv: Optional[str] = None) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates predictions across multiple probability thresholds and saves to CSV.
    Returns (best_threshold, best_metrics).
    """
    # Group pairs by s1_id: s1_id -> list of (target_id, prob)
    s1_candidates = {s1_id: [] for s1_id in ground_truth}
    for s1_id, target_id, prob in candidate_pairs:
        if s1_id in s1_candidates:
            s1_candidates[s1_id].append((target_id, prob))

    results = []
    best_thresh = 0.75
    best_f05 = -1.0
    best_metrics = {}

    print("\nRunning threshold evaluation experiments...")
    for t in thresholds:
        preds = {}
        for s1_id, pairs in s1_candidates.items():
            matched = {tid for tid, prob in pairs if prob >= t}
            preds[s1_id] = matched

        metrics = evaluate_predictions(ground_truth, preds)
        metrics["threshold"] = t
        results.append(metrics)
        print(f"  Threshold {t:.2f} -> Macro F_0.5: {metrics['macro_f05']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, Pred Singletons: {metrics['pred_singletons']}")

        if metrics["macro_f05"] > best_f05:
            best_f05 = metrics["macro_f05"]
            best_thresh = t
            best_metrics = metrics

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "threshold", "macro_f05", "precision", "recall",
                "singleton_accuracy", "total_entities", "total_pred_matches",
                "pred_singletons"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved threshold results to {output_csv}")

    print(f"\nOptimal Threshold: {best_thresh} with Macro F_0.5 = {best_f05:.4f}")
    return best_thresh, best_metrics
