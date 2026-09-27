"""
predict.py
----------
Inference pipeline for candidate generation, feature computation,
model scoring, and thresholding.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
from data_loader import EntityRecord
from blocking import BlockingEngine
from features import extract_pair_features
from model import EntityPairClassifier


def run_inference(query_records: Dict[str, EntityRecord],
                  target_records: Dict[str, EntityRecord],
                  blocking_engine: BlockingEngine,
                  strategies: Set[str],
                  classifier: EntityPairClassifier,
                  threshold: float) -> Tuple[Dict[str, List[str]], Dict[str, Set[str]]]:
    """
    Executes end-to-end inference over query records:
    1. Generates candidates via blocking engine
    2. Feeds candidate pairs to feature extractor
    3. Scores pairs with classifier
    4. Applies threshold to select matches
    Returns (all_candidates, all_matches).
    """
    candidates_dict = {}
    matches_dict = {}

    batch_s1_records = list(query_records.values())
    total_queries = len(batch_s1_records)
    print(f"Running inference over {total_queries:,} query entities...")

    # Process in chunks to maintain low memory profile
    chunk_size = 50000
    for chunk_start in range(0, total_queries, chunk_size):
        chunk = batch_s1_records[chunk_start:chunk_start + chunk_size]
        pairs_to_score = []
        pair_identities = []

        for s1 in chunk:
            cands = blocking_engine.retrieve_candidates(s1, strategies)
            candidates_dict[s1.entity_id] = cands

            for tid in cands:
                target = target_records.get(tid)
                if target:
                    pairs_to_score.append(extract_pair_features(s1, target))
                    pair_identities.append((s1.entity_id, tid))

        # Score pairs if any candidates generated
        chunk_matches = {s1.entity_id: set() for s1 in chunk}
        if pairs_to_score:
            X_chunk = np.array(pairs_to_score, dtype=np.float32)
            probs = classifier.predict_proba(X_chunk)

            for (s1_id, tid), prob in zip(pair_identities, probs):
                if prob >= threshold:
                    chunk_matches[s1_id].add(tid)

        matches_dict.update(chunk_matches)
        scored_so_far = min(chunk_start + chunk_size, total_queries)
        if scored_so_far % 100000 == 0 or scored_so_far == total_queries:
            print(f"  Processed {scored_so_far:,} / {total_queries:,} queries...")

    return candidates_dict, matches_dict
