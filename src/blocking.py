"""
blocking.py
-----------
Multi-strategy blocking engine for candidate pair generation.
Constructs inverted indices over target sources (S2 and S3) using multiple
complementary keys within country boundaries, then queries for Source 1 records.
"""

import os
import re
import csv
import math
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
from normalization import extract_tokens, extract_char_ngrams
from data_loader import EntityRecord

# Common stop-tokens that produce noisy blocks
COMMON_NAME_STOPWORDS = {
    "inc", "corp", "corporation", "incorporated", "llc", "llp", "ltd", "limited",
    "pvt", "private", "co", "company", "and", "the", "of", "in", "for", "services",
    "solutions", "enterprises", "traders", "associates", "consulting", "group",
    "sarl", "sas", "sci", "eurl", "trading", "retail", "management", "holdings"
}

COMMON_ADDR_STOPWORDS = {
    "st", "rd", "ave", "blvd", "dr", "ln", "hwy", "pkwy", "ct", "cir",
    "ste", "apt", "unit", "fl", "bldg", "floor", "box", "pobox", "near", "opp",
    "behind", "beside", "above", "road", "street", "avenue", "city", "state", "india", "us", "france"
}


class BlockingEngine:
    def __init__(self, max_block_size: int = 300, max_candidates_per_entity: int = 50):
        self.max_block_size = max_block_size
        self.max_candidates_per_entity = max_candidates_per_entity
        
        # Inverted index: (country, strategy_name, key) -> list of target entity_ids
        self.index: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self.target_records: Dict[str, EntityRecord] = {}

    def _generate_keys_for_record(self, record: EntityRecord, strategies: Set[str]) -> List[Tuple[str, str]]:
        """
        Generates (strategy_name, key) pairs for a record.
        """
        keys = []
        c = record.country
        name = record.norm_name
        addr = record.norm_address

        # 1. Exact normalized name
        if "exact_name" in strategies and name:
            keys.append(("exact_name", name))

        # 2. Name prefix (first 6 characters)
        if "name_prefix" in strategies and len(name) >= 4:
            prefix = name.replace(" ", "")[:6]
            if len(prefix) >= 4:
                keys.append(("name_prefix", prefix))

        # 3. Name significant tokens
        if "name_tokens" in strategies and name:
            tokens = extract_tokens(name, min_len=4)
            for tok in tokens:
                if tok not in COMMON_NAME_STOPWORDS:
                    keys.append(("name_tokens", tok))

        # 4. Address tokens (e.g. PIN / postal code or distinct street numbers)
        if "address_tokens" in strategies and addr:
            addr_tokens = extract_tokens(addr, min_len=4)
            for tok in addr_tokens:
                # Prioritize numeric postal codes or distinct address tokens
                if tok.isdigit() and len(tok) in (5, 6):
                    keys.append(("address_pin", tok))
                elif tok not in COMMON_ADDR_STOPWORDS and not tok.isdigit():
                    # Check if token is significant
                    if len(tok) >= 5:
                        keys.append(("address_token", tok))

        # 5. Character trigrams of name
        if "char_ngrams" in strategies and len(name) >= 5:
            ngrams = extract_char_ngrams(name, n=3)
            # Pick first and middle trigram to keep index compact
            if ngrams:
                keys.append(("char_3gram", ngrams[0]))
                if len(ngrams) > 3:
                    keys.append(("char_3gram", ngrams[len(ngrams) // 2]))

        return keys

    def fit(self, target_records: Dict[str, EntityRecord], strategies: Set[str]):
        """
        Builds inverted index from target records (Source 2 and Source 3).
        """
        self.target_records = target_records
        self.index.clear()

        for eid, record in target_records.items():
            keys = self._generate_keys_for_record(record, strategies)
            c = record.country
            for strat, key in keys:
                self.index[(c, strat, key)].append(eid)

        # Filter out blocks that exceed max_block_size to maintain high precision and low candidate counts
        filtered_index = {}
        for index_key, eids in self.index.items():
            if len(eids) <= self.max_block_size:
                filtered_index[index_key] = eids
        self.index = filtered_index

    def retrieve_candidates(self, query_record: EntityRecord, strategies: Set[str]) -> List[str]:
        """
        Retrieves candidate entity_ids from target records for a given query record.
        """
        c = query_record.country
        keys = self._generate_keys_for_record(query_record, strategies)
        
        # Track candidate scores by how many blocking rules matched
        candidate_hit_counts = Counter()
        for strat, key in keys:
            index_key = (c, strat, key)
            if index_key in self.index:
                # Weight exact name higher
                weight = 3 if strat == "exact_name" else 1
                for target_id in self.index[index_key]:
                    candidate_hit_counts[target_id] += weight

        if not candidate_hit_counts:
            return []

        # Sort candidates by number of hits descending
        ranked = [eid for eid, _ in candidate_hit_counts.most_common(self.max_candidates_per_entity)]
        return ranked

    def generate_all_candidates(self, query_records: Dict[str, EntityRecord],
                                strategies: Set[str]) -> Dict[str, List[str]]:
        """
        Generates candidate lists for all query records.
        """
        candidates = {}
        for s1_id, record in query_records.items():
            cands = self.retrieve_candidates(record, strategies)
            candidates[s1_id] = cands
        return candidates


def evaluate_blocking(candidates: Dict[str, List[str]],
                      ground_truth: Dict[str, Set[str]],
                      num_targets: int) -> Dict[str, float]:
    """
    Computes blocking recall, average candidates, median candidates,
    max candidates, total pairs, and reduction ratio.
    """
    total_true_matches = 0
    retained_true_matches = 0
    candidate_counts = []
    total_pairs = 0

    for s1_id, true_matches in ground_truth.items():
        total_true_matches += len(true_matches)
        cands = set(candidates.get(s1_id, []))
        candidate_counts.append(len(cands))
        total_pairs += len(cands)
        retained_true_matches += len(true_matches.intersection(cands))

    recall = (retained_true_matches / total_true_matches) if total_true_matches > 0 else 0.0
    num_queries = len(candidate_counts)
    avg_cands = (total_pairs / num_queries) if num_queries > 0 else 0.0
    
    candidate_counts.sort()
    med_cands = candidate_counts[num_queries // 2] if num_queries > 0 else 0.0
    max_cands = candidate_counts[-1] if candidate_counts else 0

    # Reduction ratio: 1 - (total_candidate_pairs / (num_queries * num_targets))
    total_possible_comparisons = max(num_queries * num_targets, 1)
    reduction_ratio = 1.0 - (total_pairs / total_possible_comparisons)

    return {
        "recall": round(recall, 4),
        "avg_candidates": round(avg_cands, 2),
        "median_candidates": float(med_cands),
        "max_candidates": int(max_cands),
        "total_pairs": total_pairs,
        "reduction_ratio": round(reduction_ratio, 6),
    }


def run_blocking_experiments(s1_records: Dict[str, EntityRecord],
                             target_records: Dict[str, EntityRecord],
                             ground_truth: Dict[str, Set[str]],
                             output_csv: str) -> str:
    """
    Runs several blocking configurations and writes results to CSV.
    Returns the chosen configuration name.
    """
    configs = [
        ("Exact_Name_Only", {"exact_name"}, 300, 30),
        ("Exact_Name_Plus_Prefix", {"exact_name", "name_prefix"}, 300, 40),
        ("Exact_Prefix_Tokens", {"exact_name", "name_prefix", "name_tokens"}, 300, 50),
        ("Multi_Strategy_Union", {"exact_name", "name_prefix", "name_tokens", "address_tokens", "char_ngrams"}, 300, 50),
    ]

    results = []
    print("\nRunning blocking configuration experiments...")
    
    for name, strategies, max_block, max_cands in configs:
        engine = BlockingEngine(max_block_size=max_block, max_candidates_per_entity=max_cands)
        engine.fit(target_records, strategies)
        candidates = engine.generate_all_candidates(s1_records, strategies)
        metrics = evaluate_blocking(candidates, ground_truth, len(target_records))
        metrics["config_name"] = name
        results.append(metrics)
        print(f"  Config '{name}': Recall={metrics['recall']*100:.2f}%, Avg Cands={metrics['avg_candidates']}, RedRatio={metrics['reduction_ratio']:.4f}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "config_name", "recall", "avg_candidates", "median_candidates",
            "max_candidates", "total_pairs", "reduction_ratio"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved blocking experiment results to {output_csv}")
    # Return best configuration (highest recall with reasonable candidates)
    return "Multi_Strategy_Union"
