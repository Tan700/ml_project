"""
submission.py
-------------
Generates official submission files:
1. output/matching_results.tsv (final entity matches)
2. output/candidate_pairs.tsv (candidate set from blocking)
Validates compliance with hackathon formatting constraints.
"""

import os
from typing import Dict, List, Set


def write_matching_results(output_path: str,
                           s1_ids: List[str],
                           matches: Dict[str, Set[str]]):
    """
    Writes output/matching_results.tsv
    Header: source1_entity_id \t matched_entity_ids
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids:
            matched_list = sorted(list(matches.get(s1_id, set())))
            matched_str = ",".join(matched_list)
            f.write(f"{s1_id}\t{matched_str}\n")


def write_candidate_pairs(output_path: str,
                          s1_ids: List[str],
                          candidates: Dict[str, List[str]]):
    """
    Writes output/candidate_pairs.tsv
    Header: source1_entity_id \t candidate_entity_ids
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids:
            cand_list = sorted(list(dict.fromkeys(candidates.get(s1_id, []))))
            cand_str = ",".join(cand_list)
            f.write(f"{s1_id}\t{cand_str}\n")
