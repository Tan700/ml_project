"""
features.py
-----------
Feature extraction for candidate pairs in Entity Resolution.
Computes string similarity, token overlap, edit distances, digit matches,
and length differences using RapidFuzz and native Python routines.
Optimized with short-circuit paths for high-throughput batch inference.
"""

import re
from typing import List, Dict, Any, Optional
from data_loader import EntityRecord

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

RE_DIGITS = re.compile(r"\b\d+\b")
RE_PIN = re.compile(r"\b\d{5,6}\b")

FEATURE_NAMES = [
    "name_exact",
    "name_norm_exact",
    "name_fuzz_ratio",
    "name_fuzz_wratio",
    "name_token_set_ratio",
    "name_jaccard_tokens",
    "name_len_diff",
    "name_len_ratio",
    "addr_exact",
    "addr_norm_exact",
    "addr_fuzz_ratio",
    "addr_token_set_ratio",
    "addr_jaccard_tokens",
    "addr_len_diff",
    "country_equal",
    "digit_overlap_ratio",
    "postal_code_overlap",
    "target_source_is_s2",
]


def jaccard_similarity(tokens1: set, tokens2: set) -> float:
    """Computes Jaccard similarity between two token sets."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return float(intersection) / float(union) if union > 0 else 0.0


def extract_pair_features(s1: EntityRecord, target: EntityRecord) -> List[float]:
    """
    Computes a vector of similarity features between a Source 1 record
    and a candidate target record (Source 2 or Source 3).
    """
    # Name features
    name_exact = 1.0 if s1.business_name == target.business_name and s1.business_name else 0.0
    name_norm_exact = 1.0 if s1.norm_name == target.norm_name and s1.norm_name else 0.0

    s1_name_len = len(s1.norm_name)
    tgt_name_len = len(target.norm_name)
    name_len_diff = float(abs(s1_name_len - tgt_name_len))
    name_len_ratio = float(min(s1_name_len, tgt_name_len)) / float(max(s1_name_len, tgt_name_len, 1))

    if name_norm_exact:
        name_ratio = 1.0
        name_wratio = 1.0
        name_token_set = 1.0
        name_jaccard = 1.0
    else:
        s1_name_toks = set(s1.norm_name.split())
        tgt_name_toks = set(target.norm_name.split())
        name_jaccard = jaccard_similarity(s1_name_toks, tgt_name_toks)

        if HAS_RAPIDFUZZ and (name_jaccard > 0.1 or name_len_diff <= 5):
            name_ratio = fuzz.ratio(s1.norm_name, target.norm_name) / 100.0
            name_wratio = fuzz.WRatio(s1.norm_name, target.norm_name) / 100.0
            name_token_set = fuzz.token_set_ratio(s1.norm_name, target.norm_name) / 100.0
        else:
            name_ratio = name_jaccard
            name_wratio = name_jaccard
            name_token_set = name_jaccard

    # Address features
    addr_exact = 1.0 if s1.business_address == target.business_address and s1.business_address else 0.0
    addr_norm_exact = 1.0 if s1.norm_address == target.norm_address and s1.norm_address else 0.0

    s1_addr_len = len(s1.norm_address)
    tgt_addr_len = len(target.norm_address)
    addr_len_diff = float(abs(s1_addr_len - tgt_addr_len))

    if addr_norm_exact and s1.norm_address:
        addr_ratio = 1.0
        addr_token_set = 1.0
        addr_jaccard = 1.0
    else:
        s1_addr_toks = set(s1.norm_address.split())
        tgt_addr_toks = set(target.norm_address.split())
        addr_jaccard = jaccard_similarity(s1_addr_toks, tgt_addr_toks)

        if HAS_RAPIDFUZZ and s1.norm_address and target.norm_address and addr_jaccard > 0.1:
            addr_ratio = fuzz.ratio(s1.norm_address, target.norm_address) / 100.0
            addr_token_set = fuzz.token_set_ratio(s1.norm_address, target.norm_address) / 100.0
        else:
            addr_ratio = addr_jaccard
            addr_token_set = addr_jaccard

    # Country feature
    country_equal = 1.0 if s1.country == target.country else 0.0

    # Digits and postal code overlap
    digit_overlap = 0.0
    postal_code_overlap = 0.0
    if s1.business_address and target.business_address:
        s1_digits = set(RE_DIGITS.findall(s1.business_address))
        tgt_digits = set(RE_DIGITS.findall(target.business_address))
        digit_overlap = jaccard_similarity(s1_digits, tgt_digits)

        s1_pins = set(RE_PIN.findall(s1.business_address))
        tgt_pins = set(RE_PIN.findall(target.business_address))
        if s1_pins and tgt_pins and bool(s1_pins.intersection(tgt_pins)):
            postal_code_overlap = 1.0

    target_source_is_s2 = 1.0 if target.entity_id.startswith("S2-") else 0.0

    return [
        name_exact,
        name_norm_exact,
        name_ratio,
        name_wratio,
        name_token_set,
        name_jaccard,
        name_len_diff,
        name_len_ratio,
        addr_exact,
        addr_norm_exact,
        addr_ratio,
        addr_token_set,
        addr_jaccard,
        addr_len_diff,
        country_equal,
        digit_overlap,
        postal_code_overlap,
        target_source_is_s2,
    ]
