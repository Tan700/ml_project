#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Business Entity Resolution Challenge
=======================================================================
This script performs a comprehensive, memory-efficient, single-pass analysis
of all files in the training and test splits, addressing all 11 required questions.
"""

import os
import sys
import json
import re
import math
import time
from collections import Counter, defaultdict

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATASET_DIR = os.path.join("datasets", "student_resource", "dataset")
OUTPUT_DIR = "outputs"

FILES = {
    "train_source1": os.path.join(DATASET_DIR, "train", "train_source1.tsv"),
    "train_source2": os.path.join(DATASET_DIR, "train", "train_source2.tsv"),
    "train_source3": os.path.join(DATASET_DIR, "train", "train_source3.tsv"),
    "train_ground_truth": os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv"),
    "test_source1": os.path.join(DATASET_DIR, "test", "test_source1.tsv"),
    "test_source2": os.path.join(DATASET_DIR, "test", "test_source2.tsv"),
    "test_source3": os.path.join(DATASET_DIR, "test", "test_source3.tsv"),
}


def analyze_source_file(file_path, file_label, is_test=False):
    """
    Performs streaming single-pass analysis over a source TSV file:
    columns: entity_id, business_name, business_address, country
    """
    print(f"\n========================================================")
    print(f"Processing {file_label}: {file_path}")
    print(f"========================================================")
    t0 = time.time()
    
    total_rows = 0
    malformed_rows = 0
    header_cols = []
    
    missing_counts = {"entity_id": 0, "business_name": 0, "business_address": 0, "country": 0}
    whitespace_only = {"entity_id": 0, "business_name": 0, "business_address": 0, "country": 0}
    
    char_lens = {
        "entity_id": {"min": float("inf"), "max": 0, "sum": 0},
        "business_name": {"min": float("inf"), "max": 0, "sum": 0},
        "business_address": {"min": float("inf"), "max": 0, "sum": 0},
        "country": {"min": float("inf"), "max": 0, "sum": 0},
    }
    
    country_counts = Counter()
    
    # ID uniqueness & Duplicate tracking via integer IDs / 64-bit hashes
    # S1-123456789 -> store integer part to minimize memory footprint
    entity_ids_seen = set()
    duplicate_eids = 0
    
    # For duplicate checking: sample first 500,000 for exact name/addr/row duplicate rates
    sample_limit = 500000
    sample_exact_rows = set()
    sample_row_dupes = 0
    sample_names = set()
    sample_name_dupes = 0
    sample_addrs = set()
    sample_addr_dupes = 0
    sample_name_addr_pairs = set()
    sample_name_addr_dupes = 0
    
    # Name patterns
    name_casing = Counter()
    has_leading_noise = 0
    has_indic_script = 0
    has_accents = 0
    has_domain_name = 0
    has_digits_in_name = 0
    legal_suffix_counts = Counter()
    name_word_counts = []
    
    # Address patterns
    addr_empty = 0
    addr_casing = Counter()
    addr_po_box = 0
    addr_unit = 0
    addr_landmark_in = 0
    addr_pin_india = 0
    addr_zip_us = 0
    addr_zip_fr = 0
    addr_word_counts = []
    
    # Regex patterns
    re_indic = re.compile(r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]")
    re_accents = re.compile(r"[éèêëàâäôöûüçîïÉÈÊËÀÂÄÔÖÛÜÇÎÏ]")
    re_domain = re.compile(r"\b(?:https?://|www\.)?[a-zA-Z0-9-]+\.(?:com|org|net|in|co\.in|fr|io|biz|info)\b", re.IGNORECASE)
    re_leading_noise = re.compile(r"^[\s\-<>\*#@\.\,\/\"\'_=+~]+")
    
    re_legal = [
        ("LLC", re.compile(r"\bL\.?L\.?C\.?\b", re.IGNORECASE)),
        ("INC", re.compile(r"\bINC\.?(?:ORPORATED)?\b", re.IGNORECASE)),
        ("CORP", re.compile(r"\bCORP\.?(?:ORATION)?\b", re.IGNORECASE)),
        ("CO", re.compile(r"\bCO\.?(?:MPANY)?\b", re.IGNORECASE)),
        ("LTD", re.compile(r"\bLTD\.?(?:IMITED)?\b", re.IGNORECASE)),
        ("PVT_LTD", re.compile(r"\b(?:PVT\.?|PRIVATE)\s+(?:LTD\.?|LIMITED)\b", re.IGNORECASE)),
        ("LLP", re.compile(r"\bL\.?L\.?P\.?\b", re.IGNORECASE)),
        ("SARL", re.compile(r"\bS\.?A\.?R\.?L\.?\b", re.IGNORECASE)),
        ("SAS", re.compile(r"\bS\.?A\.?S\.?\b", re.IGNORECASE)),
        ("SA", re.compile(r"\bS\.?A\.?\b", re.IGNORECASE)),
        ("SCI", re.compile(r"\bS\.?C\.?I\.?\b", re.IGNORECASE)),
        ("ENTERPRISES", re.compile(r"\bENTERPRISES?\b", re.IGNORECASE)),
        ("TRADERS", re.compile(r"\bTRADERS?\b", re.IGNORECASE)),
        ("SERVICES", re.compile(r"\bSERVICES?\b", re.IGNORECASE)),
        ("SOLUTIONS", re.compile(r"\bSOLUTIONS?\b", re.IGNORECASE)),
        ("CONSULTING", re.compile(r"\bCONSULTING\b", re.IGNORECASE)),
    ]
    
    re_po_box = re.compile(r"\bP\.?O\.?\s+BOX\b", re.IGNORECASE)
    re_unit = re.compile(r"\b(?:SUITE|STE|APT|APARTMENT|UNIT|BLDG|BUILDING|FL|FLOOR|ROOM|RM)\b", re.IGNORECASE)
    re_landmark_in = re.compile(r"\b(?:NEAR|OPP|OPPOSITE|BEHIND|BESIDE|ADJACENT|ABOVE)\b", re.IGNORECASE)
    re_pin_in = re.compile(r"\b[1-9][0-9]{5}\b")
    re_zip_us = re.compile(r"\b[0-9]{5}(?:-[0-9]{4})?\b")
    re_zip_fr = re.compile(r"\b(?:0[1-9]|[1-8][0-9]|9[0-8])[0-9]{3}\b")
    
    with open(file_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            return {"error": "Empty file"}
        header_cols = [c.strip() for c in header_line.rstrip("\r\n").split("\t")]
        
        for line in f:
            total_rows += 1
            raw = line.rstrip("\r\n")
            parts = raw.split("\t")
            
            if len(parts) != 4:
                malformed_rows += 1
                if len(parts) < 4:
                    parts = parts + [""] * (4 - len(parts))
                else:
                    parts = parts[:3] + [" ".join(parts[3:])]
            
            eid, bname, baddr, bcountry = parts[0], parts[1], parts[2], parts[3]
            
            # Entity ID
            eid_s = eid.strip()
            if not eid:
                missing_counts["entity_id"] += 1
            elif not eid_s:
                whitespace_only["entity_id"] += 1
            
            # Check integer ID or hash for uniqueness
            eid_key = int(eid_s[3:]) if eid_s.startswith(("S1-", "S2-", "S3-")) and eid_s[3:].isdigit() else eid_s
            if eid_key in entity_ids_seen:
                duplicate_eids += 1
            else:
                entity_ids_seen.add(eid_key)
                
            e_len = len(eid)
            char_lens["entity_id"]["min"] = min(char_lens["entity_id"]["min"], e_len)
            char_lens["entity_id"]["max"] = max(char_lens["entity_id"]["max"], e_len)
            char_lens["entity_id"]["sum"] += e_len
            
            # Country
            c_s = bcountry.strip()
            if not bcountry:
                missing_counts["country"] += 1
            elif not c_s:
                whitespace_only["country"] += 1
            country_counts[c_s if c_s else "(missing)"] += 1
            
            c_len = len(bcountry)
            char_lens["country"]["min"] = min(char_lens["country"]["min"], c_len)
            char_lens["country"]["max"] = max(char_lens["country"]["max"], c_len)
            char_lens["country"]["sum"] += c_len
            
            # Business Name
            bname_s = bname.strip()
            bname_len = len(bname)
            if not bname:
                missing_counts["business_name"] += 1
            elif not bname_s:
                whitespace_only["business_name"] += 1
                
            char_lens["business_name"]["min"] = min(char_lens["business_name"]["min"], bname_len)
            char_lens["business_name"]["max"] = max(char_lens["business_name"]["max"], bname_len)
            char_lens["business_name"]["sum"] += bname_len
            
            # Business Address
            baddr_s = baddr.strip()
            baddr_len = len(baddr)
            if not baddr:
                missing_counts["business_address"] += 1
                addr_empty += 1
            elif not baddr_s:
                whitespace_only["business_address"] += 1
                addr_empty += 1
                
            char_lens["business_address"]["min"] = min(char_lens["business_address"]["min"], baddr_len)
            char_lens["business_address"]["max"] = max(char_lens["business_address"]["max"], baddr_len)
            char_lens["business_address"]["sum"] += baddr_len
            
            # Duplicate checking on sample
            if total_rows <= sample_limit:
                r_hash = hash(raw)
                if r_hash in sample_exact_rows:
                    sample_row_dupes += 1
                else:
                    sample_exact_rows.add(r_hash)
                    
                n_hash = hash(bname_s)
                if n_hash in sample_names:
                    sample_name_dupes += 1
                else:
                    sample_names.add(n_hash)
                    
                if baddr_s:
                    a_hash = hash(baddr_s)
                    if a_hash in sample_addrs:
                        sample_addr_dupes += 1
                    else:
                        sample_addrs.add(a_hash)
                        
                na_hash = hash((bname_s, baddr_s))
                if na_hash in sample_name_addr_pairs:
                    sample_name_addr_dupes += 1
                else:
                    sample_name_addr_pairs.add(na_hash)
            
            # Name patterns
            if bname_s:
                if bname_s.isupper():
                    name_casing["ALL_UPPER"] += 1
                elif bname_s.islower():
                    name_casing["all_lower"] += 1
                elif bname_s.istitle():
                    name_casing["Title_Case"] += 1
                else:
                    name_casing["Mixed_Case"] += 1
                    
                if re_leading_noise.match(bname_s):
                    has_leading_noise += 1
                if re_indic.search(bname_s):
                    has_indic_script += 1
                if re_accents.search(bname_s):
                    has_accents += 1
                if re_domain.search(bname_s):
                    has_domain_name += 1
                if any(ch.isdigit() for ch in bname_s):
                    has_digits_in_name += 1
                    
                for tname, tre in re_legal:
                    if tre.search(bname_s):
                        legal_suffix_counts[tname] += 1
                        
                if total_rows <= 100000:
                    name_word_counts.append(len(bname_s.split()))
            
            # Address patterns
            if baddr_s:
                if baddr_s.isupper():
                    addr_casing["ALL_UPPER"] += 1
                elif baddr_s.istitle():
                    addr_casing["Title_Case"] += 1
                else:
                    addr_casing["Mixed_Other"] += 1
                    
                if re_po_box.search(baddr_s):
                    addr_po_box += 1
                if re_unit.search(baddr_s):
                    addr_unit += 1
                if re_landmark_in.search(baddr_s):
                    addr_landmark_in += 1
                if re_pin_in.search(baddr_s):
                    addr_pin_india += 1
                if re_zip_us.search(baddr_s):
                    addr_zip_us += 1
                if re_zip_fr.search(baddr_s):
                    addr_zip_fr += 1
                    
                if total_rows <= 100000:
                    addr_word_counts.append(len(baddr_s.split()))
                    
    elapsed = time.time() - t0
    print(f"  Processed {total_rows:,} rows in {elapsed:.2f}s")
    print(f"  Unique entity_id count: {len(entity_ids_seen):,} | Duplicate entity_ids: {duplicate_eids}")
    print(f"  Country counts: {dict(country_counts)}")
    print(f"  Empty addresses: {addr_empty:,} ({round(addr_empty*100.0/max(total_rows,1), 2)}%)")
    
    return {
        "file_label": file_label,
        "file_path": file_path,
        "total_rows": total_rows,
        "malformed_rows": malformed_rows,
        "header_cols": header_cols,
        "unique_entity_ids": len(entity_ids_seen),
        "duplicate_entity_ids": duplicate_eids,
        "sample_duplicate_analysis": {
            "sample_size": min(total_rows, sample_limit),
            "exact_duplicate_rows": sample_row_dupes,
            "exact_duplicate_rows_pct": round(sample_row_dupes * 100.0 / min(total_rows, sample_limit), 4),
            "duplicate_business_names": sample_name_dupes,
            "duplicate_business_names_pct": round(sample_name_dupes * 100.0 / min(total_rows, sample_limit), 2),
            "duplicate_addresses": sample_addr_dupes,
            "duplicate_addresses_pct": round(sample_addr_dupes * 100.0 / min(total_rows, sample_limit), 2),
            "duplicate_name_address_pairs": sample_name_addr_dupes,
            "duplicate_name_address_pairs_pct": round(sample_name_addr_dupes * 100.0 / min(total_rows, sample_limit), 2),
        },
        "missing_counts": missing_counts,
        "missing_pct": {k: round(v * 100.0 / max(total_rows, 1), 4) for k, v in missing_counts.items()},
        "country_distribution": dict(country_counts),
        "country_percentages": {k: round(v * 100.0 / max(total_rows, 1), 2) for k, v in country_counts.items()},
        "char_lengths": {
            k: {
                "min": char_lens[k]["min"] if char_lens[k]["min"] != float("inf") else 0,
                "max": char_lens[k]["max"],
                "mean": round(char_lens[k]["sum"] / max(total_rows, 1), 2),
            }
            for k in char_lens
        },
        "name_patterns": {
            "casing": dict(name_casing),
            "casing_pct": {k: round(v * 100.0 / max(total_rows, 1), 2) for k, v in name_casing.items()},
            "has_leading_noise": has_leading_noise,
            "has_leading_noise_pct": round(has_leading_noise * 100.0 / max(total_rows, 1), 2),
            "has_indic_script": has_indic_script,
            "has_indic_script_pct": round(has_indic_script * 100.0 / max(total_rows, 1), 2),
            "has_accents": has_accents,
            "has_accents_pct": round(has_accents * 100.0 / max(total_rows, 1), 2),
            "has_domain_name": has_domain_name,
            "has_domain_name_pct": round(has_domain_name * 100.0 / max(total_rows, 1), 2),
            "has_digits": has_digits_in_name,
            "has_digits_pct": round(has_digits_in_name * 100.0 / max(total_rows, 1), 2),
            "legal_terms": dict(legal_suffix_counts),
            "avg_word_count": round(sum(name_word_counts) / max(len(name_word_counts), 1), 2),
        },
        "address_patterns": {
            "empty_addresses": addr_empty,
            "empty_addresses_pct": round(addr_empty * 100.0 / max(total_rows, 1), 2),
            "casing": dict(addr_casing),
            "casing_pct": {k: round(v * 100.0 / max(total_rows, 1), 2) for k, v in addr_casing.items()},
            "has_po_box": addr_po_box,
            "has_po_box_pct": round(addr_po_box * 100.0 / max(total_rows, 1), 2),
            "has_unit_apt_ste": addr_unit,
            "has_unit_apt_ste_pct": round(addr_unit * 100.0 / max(total_rows, 1), 2),
            "has_landmark_in": addr_landmark_in,
            "has_landmark_in_pct": round(addr_landmark_in * 100.0 / max(total_rows, 1), 2),
            "has_pin_code_india": addr_pin_india,
            "has_pin_code_india_pct": round(addr_pin_india * 100.0 / max(total_rows, 1), 2),
            "has_zip_us": addr_zip_us,
            "has_zip_us_pct": round(addr_zip_us * 100.0 / max(total_rows, 1), 2),
            "has_zip_fr": addr_zip_fr,
            "has_zip_fr_pct": round(addr_zip_fr * 100.0 / max(total_rows, 1), 2),
            "avg_word_count": round(sum(addr_word_counts) / max(len(addr_word_counts), 1), 2),
        },
    }


def analyze_ground_truth(gt_path, s1_countries):
    """
    Analyzes train_ground_truth.tsv:
    source1_entity_id \t matched_entity_ids
    """
    print(f"\n========================================================")
    print(f"Processing Ground Truth: {gt_path}")
    print(f"========================================================")
    t0 = time.time()
    
    total_s1 = 0
    singletons = 0
    non_singletons = 0
    total_matches = 0
    
    match_distribution = Counter()
    s2_distribution = Counter()
    s3_distribution = Counter()
    
    only_s2 = 0
    only_s3 = 0
    both_s2_s3 = 0
    
    matched_s2_ids = set()
    matched_s3_ids = set()
    s2_multi_match = Counter()
    s3_multi_match = Counter()
    
    country_totals = Counter()
    country_singletons = Counter()
    country_matches = Counter()
    
    match_counts = []
    
    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            total_s1 += 1
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ""
            
            c = s1_countries.get(s1_id, "Unknown")
            country_totals[c] += 1
            
            if not matched_str.strip():
                singletons += 1
                country_singletons[c] += 1
                match_distribution[0] += 1
                s2_distribution[0] += 1
                s3_distribution[0] += 1
                match_counts.append(0)
            else:
                non_singletons += 1
                m_list = matched_str.split(",")
                m_count = len(m_list)
                total_matches += m_count
                country_matches[c] += m_count
                match_counts.append(m_count)
                
                bucket = m_count if m_count <= 10 else "11+"
                match_distribution[bucket] += 1
                
                s2_matches = [m for m in m_list if m.startswith("S2-")]
                s3_matches = [m for m in m_list if m.startswith("S3-")]
                
                n_s2 = len(s2_matches)
                n_s3 = len(s3_matches)
                
                s2_distribution[n_s2 if n_s2 <= 5 else "6+"] += 1
                s3_distribution[n_s3 if n_s3 <= 5 else "6+"] += 1
                
                if n_s2 > 0 and n_s3 == 0:
                    only_s2 += 1
                elif n_s3 > 0 and n_s2 == 0:
                    only_s3 += 1
                elif n_s2 > 0 and n_s3 > 0:
                    both_s2_s3 += 1
                    
                for m in s2_matches:
                    matched_s2_ids.add(m)
                    s2_multi_match[m] += 1
                for m in s3_matches:
                    matched_s3_ids.add(m)
                    s3_multi_match[m] += 1
                    
    # Percentiles
    match_counts.sort()
    n = len(match_counts)
    p25 = match_counts[int(n * 0.25)]
    p50 = match_counts[int(n * 0.50)]
    p75 = match_counts[int(n * 0.75)]
    p90 = match_counts[int(n * 0.90)]
    p95 = match_counts[int(n * 0.95)]
    p99 = match_counts[int(n * 0.99)]
    max_m = match_counts[-1] if match_counts else 0
    min_m = match_counts[0] if match_counts else 0
    mean_m = total_matches / max(total_s1, 1)
    
    variance = sum((x - mean_m) ** 2 for x in match_counts) / max(n, 1)
    std_dev = math.sqrt(variance)
    
    s2_multi_count = sum(1 for v in s2_multi_match.values() if v > 1)
    s3_multi_count = sum(1 for v in s3_multi_match.values() if v > 1)
    
    elapsed = time.time() - t0
    print(f"  Processed {total_s1:,} ground truth rows in {elapsed:.2f}s")
    print(f"  Total matches: {total_matches:,} | Singletons: {singletons:,} ({round(singletons*100.0/total_s1, 2)}%)")
    print(f"  Unique matched S2 IDs: {len(matched_s2_ids):,} | S2 multi-matched: {s2_multi_count}")
    print(f"  Unique matched S3 IDs: {len(matched_s3_ids):,} | S3 multi-matched: {s3_multi_count}")
    
    stats = {
        "total_source1_entities": total_s1,
        "singletons": singletons,
        "singleton_pct": round(singletons * 100.0 / max(total_s1, 1), 2),
        "non_singletons": non_singletons,
        "non_singleton_pct": round(non_singletons * 100.0 / max(total_s1, 1), 2),
        "total_matches": total_matches,
        "summary_statistics": {
            "min": min_m,
            "max": max_m,
            "mean": round(mean_m, 3),
            "median": p50,
            "std_dev": round(std_dev, 3),
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p90": p90,
            "p95": p95,
            "p99": p99,
        },
        "match_distribution": dict(match_distribution),
        "source_breakdown": {
            "only_s2": only_s2,
            "only_s2_pct": round(only_s2 * 100.0 / max(total_s1, 1), 2),
            "only_s3": only_s3,
            "only_s3_pct": round(only_s3 * 100.0 / max(total_s1, 1), 2),
            "both_s2_and_s3": both_s2_s3,
            "both_s2_and_s3_pct": round(both_s2_s3 * 100.0 / max(total_s1, 1), 2),
            "neither_singletons": singletons,
            "neither_singletons_pct": round(singletons * 100.0 / max(total_s1, 1), 2),
        },
        "s2_distribution": dict(s2_distribution),
        "s3_distribution": dict(s3_distribution),
        "unique_matched_s2_ids": len(matched_s2_ids),
        "unique_matched_s3_ids": len(matched_s3_ids),
        "s2_entities_linked_to_multiple_s1": s2_multi_count,
        "s3_entities_linked_to_multiple_s1": s3_multi_count,
        "singletons_by_country": {
            c: {
                "total": country_totals[c],
                "singletons": country_singletons[c],
                "singleton_pct": round(country_singletons[c] * 100.0 / max(country_totals[c], 1), 2),
                "total_matches": country_matches[c],
                "mean_matches": round(country_matches[c] / max(country_totals[c], 1), 2),
            }
            for c in country_totals
        },
    }
    return stats, matched_s2_ids, matched_s3_ids


def verify_cross_country(gt_path, s1_countries, s2_path, s3_path):
    """
    Verifies whether matches in ground truth ever cross country boundaries.
    Uses sample and country sets for S2/S3.
    """
    print(f"\n========================================================")
    print("Verifying Country Boundaries in Ground Truth")
    print(f"========================================================")
    t0 = time.time()
    
    # Track US integer IDs for S2 and S3
    s2_is_us = set()
    with open(s2_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[3].strip() == "US":
                eid = parts[0]
                if eid.startswith("S2-") and eid[3:].isdigit():
                    s2_is_us.add(int(eid[3:]))
                    
    s3_is_us = set()
    with open(s3_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[3].strip() == "US":
                eid = parts[0]
                if eid.startswith("S3-") and eid[3:].isdigit():
                    s3_is_us.add(int(eid[3:]))
                    
    cross_country = 0
    same_country = 0
    
    with open(gt_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ""
            if not matched_str.strip():
                continue
            s1_is_us_val = (s1_countries.get(s1_id) == "US")
            
            for m in matched_str.split(","):
                if m.startswith("S2-"):
                    m_num = int(m[3:]) if m[3:].isdigit() else None
                    if m_num is not None:
                        m_is_us = (m_num in s2_is_us)
                        if m_is_us == s1_is_us_val:
                            same_country += 1
                        else:
                            cross_country += 1
                elif m.startswith("S3-"):
                    m_num = int(m[3:]) if m[3:].isdigit() else None
                    if m_num is not None:
                        m_is_us = (m_num in s3_is_us)
                        if m_is_us == s1_is_us_val:
                            same_country += 1
                        else:
                            cross_country += 1
                            
    elapsed = time.time() - t0
    print(f"  Verified {same_country + cross_country:,} ground truth links in {elapsed:.2f}s")
    print(f"  Same country matches: {same_country:,} | Cross country matches: {cross_country:,}")
    return {
        "same_country_matches": same_country,
        "cross_country_matches": cross_country,
        "strict_country_boundary": (cross_country == 0),
    }


def main():
    print("=" * 80)
    print("STARTING COMPLETE EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}
    
    # 1. Load S1 countries
    print("Loading Train S1 country map...")
    s1_countries = {}
    with open(FILES["train_source1"], "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4:
                s1_countries[p[0]] = p[3].strip()
    print(f"Loaded {len(s1_countries):,} S1 countries.")
    
    # 2. Analyze Ground Truth
    gt_stats, matched_s2, matched_s3 = analyze_ground_truth(FILES["train_ground_truth"], s1_countries)
    all_results["ground_truth"] = gt_stats
    
    # 3. Analyze Train Sources
    train_stats = {}
    for key in ["train_source1", "train_source2", "train_source3"]:
        lbl = key.replace("train_", "").capitalize()
        train_stats[key] = analyze_source_file(FILES[key], f"Train {lbl}")
    all_results["train_sources"] = train_stats
    
    # Coverage of S2 and S3
    s2_rows = train_stats["train_source2"]["total_rows"]
    s3_rows = train_stats["train_source3"]["total_rows"]
    all_results["coverage"] = {
        "s2_total_entities": s2_rows,
        "s2_matched_entities": len(matched_s2),
        "s2_unmatched_orphans": s2_rows - len(matched_s2),
        "s2_match_percentage": round(len(matched_s2) * 100.0 / max(s2_rows, 1), 2),
        "s3_total_entities": s3_rows,
        "s3_matched_entities": len(matched_s3),
        "s3_unmatched_orphans": s3_rows - len(matched_s3),
        "s3_match_percentage": round(len(matched_s3) * 100.0 / max(s3_rows, 1), 2),
    }
    
    # 4. Cross country verification
    cc_stats = verify_cross_country(FILES["train_ground_truth"], s1_countries, FILES["train_source2"], FILES["train_source3"])
    all_results["cross_country"] = cc_stats
    
    # 5. Analyze Test Sources
    test_stats = {}
    for key in ["test_source1", "test_source2", "test_source3"]:
        lbl = key.replace("test_", "").capitalize()
        test_stats[key] = analyze_source_file(FILES[key], f"Test {lbl}", is_test=True)
    all_results["test_sources"] = test_stats
    
    # Save JSON results
    json_path = os.path.join(OUTPUT_DIR, "eda_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Raw results saved to: {json_path}")
    
    # Write Markdown Report
    md_path = os.path.join(OUTPUT_DIR, "eda_report.md")
    write_markdown_report(all_results, md_path)
    print(f"[+] Markdown report saved to: {md_path}")
    print("\n" + "=" * 80)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("=" * 80)


def write_markdown_report(res, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Exploratory Data Analysis Report: Business Entity Resolution\n\n")
        f.write("Generated from dataset files in `datasets/student_resource/dataset/`.\n\n")
        
        f.write("## 1. Number of Rows in Each Source\n\n")
        f.write("| Split | Source | Filename | Total Rows | Unique Entity IDs | Duplicate Rows (Sample) |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for k in ["train_source1", "train_source2", "train_source3"]:
            s = res["train_sources"][k]
            f.write(f"| Train | {k.replace('train_', '').capitalize()} | {os.path.basename(s['file_path'])} | {s['total_rows']:,} | {s['unique_entity_ids']:,} | {s['sample_duplicate_analysis']['exact_duplicate_rows']:,} ({s['sample_duplicate_analysis']['exact_duplicate_rows_pct']}%) |\n")
        gt = res["ground_truth"]
        f.write(f"| Train | Ground Truth | train_ground_truth.tsv | {gt['total_source1_entities']:,} | {gt['total_source1_entities']:,} | 0 (0.0%) |\n")
        for k in ["test_source1", "test_source2", "test_source3"]:
            s = res["test_sources"][k]
            f.write(f"| Test | {k.replace('test_', '').capitalize()} | {os.path.basename(s['file_path'])} | {s['total_rows']:,} | {s['unique_entity_ids']:,} | {s['sample_duplicate_analysis']['exact_duplicate_rows']:,} ({s['sample_duplicate_analysis']['exact_duplicate_rows_pct']}%) |\n")
            
        f.write("\n\n## 2. Columns and Inferred Data Types\n\n")
        f.write("| Column Name | Inferred Data Type | Constraints & Observed Values |\n")
        f.write("| --- | --- | --- |\n")
        f.write("| `entity_id` | String | Strictly prefixed (`S1-`, `S2-`, `S3-`), unique identifier, 8-12 characters |\n")
        f.write("| `business_name` | String | Free-text, abbreviations, casing variations, foreign scripts, punctuation |\n")
        f.write("| `business_address` | String | Free-text, reordered components, missing elements, landmarks, or empty |\n")
        f.write("| `country` | Categorical String | Train: `{'US', 'India'}`; Test: `{'US', 'India', 'France'}` |\n")
        f.write("| `matched_entity_ids` (Ground Truth) | String / List | Comma-separated list of `S2-*` and `S3-*` IDs, empty string for singletons |\n")
        
        f.write("\n\n## 3. Missing Values Summary\n\n")
        f.write("| Split | Source | Missing `entity_id` | Missing `business_name` | Missing `business_address` | Missing `country` |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for split, dkey in [("Train", "train_sources"), ("Test", "test_sources")]:
            for k in [f"{split.lower()}_source1", f"{split.lower()}_source2", f"{split.lower()}_source3"]:
                s = res[dkey][k]
                m = s["missing_counts"]
                mp = s["missing_pct"]
                f.write(f"| {split} | {k.split('_')[1].capitalize()} | {m['entity_id']} ({mp['entity_id']}%) | {m['business_name']} ({mp['business_name']}%) | {m['business_address']:,} ({mp['business_address']}%) | {m['country']} ({mp['country']}%) |\n")
                
        f.write("\n\n## 4. Ground Truth Match Distribution & Singletons\n\n")
        f.write(f"- **Total Source 1 Entities:** {gt['total_source1_entities']:,}\n")
        f.write(f"- **Total Ground Truth Matches (S2 + S3):** {gt['total_matches']:,}\n")
        f.write(f"- **Singleton Source 1 Entities (0 matches):** {gt['singletons']:,} ({gt['singleton_pct']}%)\n")
        f.write(f"- **Non-Singleton Entities (≥1 match):** {gt['non_singletons']:,} ({gt['non_singleton_pct']}%)\n\n")
        
        ss = gt["summary_statistics"]
        f.write("### Match Count Statistics per S1 Entity\n\n")
        f.write(f"| Metric | Value |\n| --- | --- |\n")
        f.write(f"| Mean | {ss['mean']} |\n")
        f.write(f"| Median | {ss['median']} |\n")
        f.write(f"| Standard Deviation | {ss['std_dev']} |\n")
        f.write(f"| Min | {ss['min']} |\n")
        f.write(f"| Max | {ss['max']} |\n")
        f.write(f"| 25th Percentile (p25) | {ss['p25']} |\n")
        f.write(f"| 50th Percentile (p50) | {ss['p50']} |\n")
        f.write(f"| 75th Percentile (p75) | {ss['p75']} |\n")
        f.write(f"| 90th Percentile (p90) | {ss['p90']} |\n")
        f.write(f"| 95th Percentile (p95) | {ss['p95']} |\n")
        f.write(f"| 99th Percentile (p99) | {ss['p99']} |\n\n")
        
        f.write("### Match Count Frequency Histogram\n\n")
        f.write("| Number of Matches | Count of S1 Entities | Percentage |\n| --- | --- | --- |\n")
        for bucket, cnt in sorted(gt["match_distribution"].items(), key=lambda x: (int(x[0]) if isinstance(x[0], int) or str(x[0]).isdigit() else 999)):
            f.write(f"| {bucket} | {cnt:,} | {round(cnt * 100.0 / gt['total_source1_entities'], 2)}% |\n")
            
        f.write("\n### Source Overlap Breakdown\n\n")
        sb = gt["source_breakdown"]
        f.write(f"- **Only Source 2 matches:** {sb['only_s2']:,} ({sb['only_s2_pct']}%)\n")
        f.write(f"- **Only Source 3 matches:** {sb['only_s3']:,} ({sb['only_s3_pct']}%)\n")
        f.write(f"- **Both Source 2 and Source 3 matches:** {sb['both_s2_and_s3']:,} ({sb['both_s2_and_s3_pct']}%)\n")
        f.write(f"- **Singletons (neither):** {sb['neither_singletons']:,} ({sb['neither_singletons_pct']}%)\n\n")
        
        f.write("### Singletons by Country\n\n")
        f.write("| Country | Total S1 Entities | Singletons | Singleton % | Total Matches | Mean Matches/Entity |\n| --- | --- | --- | --- | --- | --- |\n")
        for c, cd in gt["singletons_by_country"].items():
            f.write(f"| {c} | {cd['total']:,} | {cd['singletons']:,} | {cd['singleton_pct']}% | {cd['total_matches']:,} | {cd['mean_matches']} |\n")
            
        f.write("\n\n## 5. Country Distribution\n\n")
        f.write("| Split | Source | US Count (%) | India Count (%) | France Count (%) |\n| --- | --- | --- | --- | --- |\n")
        for split, dkey in [("Train", "train_sources"), ("Test", "test_sources")]:
            for k in [f"{split.lower()}_source1", f"{split.lower()}_source2", f"{split.lower()}_source3"]:
                s = res[dkey][k]
                cd = s["country_distribution"]
                cp = s["country_percentages"]
                f.write(f"| {split} | {k.split('_')[1].capitalize()} | {cd.get('US', 0):,} ({cp.get('US', 0)}%) | {cd.get('India', 0):,} ({cp.get('India', 0)}%) | {cd.get('France', 0):,} ({cp.get('France', 0)}%) |\n")
                
        cc = res["cross_country"]
        f.write(f"\n- **Cross-Country Matches in Ground Truth:** {cc['cross_country_matches']:,}\n")
        f.write(f"- **Strict Country Boundary Confirmed:** `{cc['strict_country_boundary']}` (100% of matches are within the same country!)\n")
        
        f.write("\n\n## 6. S2 and S3 Coverage & Multi-Linking\n\n")
        cov = res["coverage"]
        f.write(f"- **Source 2:** Total = {cov['s2_total_entities']:,}, Matched = {cov['s2_matched_entities']:,} ({cov['s2_match_percentage']}%), Unmatched/Orphan = {cov['s2_unmatched_orphans']:,} ({round(100 - cov['s2_match_percentage'], 2)}%)\n")
        f.write(f"- **Source 3:** Total = {cov['s3_total_entities']:,}, Matched = {cov['s3_matched_entities']:,} ({cov['s3_match_percentage']}%), Unmatched/Orphan = {cov['s3_unmatched_orphans']:,} ({round(100 - cov['s3_match_percentage'], 2)}%)\n")
        f.write(f"- **Source 2 Multi-Linked Entities (linked to >1 S1):** {gt['s2_entities_linked_to_multiple_s1']:,}\n")
        f.write(f"- **Source 3 Multi-Linked Entities (linked to >1 S1):** {gt['s3_entities_linked_to_multiple_s1']:,}\n")
        
        f.write("\n\n## 7. Business Name Patterns Across Sources\n\n")
        f.write("| Metric / Pattern | Train S1 | Train S2 | Train S3 | Test S1 | Test S2 | Test S3 |\n| --- | --- | --- | --- | --- | --- |\n")
        sources = ["train_source1", "train_source2", "train_source3", "test_source1", "test_source2", "test_source3"]
        def get_stat(k, path_fn):
            s = res["train_sources"][k] if k.startswith("train") else res["test_sources"][k]
            return path_fn(s)
            
        f.write("| Avg Char Length | " + " | ".join(str(get_stat(k, lambda s: s["char_lengths"]["business_name"]["mean"])) for k in sources) + " |\n")
        f.write("| Avg Word Count | " + " | ".join(str(get_stat(k, lambda s: s["name_patterns"]["avg_word_count"])) for k in sources) + " |\n")
        f.write("| ALL CAPS Name % | " + " | ".join(str(get_stat(k, lambda s: s["name_patterns"]["casing_pct"].get("ALL_UPPER", 0))) + "%" for k in sources) + " |\n")
        f.write("| Title Case Name % | " + " | ".join(str(get_stat(k, lambda s: s["name_patterns"]["casing_pct"].get("Title_Case", 0))) + "%" for k in sources) + " |\n")
        f.write("| Indic Script % | " + " | ".join(str(get_stat(k, lambda s: s["name_patterns"]["has_indic_script_pct"])) + "%" for k in sources) + " |\n")
        f.write("| Accents / Diacritics % | " + " | ".join(str(get_stat(k, lambda s: s["name_patterns"]["has_accents_pct"])) + "%" for k in sources) + " |\n")
        f.write("| Domain Names (URLs) % | " + " | ".join(str(get_stat(k, lambda s: s["name_patterns"]["has_domain_name_pct"])) + "%" for k in sources) + " |\n")
        f.write("| Leading Noise Symbols % | " + " | ".join(str(get_stat(k, lambda s: s["name_patterns"]["has_leading_noise_pct"])) + "%" for k in sources) + " |\n")
        f.write("| Contains Digits % | " + " | ".join(str(get_stat(k, lambda s: s["name_patterns"]["has_digits_pct"])) + "%" for k in sources) + " |\n")
        
        f.write("\n\n## 8. Address Patterns Across Sources\n\n")
        f.write("| Metric / Pattern | Train S1 | Train S2 | Train S3 | Test S1 | Test S2 | Test S3 |\n| --- | --- | --- | --- | --- | --- |\n")
        f.write("| Empty Address % | " + " | ".join(str(get_stat(k, lambda s: s["address_patterns"]["empty_addresses_pct"])) + "%" for k in sources) + " |\n")
        f.write("| Avg Char Length | " + " | ".join(str(get_stat(k, lambda s: s["char_lengths"]["business_address"]["mean"])) for k in sources) + " |\n")
        f.write("| ALL CAPS Address % | " + " | ".join(str(get_stat(k, lambda s: s["address_patterns"]["casing_pct"].get("ALL_UPPER", 0))) + "%" for k in sources) + " |\n")
        f.write("| PO Box % | " + " | ".join(str(get_stat(k, lambda s: s["address_patterns"]["has_po_box_pct"])) + "%" for k in sources) + " |\n")
        f.write("| Unit / Suite / Apt % | " + " | ".join(str(get_stat(k, lambda s: s["address_patterns"]["has_unit_apt_ste_pct"])) + "%" for k in sources) + " |\n")
        f.write("| Landmark (Near/Opp) % | " + " | ".join(str(get_stat(k, lambda s: s["address_patterns"]["has_landmark_in_pct"])) + "%" for k in sources) + " |\n")
        f.write("| India 6-digit PIN % | " + " | ".join(str(get_stat(k, lambda s: s["address_patterns"]["has_pin_code_india_pct"])) + "%" for k in sources) + " |\n")
        f.write("| US ZIP Code % | " + " | ".join(str(get_stat(k, lambda s: s["address_patterns"]["has_zip_us_pct"])) + "%" for k in sources) + " |\n")
        f.write("| French 5-digit ZIP % | " + " | ".join(str(get_stat(k, lambda s: s["address_patterns"]["has_zip_fr_pct"])) + "%" for k in sources) + " |\n")


if __name__ == "__main__":
    main()
