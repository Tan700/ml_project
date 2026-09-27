"""
data_loader.py
--------------
Data loading and splitting utilities for the Business Entity Resolution dataset.
"""

import os
import random
from typing import Dict, List, Set, Tuple, Optional
from normalization import normalize_business_name, normalize_address


class EntityRecord:
    __slots__ = ("entity_id", "business_name", "norm_name", "business_address", "norm_address", "country")

    def __init__(self, entity_id: str, business_name: str, norm_name: str,
                 business_address: str, norm_address: str, country: str):
        self.entity_id = entity_id
        self.business_name = business_name
        self.norm_name = norm_name
        self.business_address = business_address
        self.norm_address = norm_address
        self.country = country


def load_source_records(file_path: str, normalize: bool = True,
                        limit: Optional[int] = None) -> Dict[str, EntityRecord]:
    """
    Loads TSV records from a source file into a dictionary indexed by entity_id.
    """
    records = {}
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline()
        count = 0
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                parts = parts + [""] * (4 - len(parts))
            elif len(parts) > 4:
                parts = parts[:3] + [" ".join(parts[3:])]

            eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            if normalize:
                norm_name = normalize_business_name(name)
                norm_addr = normalize_address(addr)
            else:
                norm_name = name.lower()
                norm_addr = addr.lower()

            records[eid] = EntityRecord(
                entity_id=eid,
                business_name=name,
                norm_name=norm_name,
                business_address=addr,
                norm_address=norm_addr,
                country=country.strip()
            )
            count += 1
            if limit is not None and count >= limit:
                break

    return records


def load_ground_truth(file_path: str) -> Dict[str, Set[str]]:
    """
    Loads ground truth mapping from source1_entity_id -> set of matched_entity_ids.
    """
    gt = {}
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            matched_str = parts[1].strip() if len(parts) > 1 else ""
            if matched_str:
                gt[s1_id] = set(m.strip() for m in matched_str.split(",") if m.strip())
            else:
                gt[s1_id] = set()

    return gt


def train_val_split_entities(s1_ids: List[str], val_ratio: float = 0.2,
                             seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Splits Source 1 entity IDs into train and validation sets at entity level.
    """
    rng = random.Random(seed)
    shuffled = list(s1_ids)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    val_ids = shuffled[:n_val]
    train_ids = shuffled[n_val:]
    return train_ids, val_ids
