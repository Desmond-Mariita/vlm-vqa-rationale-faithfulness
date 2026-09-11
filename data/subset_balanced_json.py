#!/usr/bin/env python3
"""
subset_balanced_json.py

Create a balanced subset from a JSON dataset while preserving the original file structure.

What this does:
- Loads an input JSON file whose samples live either at the top-level list or inside
  a top-level dict (e.g., {"data": [...], "meta": {...}}).
- Detects the list of samples and the label field automatically (default "label" but
  robust to alternatives like "answer_label", "target", "gold_label", "y").
- Computes a per-class balanced sample count based on the requested percentage.
- Selects samples in a class-balanced way (with deterministic shuffling via --seed).
- Writes the output JSON with the SAME structure as input (same keys, only the sample
  list is reduced), pretty-printed.

Usage:
    python subset_balanced_json.py \
        --input /path/to/sample.json \
        --output /path/to/sample_subset_10.json \
        --pct 10 \
        --label-key label \
        --seed 42
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

# -----------------------------
# I/O + Structure Detection
# -----------------------------

def load_json(path: str) -> Any:
    """
    Load JSON from disk.

    Returns:
        The parsed JSON object (could be a list or dict).
    """
    with open(path, "r") as f:
        return json.load(f)


def find_sample_list_root(obj: Any) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    """
    Locate the list of samples in the loaded JSON.

    Returns:
        (sample_list, root_type, key)
        - sample_list: the list of sample dicts we will subset
        - root_type: either "list" or "dict"
        - key: if root_type == "dict", the key where the list lives (e.g., "data");
               if root_type == "list", this is None.

    Raises:
        ValueError: if no suitable list-of-dicts is found.
    """
    # Case 1: top-level is already a list of dicts
    if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict)):
        return obj, "list", None

    # Case 2: top-level is a dict; find the first key that looks like a list of dicts
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
                return v, "dict", k

    raise ValueError(
        "Could not locate a list of samples. Expected either a top-level list of dicts "
        "or a dict containing a key with a list-of-dicts (e.g., {'data': [...]})."
    )


# -----------------------------
# Label Handling
# -----------------------------

CANDIDATE_LABEL_KEYS = ["label", "answer_label", "target", "gold_label", "y"]

def resolve_label_key(samples: List[Dict[str, Any]], preferred: Optional[str] = None) -> str:
    """
    Determine which key contains the class label.

    Order of precedence:
    1) A user-provided key via --label-key if it exists in at least one sample.
    2) The first key from CANDIDATE_LABEL_KEYS that exists in at least one sample.

    Raises:
        ValueError if no label-like key is found.
    """
    if preferred:
        if any(preferred in s for s in samples):
            return preferred

    for k in CANDIDATE_LABEL_KEYS:
        if any(k in s for s in samples):
            return k

    raise ValueError(
        f"No label key found. Tried preferred={preferred!r} and candidates={CANDIDATE_LABEL_KEYS}."
    )


def coerce_label(value: Any) -> Any:
    """
    Normalize label values for grouping.

    - Leaves ints/strings as-is.
    - Converts other hashable types to string; unhashable to repr.

    Returns:
        A hashable label usable as a dict key.
    """
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


# -----------------------------
# Balanced Subset Selection
# -----------------------------

def compute_balanced_counts(
    by_class: Dict[Any, List[int]],
    total_desired: int
) -> Dict[Any, int]:
    """
    Compute how many samples to take from each class to achieve a balanced subset.

    Strategy:
    - Start with base = floor(total_desired / num_classes) for each class.
    - Cap by available class size.
    - Distribute leftover (from rounding and small classes) to classes with remaining capacity.

    Args:
        by_class: mapping label -> list of indices in the original sample list.
        total_desired: total number of items we want in the subset.

    Returns:
        dict label -> count to take
    """
    labels = list(by_class.keys())
    num_classes = len(labels)
    if num_classes == 0:
        return {}

    base = total_desired // num_classes
    counts = {c: min(base, len(by_class[c])) for c in labels}
    taken = sum(counts.values())

    # Leftover from rounding or classes with fewer than base
    leftover = total_desired - taken
    if leftover <= 0:
        return counts

    # Compute per-class remaining capacity
    capacity = {c: len(by_class[c]) - counts[c] for c in labels}

    # Distribute leftover one-by-one to classes with capacity, in round-robin
    # (stable, simple, sufficiently balanced)
    while leftover > 0 and any(capacity[c] > 0 for c in labels):
        for c in labels:
            if leftover == 0:
                break
            if capacity[c] > 0:
                counts[c] += 1
                capacity[c] -= 1
                leftover -= 1

    return counts


def select_balanced_subset_indices(
    samples: List[Dict[str, Any]],
    label_key: str,
    pct: float,
    seed: int = 42
) -> List[int]:
    """
    Compute a balanced subset of indices given a target percentage.

    Steps:
    1) Group sample indices by label.
    2) Shuffle indices per class deterministically.
    3) Compute how many to take per class via compute_balanced_counts.
    4) Slice the shuffled lists per class and concatenate.

    Args:
        samples: list of sample dicts.
        label_key: key used to read the class label in each sample.
        pct: percentage (0-100] of the full dataset to keep.
        seed: random seed for deterministic shuffling.

    Returns:
        A list of chosen indices (sorted for stable output).
    """
    assert 0 < pct <= 100, "--pct must be in (0, 100]."
    total = len(samples)
    total_desired = max(1, math.floor((pct / 100.0) * total))

    by_class: Dict[Any, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        raw = s.get(label_key, None)
        lbl = coerce_label(raw)
        by_class[lbl].append(i)

    rng = random.Random(seed)
    for lst in by_class.values():
        rng.shuffle(lst)

    per_class_counts = compute_balanced_counts(by_class, total_desired)

    chosen = []
    for lbl, cnt in per_class_counts.items():
        chosen.extend(by_class[lbl][:cnt])

    # If extremely imbalanced and still short, fill with remaining (deterministic order)
    if len(chosen) < total_desired:
        remaining = []
        for lbl, idxs in by_class.items():
            remaining.extend(idxs[per_class_counts[lbl]:])
        chosen_set = set(chosen)
        for idx in remaining:
            if len(chosen) >= total_desired:
                break
            if idx not in chosen_set:
                chosen.append(idx)
                chosen_set.add(idx)

    return sorted(chosen)


# -----------------------------
# Subset + Save (preserve structure)
# -----------------------------

def build_output_structure(
    original: Union[List[Dict[str, Any]], Dict[str, Any]],
    root_type: str,
    list_key: Optional[str],
    subset_indices: List[int]
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Reconstruct the JSON object with the subset while preserving the original shape.

    Args:
        original: the loaded JSON (list or dict).
        root_type: "list" or "dict" as returned by find_sample_list_root.
        list_key: if root_type == "dict", the key where the samples live; else None.
        subset_indices: indices to keep from the sample list.

    Returns:
        A new JSON object with the same structure, but reduced sample list.
    """
    if root_type == "list":
        # Simply pick the subset from the top-level list
        return [original[i] for i in subset_indices]

    # root_type == "dict"
    out = dict(original)  # shallow copy
    samples = original[list_key]
    out[list_key] = [samples[i] for i in subset_indices]
    return out


def save_json(obj: Any, path: str) -> None:
    """
    Save JSON to disk with pretty formatting and UTF-8 encoding.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for creating a balanced subset.
    """
    p = argparse.ArgumentParser(description="Create a balanced subset from a JSON dataset.")
    p.add_argument("--input", required=True, help="Path to the input JSON (e.g., sample.json)")
    p.add_argument("--output", required=True, help="Path to write the subset JSON")
    p.add_argument("--pct", type=float, required=True, help="Percentage of data to keep (0-100]")
    p.add_argument("--label-key", default='answer_label', help="Label key (default: auto-detect; e.g., 'label')")
    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic shuffling")
    return p.parse_args()


def main() -> None:
    """
    Entry point: load → detect structure → select indices → rebuild → save.
    """
    args = parse_args()

    data = load_json(args.input)
    samples, root_type, list_key = find_sample_list_root(data)
    label_key = resolve_label_key(samples, preferred=args.label_key)

    subset_indices = select_balanced_subset_indices(
        samples=samples,
        label_key=label_key,
        pct=args.pct,
        seed=args.seed,
    )

    out_obj = build_output_structure(
        original=data,
        root_type=root_type,
        list_key=list_key,
        subset_indices=subset_indices,
    )
    save_json(out_obj, args.output)

    kept = len(subset_indices)
    total = len(samples)
    print(f"✅ Saved balanced subset: {kept}/{total} samples ({kept/total:.1%}) → {args.output}")
    print(f"   Structure preserved: root={root_type}, list_key={list_key or '[top-level]'}, label_key={label_key}")


if __name__ == "__main__":
    main()
