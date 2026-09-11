#!/usr/bin/env python3
"""Patch the `rationales` field in cleaned VCR JSONs.

The original cleaning scripts (clean_vcr_dataset_generator.py and
vcr_dataset_clean4.py) selected `rationale_choices[answer_label]` as the
ground-truth rationale; that field is independent of `rationale_label` in
VCR, so the picked rationale matches the canonical correct rationale only
in ~25% of records and is a distractor for the remaining ~75%.

This script rewrites every `vcr_*.json` under `data/final/` so that
`rationales` holds `rationale_choices[rationale_label]` (rendered with the
same person_X / object_Y substitution the original cleaning step used).
Patched files are written to `data/final_v2/` so the live training run is
untouched until the user is ready to swap.

Run:
    python data_prep/canonical/patch_rationales_to_label.py
"""

from __future__ import annotations
import json
import os
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm

PROJECT = Path(".")
RAW_ANN = PROJECT / "data" / "raw" / "vcr1annots"
LIVE = PROJECT / "data" / "final"
STAGING = PROJECT / "data" / "final_v2"
STAGING.mkdir(parents=True, exist_ok=True)


# ---- token-list rendering (mirror of vcr_dataset_clean4.py) ----------------
def deep_flatten(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(deep_flatten(i) for i in x)
    return str(x)


_REF_RE = re.compile(r"(?<!\w)(\d+)(?!\w)")


def render(tokens, names):
    text = deep_flatten(tokens)

    def repl(m):
        i = int(m.group(1))
        return f"{names[i]}_{i}" if 0 <= i < len(names) else m.group(0)

    return _REF_RE.sub(repl, text)


# ---- build (image_basename, rendered_question) -> rationales index ---------
def build_raw_index():
    """Index raw VCR by (basename, question, choices, answer_label).

    The (image, question) pair is not unique in raw VCR --- the same
    image+question is sometimes annotated twice by different workers with
    different rationales. We disambiguate by including the rendered answer
    choices and the answer_label, which together pin the specific
    annotator's record.
    """
    print("Indexing raw VCR train+val…")
    idx: dict[tuple, tuple[str, str]] = {}
    collisions = 0
    for split in ("train", "val"):
        path = RAW_ANN / f"{split}.jsonl"
        with open(path) as f:
            for line in tqdm(f, desc=f"  {split}"):
                r = json.loads(line)
                basename = os.path.basename(r["img_fn"])
                names = r.get("objects", [])
                rq = render(r["question"], names)
                choices = tuple(render(c, names) for c in r["answer_choices"])
                al = r["answer_label"]
                rl = r["rationale_label"]
                key = (basename, rq, choices, al)
                if key in idx:
                    collisions += 1
                new_rat = render(r["rationale_choices"][rl], names)
                old_rat = render(r["rationale_choices"][al], names)
                idx[key] = (new_rat, old_rat)
    print(f"  {len(idx)} keys indexed; {collisions} same-key collisions (overwrite)")
    return idx


# ---- strip the leading 8-char hash from cleaned orig_image_id --------------
# Most records have one hash prefix (e.g. "<hash>_<orig.jpg>"). One val record
# was found with two stacked hash prefixes, presumably from a buggy merge step
# that copied the polygon image_id into orig_image_id; we strip both to recover.
_HASH_RE = re.compile(r"^[0-9a-f]{8}_(.+)$")


def strip_hash(name: str) -> str:
    for _ in range(2):
        m = _HASH_RE.match(name)
        if not m:
            break
        name = m.group(1)
    return name


def patch_file(src: Path, idx, dst: Path):
    data = json.load(open(src))
    stats = Counter()
    miss_examples = []
    sanity_examples = []  # rationale-currently-in-cleaned vs rationale-at-answer_label

    for rec in data:
        stats["total"] += 1
        basename = strip_hash(rec["orig_image_id"])
        choices = tuple(rec["choices"])
        key = (basename, rec["question"], choices, rec["answer_label"])
        hit = idx.get(key)
        if hit is None:
            stats["miss"] += 1
            if len(miss_examples) < 3:
                miss_examples.append({"basename": basename, "question": rec["question"]})
            continue
        new_rat, old_rat = hit
        cur = rec["rationales"].strip()
        # Sanity: the rationale currently in cleaned JSON should match old_rat
        # (the rationale at answer_label). Soft-check, log mismatches.
        if cur != old_rat.strip() and len(sanity_examples) < 3:
            sanity_examples.append(
                {"cur": cur[:90], "old_rat": old_rat[:90]}
            )
        if cur == new_rat.strip():
            stats["unchanged"] += 1
        else:
            rec["rationales"] = new_rat
            stats["changed"] += 1

    json.dump(data, open(dst, "w"), indent=2, ensure_ascii=False)
    return stats, miss_examples, sanity_examples


def main():
    idx = build_raw_index()
    files = sorted(LIVE.glob("vcr_*.json"))
    files = [f for f in files if "_pjx" not in f.name]  # skip older pjx-format files
    grand = Counter()
    print(f"\nPatching {len(files)} JSON files…")
    for src in files:
        dst = STAGING / src.name
        stats, misses, sanity = patch_file(src, idx, dst)
        grand.update(stats)
        print(
            f"  {src.name:42s}  total={stats['total']:>6}  "
            f"changed={stats['changed']:>6}  "
            f"unchanged={stats['unchanged']:>4}  miss={stats['miss']}"
        )
        if misses:
            print(f"    misses (first 3): {misses}")
        if sanity:
            print(f"    sanity: cleaned-vs-answer_label diffs (first 3): {sanity}")
    print()
    print("Grand totals:")
    for k in ("total", "changed", "unchanged", "miss"):
        print(f"  {k:>10}: {grand[k]}")
    print(f"\nPatched JSONs in {STAGING}")
    print(
        "Next step (after Stage 1 finishes): swap the JSONs into place, e.g.\n"
        "    cd data && mkdir -p final/_oldjsons && mv final/vcr_*.json final/_oldjsons/\n"
        "    mv final_v2/vcr_*.json final/ && rmdir final_v2"
    )


if __name__ == "__main__":
    main()
