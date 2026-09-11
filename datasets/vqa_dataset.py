# -*- coding: utf-8 -*-
"""
VQADataset (Route 2: Qwen2.5-VL + LoRA)

Features
--------
- Separate JSON files for train/val
- Separate image roots for train/val and for plain/polygon sets
- REQUIRED field mapping:
    image_id       -> POLYGON image (lives in images_point_[split])
    orig_image_id  -> PLAIN image   (lives in images_plain_[split])
- Arms:
    "plain"       -> load PLAIN image only
    "point"       -> load POLYGON image only
    "plain_desc"  -> PLAIN image + caption (from `caption_key`)
    "point_desc"  -> POLYGON image + caption (from `caption_key`)
- Returns explicit IDs and resolved absolute paths so later stages (tiles/reports)
  can render images without guessing:
    image_id, orig_image_id, image_path_point, image_path_plain, index
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from torch.utils.data import Dataset

__all__ = ["VQADataset", "vqa_collate_fn"]

_ALLOWED_SPLITS = {"train", "val"}
_ALLOWED_ARMS = {"plain", "point", "plain_desc", "point_desc"}


class VQADataset(Dataset):
    """Multi-arm VQA dataset for Qwen2.5-VL + LoRA training.

    Supports four experimental arms (plain, point, plain_desc, point_desc)
    by resolving separate image roots for plain and polygon-annotated images
    and optionally attaching captions. Records with missing files or fields
    can be silently dropped or raised as errors.
    """

    def __init__(
        self,
        json_train_path: str,
        json_val_path: str,
        images_plain_train: str,
        images_plain_val: str,
        images_point_train: str,
        images_point_val: str,
        split: str = "train",
        arm: str = "plain",
        caption_key: str = "image_descriptions.caption",
        drop_missing: bool = True,
    ) -> None:
        """Initialise the dataset by loading and validating JSON records.

        Args:
            json_train_path: Path to the training-split JSON/JSONL file.
            json_val_path: Path to the validation-split JSON/JSONL file.
            images_plain_train: Root directory for plain (no-polygon) training images.
            images_plain_val: Root directory for plain validation images.
            images_point_train: Root directory for polygon-annotated training images.
            images_point_val: Root directory for polygon-annotated validation images.
            split: Dataset split to use, either ``"train"`` or ``"val"``.
            arm: Experimental arm. One of ``"plain"``, ``"point"``,
                ``"plain_desc"``, or ``"point_desc"``.
            caption_key: Dot-separated path into the JSON record for
                retrieving captions (used only for ``*_desc`` arms).
            drop_missing: If ``True``, silently drop records whose required
                image files or fields are missing; otherwise raise an error.

        Raises:
            ValueError: If *split* or *arm* is not in the allowed set, or if
                no usable records remain after validation.
        """
        if split not in _ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {_ALLOWED_SPLITS}, got {split}")
        if arm not in _ALLOWED_ARMS:
            raise ValueError(f"arm must be one of {_ALLOWED_ARMS}, got {arm}")

        self.split = split
        self.arm = arm
        self.caption_key = caption_key
        self.drop_missing = drop_missing

        # Resolve roots by split
        self.json_path = Path(json_train_path if split == "train" else json_val_path)
        self.images_dir_plain = Path(images_plain_train if split == "train" else images_plain_val)
        self.images_dir_point = Path(images_point_train if split == "train" else images_point_val)

        # Load
        records = self._load_records(self.json_path)

        # If the file has a 'split' key, filter; otherwise assume it's already split-specific
        if any(isinstance(r, dict) and ("split" in r) for r in records):
            records = [r for r in records if r.get("split") == self.split]

        if not records:
            raise ValueError(f"No records found in {self.json_path} for split={self.split}")

        # Validate & keep only usable rows
        n_pre = len(records)
        drop_missing_ids = 0
        drop_missing_image = 0
        drop_bad_choices = 0
        validated: List[Dict[str, Any]] = []
        for r in records:
            poly_name = r.get("image_id")         # polygon image filename
            plain_name = r.get("orig_image_id")   # plain image filename
            if not poly_name or not plain_name:
                if self.drop_missing:
                    drop_missing_ids += 1
                    continue
                raise ValueError("Record missing required 'image_id' or 'orig_image_id'.")

            # Resolve absolute paths for both versions
            point_path = self.images_dir_point / poly_name
            plain_path = self.images_dir_plain / plain_name

            # You may want to require both files to exist; here we require only the one needed for current arm.
            if self.arm.startswith("plain"):
                # "plain" and "plain_desc" require the plain image to exist
                need_path = plain_path
            else:
                # "point" and "point_desc" require the polygon image
                need_path = point_path

            if self.drop_missing and not need_path.exists():
                drop_missing_image += 1
                continue

            # choices normalization to length 4
            choices = r.get("choices", [])
            if not isinstance(choices, list):
                if self.drop_missing:
                    drop_bad_choices += 1
                    continue
                raise ValueError("`choices` must be a list.")
            if len(choices) != 4:
                r["choices"] = (choices + ["", "", "", ""])[:4]

            # keep resolved paths for downstream reporting (even if one side is missing)
            r["_image_path_point"] = str(point_path) if point_path.exists() else None
            r["_image_path_plain"] = str(plain_path) if plain_path.exists() else None

            validated.append(r)

        if not validated:
            raise ValueError("All records dropped due to missing files/fields for the selected split/arm.")

        n_post = len(validated)
        n_dropped = n_pre - n_post
        print(
            f"[VQADataset] split={self.split} arm={self.arm} "
            f"pre={n_pre} post={n_post} dropped={n_dropped} "
            f"(missing_ids={drop_missing_ids}, missing_image={drop_missing_image}, "
            f"bad_choices={drop_bad_choices})",
            flush=True,
        )

        # Persist the kept-IDs manifest (fail-open: don't block training if disk is unwritable).
        manifest_dir = os.environ.get("VQA_DATASET_MANIFEST_DIR")
        if manifest_dir:
            try:
                Path(manifest_dir).mkdir(parents=True, exist_ok=True)
                manifest_path = Path(manifest_dir) / f"vqa_kept_{self.split}_{self.arm}.json"
                kept_ids = [
                    {
                        "question_id": r.get("question_id"),
                        "image_id": r.get("image_id"),
                        "orig_image_id": r.get("orig_image_id"),
                    }
                    for r in validated
                ]
                with manifest_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "split": self.split,
                            "arm": self.arm,
                            "pre": n_pre,
                            "post": n_post,
                            "dropped": n_dropped,
                            "drop_missing_ids": drop_missing_ids,
                            "drop_missing_image": drop_missing_image,
                            "drop_bad_choices": drop_bad_choices,
                            "kept": kept_ids,
                        },
                        f,
                    )
                print(f"[VQADataset] kept-IDs manifest -> {manifest_path}", flush=True)
            except Exception as e:
                print(f"[VQADataset] WARN: failed to write kept-IDs manifest: {e}", flush=True)

        self._records = validated

    # ---------- IO ----------
    def _load_records(self, path: Path) -> List[Dict[str, Any]]:
        """Load records from a JSON or JSONL file.

        Args:
            path: Path to the data file. JSONL files are read line-by-line;
                JSON files are expected to be a list or a dict with a
                ``"data"`` key containing a list.

        Returns:
            List of record dictionaries.
        """
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        else:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data["data"] if isinstance(data, dict) and "data" in data else data

    @staticmethod
    def _get_nested(d: Dict[str, Any], dotted: str) -> Optional[Any]:
        """Retrieve a value from a nested dict using a dot-separated key path.

        Args:
            d: The dictionary to traverse.
            dotted: Dot-separated key path (e.g. ``"image_descriptions.caption"``).

        Returns:
            The value at the nested key, or ``None`` if any key is missing.
        """
        cur: Any = d
        for key in dotted.split("."):
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return cur

    # ---------- Dataset API ----------
    def __len__(self) -> int:
        """Return the number of validated records in the dataset."""
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch a single sample by index.

        Loads the appropriate image (plain or polygon) based on the
        configured arm, resolves absolute paths for both image variants,
        and extracts question, choices, label, caption, and gold rationale.

        Args:
            idx: Zero-based index into the validated record list.

        Returns:
            Dictionary containing ``"id"``, ``"index"``, ``"image"``
            (PIL.Image), ``"question"``, ``"choices"``, ``"label"``,
            ``"caption"``, ``"image_id"``, ``"orig_image_id"``,
            ``"image_path_point"``, ``"image_path_plain"``,
            ``"metadata"``, and ``"rationale_gold"``.
        """
        r = self._records[idx]
        poly_name: str = r["image_id"]
        plain_name: str = r["orig_image_id"]

        # Choose which image to load for model input based on arm
        if self.arm.startswith("plain"):
            img_path = Path(r["_image_path_plain"]) if r.get("_image_path_plain") else (self.images_dir_plain / plain_name)
        else:
            # "point" and "point_desc" read the polygon image
            img_path = Path(r["_image_path_point"]) if r.get("_image_path_point") else (self.images_dir_point / poly_name)

        # Open image (required for model)
        image = Image.open(img_path).convert("RGB")

        # Fields
        question = r.get("question", "")
        choices = r.get("choices", ["", "", "", ""])
        label = r.get("answer_label", -1)
        caption = self._get_nested(r, self.caption_key) if self.arm.endswith("_desc") else None

        # --- robust handling of `rationales` field ---
        raw_rat = r.get("rationales", None)
        if isinstance(raw_rat, str):
            rationale_gold = raw_rat
        elif isinstance(raw_rat, list) and raw_rat:
            rationale_gold = raw_rat[0]
        else:
            rationale_gold = None

        # Resolved absolute paths (both variants), even if we only loaded one for the model
        point_abs = r.get("_image_path_point")
        plain_abs = r.get("_image_path_plain")
        if point_abs is None:
            p = self.images_dir_point / poly_name
            point_abs = str(p) if p.exists() else None
        if plain_abs is None:
            p = self.images_dir_plain / plain_name
            plain_abs = str(p) if p.exists() else None

        return {
            "id": r.get("question_id", idx),
            "index": idx,
            "image": image,                 # PIL.Image for HF processor
            "question": question,
            "choices": choices,             # list[str] of len 4
            "label": label,
            "caption": caption,             # only non-None for point_desc
            # Explicit IDs and paths for reporting
            "image_id": poly_name,          # polygon filename
            "orig_image_id": plain_name,    # plain filename
            "image_path_point": point_abs,  # absolute if exists, else None
            "image_path_plain": plain_abs,  # absolute if exists, else None
            # Keep a small metadata blob for provenance
            "metadata": {
                "split": self.split,
                "json_path": str(self.json_path),
            },
            "rationale_gold": rationale_gold,
        }



# ---------- Collate ----------
def vqa_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of dataset samples into a batched dictionary.

    Keeps PIL images and raw strings (no tensor conversion) so that
    the HuggingFace processor can handle tokenisation downstream.
    Also collates explicit IDs and resolved paths for evaluation.

    Args:
        batch: List of sample dicts as returned by
            :meth:`VQADataset.__getitem__`.

    Returns:
        Dictionary with batched lists keyed by ``"ids"``, ``"images"``,
        ``"questions"``, ``"choices"``, ``"labels"``, ``"captions"``,
        ``"image_id"``, ``"orig_image_id"``, ``"image_path_point"``,
        ``"image_path_plain"``, ``"metadata"``, ``"meta"``, and
        ``"rationale_gold"``.
    """
    return {
        "ids": [b["id"] for b in batch],
        "index": [b["index"] for b in batch],
        "images": [b["image"] for b in batch],
        "questions": [b["question"] for b in batch],
        "choices": [b["choices"] for b in batch],
        "labels": [b["label"] for b in batch],
        "captions": [b["caption"] for b in batch],
        # explicit identifiers and resolved paths
        "image_id": [b.get("image_id") for b in batch],
        "orig_image_id": [b.get("orig_image_id") for b in batch],
        "image_path_point": [b.get("image_path_point") for b in batch],
        "image_path_plain": [b.get("image_path_plain") for b in batch],
        # meta blobs (and alias to 'meta' for older code paths)
        "metadata": [b.get("metadata", {}) for b in batch],
        "meta": [b.get("metadata", {}) for b in batch],
        "rationale_gold": [b["rationale_gold"] for b in batch],
    }
