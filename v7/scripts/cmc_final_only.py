#!/usr/bin/env python3
"""Precommitted v7 cross-modal consistency analysis.

This script is deliberately separate from the legacy CMC implementation.  It
uses only a provenance-verified ``rationale_final`` span, reports conditional
and end-to-end (pipeline) estimands, evaluates a question-only baseline, and
clusters all uncertainty calculations by source frame.

The frozen inputs and decisions live in
``v7/Artifacts/hardening_20260817/PROTOCOL_MANIFEST.json``.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
V7_ROOT = REPO_ROOT / "v7"
MANIFEST_PATH = V7_ROOT / "Artifacts" / "hardening_20260817" / "PROTOCOL_MANIFEST.json"
DEFAULT_OUT = V7_ROOT / "Artifacts" / "hardening_20260817" / "cmc_clean"

_TAG = re.compile(r"<\s*(/?)\s*(reason\w*|final\w*)[^\s<>]*\s*>?", re.IGNORECASE)
_DEBRIS = re.compile(r"<\s*/?\s*(reason|final)", re.IGNORECASE)
_ANY_TAGISH = re.compile(r"<\s*/?\s*\w[^<>]*>?")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NON_ALNUM = re.compile(r"[^\w]+", re.UNICODE)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_match(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    return normalize_space(_NON_ALNUM.sub(" ", value))


def has_tag_debris(text: Any) -> bool:
    return isinstance(text, str) and bool(_DEBRIS.search(text))


def _tag_kind(slash: str, name: str) -> str:
    base = "reason" if name.lower().startswith("reason") else "final"
    return ("close_" if slash else "open_") + base


def _strip_tagish(text: str) -> str:
    return normalize_space(_ANY_TAGISH.sub(" ", _TAG.sub(" ", text)))


def parse_cot(text: Any) -> Tuple[str, str]:
    """Local copy of the frozen malformation-tolerant span parser."""
    if not isinstance(text, str) or not text.strip():
        return "", ""
    value = text.strip()
    tags = [(_tag_kind(m.group(1), m.group(2)), m.start(), m.end()) for m in _TAG.finditer(value)]
    reasoning = ""
    final = ""
    for i, (kind, _start, end) in enumerate(tags):
        if kind != "open_reason":
            continue
        stop = len(value)
        for next_kind, next_start, _next_end in tags[i + 1 :]:
            if next_kind in ("close_reason", "open_final"):
                stop = next_start
                break
        reasoning = _strip_tagish(value[end:stop])
        break
    for i, (kind, _start, end) in enumerate(tags):
        if kind != "open_final":
            continue
        stop = len(value)
        for next_kind, next_start, _next_end in tags[i + 1 :]:
            if next_kind == "close_final":
                stop = next_start
                break
        final = _strip_tagish(value[end:stop])
        break
    if not reasoning and not final:
        clean = _strip_tagish(value)
        sentences = [part.strip() for part in _SENT_SPLIT.split(clean) if part.strip()]
        if len(sentences) >= 2:
            reasoning, final = " ".join(sentences[:-1]), sentences[-1]
        else:
            final = clean
    return reasoning, final


def lineage_status(current: Mapping[str, Any], original: Mapping[str, Any], same_file: bool) -> Tuple[str, bool]:
    """Return an exact provenance category and final-span eligibility flag."""
    current_final = normalize_space(current.get("rationale_final"))
    if not current_final:
        return "empty_final", False

    original_gen = normalize_space(original.get("rationale_gen"))
    original_final = normalize_space(original.get("rationale_final"))
    original_reasoning = normalize_space(original.get("rationale_reasoning"))

    if same_file:
        if current_final == original_final == original_gen:
            return "verified_writer_completion", True
        return "ambiguous_writer_field_mismatch", False

    fields = [
        ("rationale_gen", original_gen),
        ("rationale_final", original_final),
        ("rationale_reasoning", original_reasoning),
    ]
    if not any(has_tag_debris(value) for _name, value in fields):
        if current_final in {original_gen, original_final}:
            return "verified_unchanged_completion", True
        return "ambiguous_unexpected_reparse_without_debris", False

    # This exactly mirrors max((gen, fin, rea), key=len): ties go to the first
    # occurrence, which is rationale_gen.
    selected_name, selected_text = max(fields, key=lambda item: len(item[1]))
    _derived_reasoning, derived_final = parse_cot(selected_text)
    if current_final != normalize_space(derived_final):
        return "ambiguous_reparse_mismatch", False
    if selected_name in {"rationale_gen", "rationale_final"}:
        return f"verified_reparsed_from_{selected_name}", True
    return "ambiguous_reparsed_from_reasoning", False


def render_baseline_prompt(question: str, choices: Sequence[str]) -> str:
    choice_text = "\n".join(f"({index}) {choice}" for index, choice in enumerate(choices))
    return (
        "You are answering a multiple-choice question without access to its image.\n"
        "Use only the question and answer choices. Pick the single best choice.\n\n"
        f"Question: {question}\n\nChoices:\n{choice_text}\n\n"
        "Answer with exactly one digit: 0, 1, 2, or 3.\nAnswer:"
    )


def render_rationale_prompt(question: str, choices: Sequence[str], rationale_final: str) -> str:
    choice_text = "\n".join(f"({index}) {choice}" for index, choice in enumerate(choices))
    return (
        "You are judging a multiple-choice question without access to its image.\n"
        "Another system supplied an explanation. Use only the question, choices, and explanation.\n\n"
        f"Question: {question}\n\nChoices:\n{choice_text}\n\n"
        f"Explanation:\n{rationale_final.strip()}\n\n"
        "Which choice is best supported? Answer with exactly one digit: 0, 1, 2, or 3.\nAnswer:"
    )


def prompt_sentinel_test() -> Dict[str, Any]:
    forbidden = {
        "image_id": "FORBIDDEN_IMAGE_MARKER_7D2D",
        "arm": "FORBIDDEN_ARM_MARKER_7D2D",
        "answer_label": "FORBIDDEN_ANSWER_MARKER_7D2D",
        "generation_flag": "FORBIDDEN_FLAG_MARKER_7D2D",
        "rationale_reasoning": "FORBIDDEN_REASONING_MARKER_7D2D",
        "rationale_gold": "FORBIDDEN_GOLD_MARKER_7D2D",
        "prompt": "FORBIDDEN_SOURCE_PROMPT_MARKER_7D2D",
        "caption": "FORBIDDEN_CAPTION_MARKER_7D2D",
    }
    question = "SENTINEL_QUESTION"
    choices = ["choice zero", "choice one", "choice two", "choice three"]
    final = "SENTINEL_FINAL_ONLY"
    baseline = render_baseline_prompt(question, choices)
    rationale = render_rationale_prompt(question, choices, final)
    leaked = [marker for marker in forbidden.values() if marker in baseline or marker in rationale]
    assert not leaked, f"Forbidden marker leaked into prompt: {leaked}"
    assert final not in baseline and final in rationale
    assert "SENTINEL_QUESTION" in baseline and "SENTINEL_QUESTION" in rationale
    return {
        "passed": True,
        "forbidden_markers_tested": sorted(forbidden.values()),
        "baseline_sha256": sha256_text(baseline),
        "rationale_sha256": sha256_text(rationale),
    }


def verify_manifest_inputs(manifest: Mapping[str, Any]) -> None:
    failures: List[str] = []
    for cell in manifest["cmc"]["cells"]:
        for path_key, hash_key in (("current_path", "current_sha256"), ("original_path", "original_sha256")):
            path = REPO_ROOT / cell[path_key]
            if not path.exists():
                failures.append(f"missing {path}")
                continue
            observed = sha256_file(path)
            if observed != cell[hash_key]:
                failures.append(f"hash mismatch {path}: {observed} != {cell[hash_key]}")
    if failures:
        raise RuntimeError("Frozen input verification failed:\n" + "\n".join(failures))


def build_audit(manifest: Mapping[str, Any], out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    verify_manifest_inputs(manifest)
    sentinel = prompt_sentinel_test()
    audit_rows: List[Dict[str, Any]] = []
    tasks: Dict[str, str] = {}
    prompt_samples: List[Dict[str, Any]] = []

    for cell in manifest["cmc"]["cells"]:
        arm = str(cell["arm"])
        setting = str(cell["setting"])
        current_path = REPO_ROOT / cell["current_path"]
        original_path = REPO_ROOT / cell["original_path"]
        current_rows = read_jsonl(current_path)
        original_rows = current_rows if current_path == original_path else read_jsonl(original_path)
        if len(current_rows) != int(cell["records"]) or len(original_rows) != len(current_rows):
            raise RuntimeError(f"Record-count mismatch for {arm}/{setting}")

        for row_index, (current, original) in enumerate(zip(current_rows, original_rows)):
            for key in ("image_id", "question"):
                if current.get(key) != original.get(key):
                    raise RuntimeError(f"Alignment failure in {arm}/{setting}, row {row_index}, field {key}")
            choices = current.get("choices") or []
            if not isinstance(choices, list) or len(choices) != 4:
                raise RuntimeError(f"Invalid choices in {arm}/{setting}, row {row_index}")
            try:
                target = int(current["answer_idx"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid answer_idx in {arm}/{setting}, row {row_index}") from exc
            if target not in (0, 1, 2, 3):
                raise RuntimeError(f"Out-of-range answer_idx in {arm}/{setting}, row {row_index}")

            status, verified = lineage_status(current, original, current_path == original_path)
            final = normalize_space(current.get("rationale_final"))
            eligible = bool(verified and final)
            baseline_prompt = render_baseline_prompt(str(current.get("question") or ""), choices)
            baseline_hash = sha256_text(baseline_prompt)
            tasks.setdefault(baseline_hash, baseline_prompt)
            rationale_hash: Optional[str] = None
            if eligible:
                rationale_prompt = render_rationale_prompt(str(current.get("question") or ""), choices, final)
                rationale_hash = sha256_text(rationale_prompt)
                tasks.setdefault(rationale_hash, rationale_prompt)

            answer_norm = normalize_match(choices[target])
            final_norm = normalize_match(final)
            answer_tokens = set(answer_norm.split())
            final_tokens = set(final_norm.split())
            overlap = len(answer_tokens & final_tokens) / len(answer_tokens) if answer_tokens else 0.0
            exact_echo = bool(answer_norm and answer_norm in final_norm)
            cluster = current.get("orig_image_id") or current.get("image_id")
            audit = {
                "arm": arm,
                "setting": setting,
                "row_index": row_index,
                "image_id": current.get("image_id"),
                "cluster_id": cluster,
                "target": target,
                "lineage_status": status,
                "lineage_verified": verified,
                "eligible_final": eligible,
                "final_chars": len(final),
                "baseline_prompt_sha256": baseline_hash,
                "rationale_prompt_sha256": rationale_hash,
                "answer_echo_exact": exact_echo if eligible else False,
                "answer_token_recall": overlap if eligible else 0.0,
            }
            audit_rows.append(audit)
            if row_index < 2:
                prompt_samples.append(
                    {
                        "arm": arm,
                        "setting": setting,
                        "row_index": row_index,
                        "baseline_prompt": baseline_prompt,
                        "baseline_prompt_sha256": baseline_hash,
                        "rationale_prompt": (
                            tasks[rationale_hash] if rationale_hash is not None else None
                        ),
                        "rationale_prompt_sha256": rationale_hash,
                        "lineage_status": status,
                    }
                )

    counts: Dict[str, Any] = {}
    for row in audit_rows:
        key = f"{row['arm']}__{row['setting']}"
        cell = counts.setdefault(key, {"n": 0, "eligible": 0, "lineage": {}})
        cell["n"] += 1
        cell["eligible"] += int(row["eligible_final"])
        status = row["lineage_status"]
        cell["lineage"][status] = cell["lineage"].get(status, 0) + 1
    for cell in counts.values():
        cell["coverage"] = cell["eligible"] / cell["n"] if cell["n"] else 0.0

    write_jsonl(out_dir / "lineage_audit.jsonl", audit_rows)
    write_jsonl(out_dir / "prompt_samples.jsonl", prompt_samples)
    write_json(out_dir / "lineage_summary.json", {"sentinel": sentinel, "cells": counts, "unique_prompts": len(tasks)})
    return audit_rows, tasks


def _model_text(tokenizer: Any, prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    cache: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        cache[str(row["prompt_sha256"])] = row
    return cache


def score_prompts(
    alias: str,
    model_id: str,
    revision: str,
    tasks: Mapping[str, str],
    out_dir: Path,
    batch_size: int,
    max_length: int,
) -> Dict[str, Dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

    class AllowOnlyTokens(LogitsProcessor):
        def __init__(self, allowed_ids: Sequence[int]):
            self.allowed_ids = tuple(sorted(set(int(value) for value in allowed_ids)))

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            mask = torch.full_like(scores, float("-inf"))
            mask[:, list(self.allowed_ids)] = scores[:, list(self.allowed_ids)]
            return mask

    cache_path = out_dir / "judge_cache" / f"{alias}.jsonl"
    meta_path = out_dir / "judge_cache" / f"{alias}.meta.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expected_meta = {
        "alias": alias,
        "model_id": model_id,
        "revision": revision,
        "max_length": max_length,
    }
    if meta_path.exists():
        observed_meta = read_json(meta_path)
        for key, expected in expected_meta.items():
            if observed_meta.get(key) != expected:
                raise RuntimeError(f"Judge-cache metadata mismatch for {alias}: {key}")
    else:
        write_json(meta_path, {**expected_meta, "script_sha256": sha256_file(Path(__file__))})

    cached = load_cache(cache_path)
    pending = [(key, text) for key, text in tasks.items() if key not in cached]
    print(f"[CMC:{alias}] cached={len(cached)} pending={len(pending)}", flush=True)
    if not pending:
        return cached

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    choice_to_token: Dict[int, int] = {}
    for choice in range(4):
        ids = tokenizer.encode(str(choice), add_special_tokens=False)
        # SentencePiece tokenizers may encode a leading standalone whitespace
        # token followed by the literal digit token.  Constrain generation to
        # the unique token whose own decoded surface is the requested digit.
        candidates = [
            int(token_id)
            for token_id in ids
            if tokenizer.decode([token_id], skip_special_tokens=True).strip() == str(choice)
        ]
        if len(set(candidates)) != 1:
            raise RuntimeError(
                f"Choice {choice} has no unique literal digit token for {model_id}: "
                f"encoded={ids}, candidates={candidates}"
            )
        choice_to_token[choice] = candidates[0]
    if len(set(choice_to_token.values())) != 4:
        raise RuntimeError(f"Choice-token collision for {model_id}: {choice_to_token}")
    token_to_choice = {token: choice for choice, token in choice_to_token.items()}
    processor = LogitsProcessorList([AllowOnlyTokens(list(token_to_choice))])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    load_kwargs = {"revision": revision, "trust_remote_code": True, "dtype": dtype}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except TypeError:
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.to(device)
    model.eval()

    with cache_path.open("a", encoding="utf-8") as sink:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            model_inputs = [_model_text(tokenizer, prompt) for _key, prompt in batch]
            encoded = tokenizer(
                model_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=1,
                    min_new_tokens=1,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    logits_processor=processor,
                )
            new_ids = generated[:, encoded["input_ids"].shape[1] :]
            for (prompt_hash, _prompt), token_row in zip(batch, new_ids):
                token_id = int(token_row[0].item()) if token_row.numel() else -1
                prediction = token_to_choice.get(token_id)
                raw = tokenizer.decode(token_row, skip_special_tokens=True).strip()
                result = {
                    "prompt_sha256": prompt_hash,
                    "prediction": prediction,
                    "parse_failed": prediction is None,
                    "raw": raw,
                    "token_id": token_id,
                }
                sink.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                cached[prompt_hash] = result
            sink.flush()
            done = min(start + len(batch), len(pending))
            print(f"[CMC:{alias}] {done}/{len(pending)} new prompts", flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cached


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def bootstrap_cell(rows: Sequence[Mapping[str, Any]], replicates: int, seed: int) -> Dict[str, Any]:
    clusters: Dict[str, List[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        clusters[str(row["cluster_id"])].append(row)
    cluster_rows = list(clusters.values())
    aggregates = []
    for group in cluster_rows:
        n = len(group)
        eligible = sum(int(row["eligible_final"]) for row in group)
        rationale_correct = sum(int(row["rationale_correct"]) for row in group)
        baseline_correct = sum(int(row["baseline_correct"]) for row in group)
        baseline_eligible_correct = sum(
            int(row["baseline_correct"]) for row in group if row["eligible_final"]
        )
        aggregates.append((n, eligible, rationale_correct, baseline_correct, baseline_eligible_correct))
    values = np.asarray(aggregates, dtype=np.float64)
    cluster_count = len(values)
    rng = np.random.default_rng(seed)
    distributions: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
    batch_replicates = 500
    for start in range(0, replicates, batch_replicates):
        size = min(batch_replicates, replicates - start)
        draws = rng.integers(0, cluster_count, size=(size, cluster_count))
        totals = values[draws].sum(axis=1)
        n = totals[:, 0]
        eligible = totals[:, 1]
        rationale_correct = totals[:, 2]
        baseline_correct = totals[:, 3]
        baseline_eligible_correct = totals[:, 4]
        with np.errstate(divide="ignore", invalid="ignore"):
            distributions["coverage"].append(eligible / n)
            distributions["conditional_cmc"].append(rationale_correct / eligible)
            distributions["pipeline_cmc"].append(rationale_correct / n)
            distributions["baseline_all"].append(baseline_correct / n)
            distributions["baseline_subset"].append(baseline_eligible_correct / eligible)
            distributions["delta_pipeline"].append((rationale_correct - baseline_correct) / n)
            distributions["delta_conditional"].append(
                (rationale_correct - baseline_eligible_correct) / eligible
            )
    packed = {name: np.concatenate(chunks) for name, chunks in distributions.items()}
    result: Dict[str, Any] = {"clusters": cluster_count, "replicates": replicates, "seed": seed, "ci": {}}
    for name, distribution in packed.items():
        valid = distribution[np.isfinite(distribution)]
        result["ci"][name] = [float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))]
    delta = packed["delta_pipeline"]
    valid_delta = delta[np.isfinite(delta)]
    left = int(np.count_nonzero(valid_delta <= 0.0))
    right = int(np.count_nonzero(valid_delta >= 0.0))
    result["delta_pipeline_p"] = min(1.0, 2.0 * (1 + min(left, right)) / (len(valid_delta) + 1))
    return result


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def evaluate_judge(
    alias: str,
    audit_rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Mapping[str, Any]],
    out_dir: Path,
    replicates: int,
    seed: int,
    apply_bh: bool,
) -> Dict[str, Any]:
    scored: List[Dict[str, Any]] = []
    for audit in audit_rows:
        baseline = cache.get(str(audit["baseline_prompt_sha256"]), {})
        baseline_prediction = baseline.get("prediction")
        eligible = bool(audit["eligible_final"])
        rationale_prediction: Optional[int] = None
        rationale_parse_failed = False
        if eligible:
            rationale = cache.get(str(audit["rationale_prompt_sha256"]), {})
            rationale_prediction = rationale.get("prediction")
            rationale_parse_failed = bool(rationale.get("parse_failed", rationale_prediction is None))
        row = dict(audit)
        row.update(
            {
                "judge": alias,
                "baseline_prediction": baseline_prediction,
                "baseline_parse_failed": bool(baseline.get("parse_failed", baseline_prediction is None)),
                "baseline_correct": bool(baseline_prediction == audit["target"]),
                "rationale_prediction": rationale_prediction,
                "rationale_parse_failed": rationale_parse_failed,
                # Ambiguous/empty lineage and invalid judge outputs are false.
                "rationale_correct": bool(eligible and rationale_prediction == audit["target"]),
            }
        )
        scored.append(row)
    write_jsonl(out_dir / f"records_{alias}.jsonl", scored)

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in scored:
        grouped[(str(row["arm"]), str(row["setting"]))].append(row)
    cells: List[Dict[str, Any]] = []
    for (arm, setting), rows in sorted(grouped.items()):
        n = len(rows)
        eligible_rows = [row for row in rows if row["eligible_final"]]
        eligible = len(eligible_rows)
        rationale_correct = sum(int(row["rationale_correct"]) for row in rows)
        baseline_correct = sum(int(row["baseline_correct"]) for row in rows)
        baseline_subset_correct = sum(int(row["baseline_correct"]) for row in eligible_rows)
        coverage = _safe_div(eligible, n)
        conditional = _safe_div(rationale_correct, eligible)
        pipeline = _safe_div(rationale_correct, n)
        baseline_all = _safe_div(baseline_correct, n)
        baseline_subset = _safe_div(baseline_subset_correct, eligible)
        bootstrap = bootstrap_cell(rows, replicates, seed)
        lineage = dict(collections.Counter(str(row["lineage_status"]) for row in rows))
        exact_echo = sum(int(row["answer_echo_exact"]) for row in eligible_rows)
        overlap_50 = sum(int(float(row["answer_token_recall"]) >= 0.5) for row in eligible_rows)
        overlap_80 = sum(int(float(row["answer_token_recall"]) >= 0.8) for row in eligible_rows)
        cell = {
            "judge": alias,
            "arm": arm,
            "setting": setting,
            "n": n,
            "eligible": eligible,
            "generation_coverage": coverage,
            "conditional_cmc": conditional,
            "pipeline_cmc": pipeline,
            "baseline_all": baseline_all,
            "baseline_exact_nonempty_subset": baseline_subset,
            "delta_pipeline": pipeline - baseline_all,
            "delta_conditional": conditional - baseline_subset,
            "pipeline_identity_residual": pipeline - coverage * conditional,
            "baseline_parse_failures": sum(int(row["baseline_parse_failed"]) for row in rows),
            "rationale_parse_failures": sum(int(row["rationale_parse_failed"]) for row in eligible_rows),
            "lineage": lineage,
            "answer_echo": {
                "eligible_n": eligible,
                "exact_n": exact_echo,
                "exact_rate": _safe_div(exact_echo, eligible),
                "token_recall_ge_0_5_n": overlap_50,
                "token_recall_ge_0_5_rate": _safe_div(overlap_50, eligible),
                "token_recall_ge_0_8_n": overlap_80,
                "token_recall_ge_0_8_rate": _safe_div(overlap_80, eligible),
            },
            "bootstrap": bootstrap,
            "delta_pipeline_p_raw": bootstrap["delta_pipeline_p"],
        }
        if abs(cell["pipeline_identity_residual"]) > 1e-12:
            raise RuntimeError(f"Pipeline identity failed for {alias}/{arm}/{setting}")
        cells.append(cell)

    if apply_bh:
        adjusted = benjamini_hochberg([float(cell["delta_pipeline_p_raw"]) for cell in cells])
        for cell, p_adjusted in zip(cells, adjusted):
            cell["delta_pipeline_p_bh"] = p_adjusted
            cell["delta_pipeline_reject_bh_0_05"] = bool(p_adjusted <= 0.05)

    summary = {"judge": alias, "primary_bh_family": apply_bh, "cells": cells}
    write_json(out_dir / f"summary_{alias}.json", summary)
    return summary


def write_markdown(summaries: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    lines = [
        "# Clean final-only CMC results",
        "",
        "All intervals use 10,000 source-frame cluster-bootstrap replicates. Ambiguous lineage, empty finals, and invalid judge outputs count as pipeline failures.",
        "",
    ]
    for summary in summaries:
        alias = summary["judge"]
        suffix = "; BH is applied only here" if summary["primary_bh_family"] else "; robustness estimates only"
        lines.extend(
            [
                f"## {alias}{suffix}",
                "",
                "| Arm | Setting | G | Conditional | Pipeline | Baseline all | Baseline subset | Delta pipeline | 95% CI | p raw | p BH |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for cell in summary["cells"]:
            ci = cell["bootstrap"]["ci"]["delta_pipeline"]
            p_bh = cell.get("delta_pipeline_p_bh")
            lines.append(
                "| {arm} | {setting} | {g:.3f} | {conditional:.3f} | {pipeline:.3f} | "
                "{base:.3f} | {subset:.3f} | {delta:+.3f} | [{low:+.3f}, {high:+.3f}] | "
                "{p:.4g} | {p_bh} |".format(
                    arm=cell["arm"],
                    setting=cell["setting"],
                    g=cell["generation_coverage"],
                    conditional=cell["conditional_cmc"],
                    pipeline=cell["pipeline_cmc"],
                    base=cell["baseline_all"],
                    subset=cell["baseline_exact_nonempty_subset"],
                    delta=cell["delta_pipeline"],
                    low=ci[0],
                    high=ci[1],
                    p=cell["delta_pipeline_p_raw"],
                    p_bh=(f"{p_bh:.4g}" if p_bh is not None else "—"),
                )
            )
        lines.append("")
    path = out_dir / "RESULTS.md"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def self_test() -> None:
    prompt_sentinel_test()
    base = {
        "rationale_gen": "A final sentence.",
        "rationale_final": "A final sentence.",
        "rationale_reasoning": "Earlier thought.",
    }
    assert lineage_status(base, base, True) == ("verified_writer_completion", True)
    original_final = {
        "rationale_gen": "<reasoning>why</reasoning><final>safe final</final>",
        "rationale_final": "<reasoning>why</reasoning><final>safe final</final>",
        "rationale_reasoning": "why",
    }
    current_final = {"rationale_final": "safe final"}
    assert lineage_status(current_final, original_final, False)[1]
    original_reasoning = {
        "rationale_gen": "<final>short</final>",
        "rationale_final": "<final>short</final>",
        "rationale_reasoning": "<reasoning>This is deliberately much longer.</reasoning><final>unsafe source</final>",
    }
    current_reasoning = {"rationale_final": "unsafe source"}
    assert lineage_status(current_reasoning, original_reasoning, False) == (
        "ambiguous_reparsed_from_reasoning",
        False,
    )
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == [0.03, 0.04, 0.04]
    print("cmc_final_only self-test: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--judges", nargs="*", choices=["qwen", "mistral"], default=[])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    manifest = read_json(args.manifest)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_rows, tasks = build_audit(manifest, out_dir)
    print(f"[CMC] audited {len(audit_rows)} cell-records; {len(tasks)} unique prompts", flush=True)
    if not args.judges:
        print("[CMC] audit-only run complete (no --judges requested)", flush=True)
        return 0

    model_key = {"qwen": "cmc_primary", "mistral": "cmc_robustness"}
    summaries = []
    for alias in args.judges:
        model_spec = manifest["models"][model_key[alias]]
        cache = score_prompts(
            alias=alias,
            model_id=model_spec["id"],
            revision=model_spec["revision"],
            tasks=tasks,
            out_dir=out_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        summaries.append(
            evaluate_judge(
                alias,
                audit_rows,
                cache,
                out_dir,
                args.bootstrap_replicates,
                args.bootstrap_seed,
                apply_bh=(alias == "qwen"),
            )
        )
    write_markdown(summaries, out_dir)
    write_json(
        out_dir / "RUN_METADATA.json",
        {
            "manifest_path": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "script_sha256": sha256_file(Path(__file__)),
            "judges": args.judges,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
        },
    )
    print(f"[CMC] complete: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
