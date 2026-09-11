import json
import os
import uuid
import re
import cv2
import hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def deep_flatten(x):
    if isinstance(x, str):
        return x
    elif isinstance(x, list):
        return " ".join(deep_flatten(i) for i in x)
    return str(x)


def replace_refs_with_labels(text, names):
    text = deep_flatten(text)
    def repl(match):
        idx = int(match.group(1))
        if 0 <= idx < len(names):
            return f"{names[idx]}_{idx}"
        return match.group(0)
    return re.sub(r'(?<!\w)(\d+)(?!\w)', repl, text)


def draw_referenced_objects_on_image(image, text, data, draw_labels=True, font_color=(255, 0, 255)):
    referenced_indices = set()

    # Match references like person_0, bottle_2
    matches = re.findall(r'(\w+)_([0-9]+)', text)
    for label, idx in matches:
        idx = int(idx)
        if 0 <= idx < len(data["names"]) and data["names"][idx] == label:
            referenced_indices.add(idx)

    for i in referenced_indices:
        if i >= len(data["boxes"]):
            continue

        x1, y1, x2, y2, _ = data["boxes"][i]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if draw_labels:
            label_text = f"{data['names'][i]}_{i}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            text_size, _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            text_width, text_height = text_size

            # Calculate center position
            center_x = x1 + (x2 - x1) // 2 - text_width // 2
            center_y = y1 + (y2 - y1) // 2 + text_height // 2
            center_x = max(0, center_x)
            center_y = max(text_height, center_y)

            cv2.putText(image, label_text, (center_x, center_y),
                        font, font_scale, font_color, thickness, lineType=cv2.LINE_AA)

        # Draw polygon if available
        if "segms" in data and i < len(data["segms"]):
            for polygon in data["segms"][i]:
                pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 255), thickness=2)

    return image

def get_file_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def process_entry(args):
    entry_idx, entry, images_base_dir, output_images_dir, split_name = args
    stats = defaultdict(int)
    result_entry = None

    try:
        img_fn = entry.get('img_fn', '')
        correct_idx = entry.get('answer_label', 0)
        rationale_choices = entry.get('rationale_choices', [])

        image_path = Path(images_base_dir) / img_fn
        if not image_path.exists():
            stats['missing_images'] += 1
            return stats, None

        json_path = image_path.with_suffix('.json')
        if not json_path.exists():
            stats['json_errors'] += 1
            return stats, None

        with open(json_path) as jf:
            json_data = json.load(jf)
        names = json_data.get("names", [])

        question = replace_refs_with_labels(entry.get("question", ""), names)
        choices = [replace_refs_with_labels(c, names) for c in entry.get("answer_choices", [])]
        # NB: VCR's rationale_label is independent of answer_label; index
        # rationale_choices by rationale_label to recover the canonical
        # correct rationale (earlier versions used answer_label, which only
        # matched in ~25% of records).
        rationale_idx = entry.get('rationale_label', correct_idx)
        rationale = ""
        if isinstance(rationale_choices, list) and 0 <= rationale_idx < len(rationale_choices):
            rationale = replace_refs_with_labels(rationale_choices[rationale_idx], names)

        # Generate hash-based image ID
        file_hash = get_file_hash(image_path)
        original_filename = image_path.name.replace(".png", ".jpg")
        question_id = str(uuid.uuid4())[:8]
        image_id = f"{file_hash[:8]}_{question_id}_{original_filename}"
        output_img_path = output_images_dir / image_id

        image = cv2.imread(str(image_path))
        if image is None:
            stats['image_load_errors'] += 1
            return stats, None

        annotated = draw_referenced_objects_on_image(
            image, question + " " + " ".join(choices) + " " + rationale, json_data
        )

        success = cv2.imwrite(str(output_img_path), annotated)
        if not success:
            stats['image_write_errors'] += 1
            return stats, None

        result_entry = {
            "split": split_name,
            "image_id": image_id,
            "question_id": str(uuid.uuid4())[:8],
            "question": question,
            "choices": choices,
            "correct_choice_idx": correct_idx,
            "rationales": rationale
        }

        stats['valid_entries'] += 1
        stats['copied_images'] += 1

    except Exception as e:
        stats['json_errors'] += 1
        print(f"❌ Error processing entry {entry_idx}: {e}")

    return stats, result_entry


def create_clean_vcr_dataset_parallel(jsonl_path, images_base_dir, output_dir, split_name):
    output_path = Path(output_dir) / split_name
    output_images_dir = output_path / "images"
    output_path.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        entries = [json.loads(line.strip()) for line in f if line.strip()]

    clean_dataset = []
    stats_total = defaultdict(int)
    args_list = [
        (idx, entry, images_base_dir, output_images_dir, split_name)
        for idx, entry in enumerate(entries)
    ]

    with Pool(processes=cpu_count()) as pool:
        for stats, result in tqdm(pool.imap_unordered(process_entry, args_list), total=len(entries)):
            for k, v in stats.items():
                stats_total[k] += v
            if result:
                clean_dataset.append(result)

    stats_total['total_entries'] = len(entries)

    # Save cleaned dataset
    out_json = output_path / f"clean_{split_name}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(clean_dataset, f, indent=2)

    # Save summary
    summary_txt = output_path / f"summary_{split_name}.txt"
    lines = []
    lines.append(f"VCR DATASET CLEANING SUMMARY - {split_name.upper()} SPLIT")
    lines.append("=" * 60)
    lines.append("\nINPUT CONFIGURATION:")
    lines.append(f"  JSONL file: {jsonl_path}")
    lines.append(f"  Images directory: {images_base_dir}")
    lines.append(f"  Split name: {split_name}\n")
    lines.append("PROCESSING STATISTICS:")
    for key in ['total_entries', 'valid_entries', 'missing_images', 'copied_images',
                'json_errors', 'image_load_errors', 'image_write_errors']:
        lines.append(f"  {key}: {stats_total[key]}")
    success_rate = (stats_total['valid_entries'] / stats_total['total_entries']) * 100 if stats_total['total_entries'] else 0
    lines.append(f"\nSUCCESS RATE: {success_rate:.2f}%\n")
    print("\n" + "\n".join(lines))
    with open(summary_txt, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    base_images_dir = os.path.join(os.getcwd(), "data", "raw", "vcr1images")
    base_output_dir = os.path.join(os.getcwd(), "data", "labeled")

    splits_config = {
        "train": os.path.join(os.getcwd(), "data", "raw", "vcr1annots", "train.jsonl"),
        "val": os.path.join(os.getcwd(), "data", "raw", "vcr1annots", "val.jsonl"),
        "test": os.path.join(os.getcwd(), "data", "raw", "vcr1annots", "test.jsonl")
    }

    for split, jsonl_path in splits_config.items():
        print(f"🔍 Reading JSONL file: {jsonl_path}")
        print(f"📂 Source images directory: {base_images_dir}")
        print(f"📁 Output directory: {base_output_dir}/{split}/images")
        print(f"🏷️  Split name: {split}")
        create_clean_vcr_dataset_parallel(jsonl_path, base_images_dir, base_output_dir, split)
