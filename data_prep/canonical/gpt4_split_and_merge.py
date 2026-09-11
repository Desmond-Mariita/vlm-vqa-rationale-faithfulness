import os
import json
from pathlib import Path
from typing import List

def split_json(input_json_path: str, output_dir: str, batch_size: int = 21300) -> List[str]:
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    output_paths = []

    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        chunk_path = os.path.join(output_dir, f"train_part_{i // batch_size:03d}.json")
        with open(chunk_path, "w", encoding="utf-8") as f_out:
            json.dump(chunk, f_out, indent=2)
        output_paths.append(chunk_path)

    return output_paths


def merge_json(input_dir: str, output_path: str) -> None:
    merged_data = []
    for file in sorted(Path(input_dir).glob("train_part_*.json")):
        with open(file, "r", encoding="utf-8") as f:
            merged_data.extend(json.load(f))

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(merged_data, f_out, indent=2)

    print(f"✅ Merged {len(merged_data)} entries into {output_path}")


# Example usage
if __name__ == "__main__":
    input_train_jsonl = "data/labeled/train/clean_train.json"
    output_batch_dir = "data/labeled/train/train_batches/final_captioned"
    final_merged_output = "data/labeled/train/vcr_train_captioned.json"

    # Split
    #split_json(input_train_jsonl, output_batch_dir, batch_size=21300)

    # Merge (optional)
    merge_json(output_batch_dir, final_merged_output)
