
import os
import json
import argparse
from tqdm import tqdm

def load_caption_responses(results_dir):
    captions = {}
    for file in os.listdir(results_dir):
        if not file.endswith(".jsonl"):
            continue
        with open(os.path.join(results_dir, file), "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    custom_id = item.get("custom_id")
                    if not custom_id:
                        continue
                    response = item.get("response", {}).get("body", {})
                    caption = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if caption:
                        captions[custom_id] = caption.strip()
                except Exception as e:
                    print(f"⚠️ Error parsing line in {file}: {e}")
    return captions

def main(args):
    print("📥 Loading GPT captions...")
    captions = load_caption_responses(args.results_dir)
    print(f"✅ Loaded {len(captions)} captions")

    print("📥 Loading original VCR-style file...")
    with open(args.original_json, "r") as f:
        data = json.load(f)

    updated = 0
    for item in tqdm(data, desc="Injecting captions"):
        image_id = item.get("image_id")
        question_id = item.get("question_id")
        key = f"{image_id}__{question_id}"
        if key in captions:
            item["gpt_caption"] = captions[key]
            updated += 1

    print(f"✅ Added captions to {updated} items")

    print(f"💾 Writing merged file to {args.output_json}")
    with open(args.output_json, "w") as out_f:
        json.dump(data, out_f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="Directory with OpenAI batch .jsonl files")
    parser.add_argument("--original_json", required=True, help="Path to the original VCR-style JSON file")
    parser.add_argument("--output_json", required=True, help="Where to save the merged output")
    args = parser.parse_args()
    main(args)
