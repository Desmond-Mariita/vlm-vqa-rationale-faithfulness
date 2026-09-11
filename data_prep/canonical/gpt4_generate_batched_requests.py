
import os
import json
import base64
import argparse
from pathlib import Path
from tqdm import tqdm

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def main(args):
    with open(args.val, "r") as f:
        data = json.load(f)

    batch_size = args.batch_size
    total_written = 0
    skipped_images = []
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

    for batch_index, batch in enumerate(tqdm(batches, desc="Writing batches")):
        output_file = os.path.join(args.out_dir, f"batch_{batch_index:03d}.jsonl")
        with open(output_file, "w") as out_f:
            for i, item in enumerate(batch):
                image_id = item.get("image_id")
                if not image_id:
                    continue
                image_path = os.path.join(args.images, image_id)
                if not os.path.isfile(image_path):
                    skipped_images.append(image_path)
                    continue
                try:
                    base64_image = encode_image_to_base64(image_path)
                    payload = {
                        "custom_id": f"{image_id}__{item.get('question_id', i)}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-4o",
                            "temperature": 0.2,
                            "max_tokens": 120,
                            "messages": [
                                {"role": "system", "content": args.system_prompt},
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": args.prompt},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                    ]
                                }
                            ]
                        }
                    }
                    out_f.write(json.dumps(payload) + "\n")
                    total_written += 1
                except Exception as e:
                    skipped_images.append(image_path)

    # Save skipped images
    if skipped_images:
        skipped_log = os.path.join(args.out_dir, "skipped_images.txt")
        with open(skipped_log, "w") as f:
            for path in skipped_images:
                f.write(path + "\n")

    print(f"✅ Wrote {total_written} total requests across {len(batches)} batch files.")
    print(f"⚠️ Skipped {len(skipped_images)} images. See skipped_images.txt if needed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", required=True, help="Input JSON file")
    parser.add_argument("--images", required=True, help="Directory containing image files")
    parser.add_argument("--out_dir", required=True, help="Output directory for batch JSONL files")
    parser.add_argument("--batch_size", type=int, default=250, help="Number of requests per batch")
    parser.add_argument("--prompt", default="Describe this scene based on labeled elements (e.g., person_0, object_1). Focus on positions, actions, and interactions without interpretation. Be concise and objective.")
    parser.add_argument("--system_prompt", default="You are a vision-language model tasked with generating factual, concise scene descriptions based on labeled image elements such as person_0 or object_1. Avoid speculation and emotional language.")
    args = parser.parse_args()
    main(args)
