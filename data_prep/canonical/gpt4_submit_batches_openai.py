
import os
import json
from pathlib import Path
import openai
from tqdm import tqdm
import argparse

def submit_batches(batch_dir, log_file, completion_window):
    Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)
    submitted = 0
    errors = []

    for batch_file in tqdm(sorted(Path(batch_dir).glob("batch_*.jsonl")), desc="Submitting batches"):
        try:
            # Upload the file first
            with open(batch_file, "rb") as f:
                uploaded = openai.files.create(file=f, purpose="batch")

            # Submit the batch using the uploaded file ID
            batch = openai.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window=completion_window
            )

            result = {
                "batch_file": str(batch_file),
                "file_id": uploaded.id,
                "batch_id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at
            }
            with open(log_file, "a") as logf:
                logf.write(json.dumps(result) + "\n")
            print(f"✅ Submitted {batch_file.name} | Batch ID: {batch.id}")
            submitted += 1

        except Exception as e:
            errors.append((str(batch_file), str(e)))
            print(f"❌ Failed: {batch_file.name} — {e}")

    print(f"✅ Done! {submitted} batches submitted.")
    if errors:
        print(f"⚠️ {len(errors)} batches failed. See console for error messages.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", required=True, help="Directory containing batch_*.jsonl files")
    parser.add_argument("--log_file", required=True, help="File to store submitted batch logs")
    parser.add_argument("--completion_window", default="24h", choices=["24h", "48h"], help="Batch completion window")
    args = parser.parse_args()

    submit_batches(args.batch_dir, args.log_file, args.completion_window)
