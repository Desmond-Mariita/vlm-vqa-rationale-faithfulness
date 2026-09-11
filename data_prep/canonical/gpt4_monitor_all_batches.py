import os
import json
import time
import openai
import argparse
from tqdm import tqdm

def monitor_batches(batch_log_path, output_dir, poll_interval):
    print(f"📖 Reading batch log: {batch_log_path}")
    print(f"📂 Output directory: {output_dir}")
    print(f"⏲️ Poll interval: {poll_interval} seconds")

    os.makedirs(output_dir, exist_ok=True)

    with open(batch_log_path, "r") as f:
        batch_records = [json.loads(line) for line in f if line.strip()]
    print(f"🔍 Found {len(batch_records)} batch records to monitor")

    for record in tqdm(batch_records, desc="Monitoring batches"):
        batch_id = record.get("batch_id")
        if not batch_id:
            print("⚠️ Skipping record without batch_id")
            continue

        output_path = os.path.join(output_dir, f"{batch_id}.jsonl")
        if os.path.exists(output_path):
            print(f"✅ Already downloaded: {output_path}")
            continue

        try:
            batch = openai.batches.retrieve(batch_id)
            status = batch.status
            print(f"⏳ Batch {batch_id} status: {status}")

            if status == "completed":
                result_file_id = batch.output_file_id
                if result_file_id:
                    result = openai.files.content(result_file_id)
                    with open(output_path, "wb") as out_f:
                        out_f.write(result.read())
                    print(f"✅ Downloaded: {output_path}")
                else:
                    print(f"⚠️ Batch {batch_id} completed but no output_file_id found.")
            elif status in ("failed", "expired", "cancelled"):
                print(f"❌ Batch {batch_id} ended with status: {status}")
            else:
                print(f"⏳ Waiting on {batch_id} (status: {status})...")

        except Exception as e:
            print(f"❌ Error checking batch {batch_id}: {e}")

        time.sleep(poll_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_log_path", required=True, help="Path to the batch log JSONL file")
    parser.add_argument("--output_dir", required=True, help="Directory to store output .jsonl files")
    parser.add_argument("--poll_interval", type=int, default=5, help="Polling interval in seconds")
    args = parser.parse_args()

    monitor_batches(args.batch_log_path, args.output_dir, args.poll_interval)
