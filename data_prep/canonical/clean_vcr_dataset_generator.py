import json
import os
import shutil
import uuid
from pathlib import Path
from tqdm import tqdm
import hashlib

def create_clean_vcr_dataset(
    jsonl_path,
    images_base_dir,
    output_dir,
    split_name,
    output_json_name=None
):
    """
    Creates a clean VCR dataset with verified images and standardized JSON format.

    Args:
        jsonl_path: Path to the input JSONL file
        images_base_dir: Base directory containing the VCR images with subdirectories
        output_dir: Directory to save cleaned dataset
        split_name: Name of the split ("train", "test", or "val")
        output_json_name: Name for the output JSON file (optional)
    """

    # Validate split name
    valid_splits = {"train", "test", "val"}
    if split_name not in valid_splits:
        raise ValueError(f"split_name must be one of {valid_splits}, got '{split_name}'")

    # Create output directories
    output_path = Path(output_dir)
    output_images_dir = output_path / "images"
    output_path.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Set output JSON filename
    if output_json_name is None:
        output_json_name = f"clean_{split_name}.json"

    # Statistics tracking
    stats = {
        'total_entries': 0,
        'valid_entries': 0,
        'missing_images': 0,
        'copied_images': 0,
        'duplicate_images': 0,
        'json_errors': 0
    }

    missing_images = []
    clean_dataset = []
    used_question_ids = set()
    copied_image_hashes = {}  # Track duplicates by content hash

    print(f"🔍 Reading JSONL file: {jsonl_path}")
    print(f"📂 Source images directory: {images_base_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🏷️  Split name: {split_name}")

    # Read and process JSONL file
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            entries = [json.loads(line.strip()) for line in f if line.strip()]
    except Exception as e:
        print(f"❌ Error reading JSONL file: {e}")
        return None

    stats['total_entries'] = len(entries)
    print(f"📊 Found {stats['total_entries']} entries in JSONL file")

    print("🔍 Processing entries and verifying images...")

    for entry_idx, entry in enumerate(tqdm(entries, desc="Processing entries")):
        try:
            # Extract required fields
            img_fn = entry.get('img_fn', '')
            question = entry.get('question', '')
            choices = entry.get('answer_choices', [])
            correct_idx = entry.get('answer_label', 3) # All test split answers will be 3
            rationales = entry.get('rationale_choices', ['test'])

            # Validate required fields
            if not img_fn:
                print(f"⚠️  Entry {entry_idx} missing 'img_fn' field")
                continue

            if not question:
                print(f"⚠️  Entry {entry_idx} missing 'question' field")
                continue

            if not choices or len(choices) == 0:
                print(f"⚠️  Entry {entry_idx} missing or empty 'answer_choices'")
                continue

            if correct_idx is None or correct_idx < 0 or correct_idx >= len(choices):
                if split_name != "test":  # Allow None for test split
                    print(f"⚠️  Entry {entry_idx} has invalid 'answer_label': {correct_idx}")
                    continue

            # Construct original image path
            original_img_path = Path(images_base_dir) / img_fn

            # Check if original image exists
            if not original_img_path.exists():
                if entry_idx < 5:
                    print(f"DEBUG: Looking for image at: {original_img_path}")
                missing_images.append(img_fn)
                stats['missing_images'] += 1
                continue

            # Generate unique image filename (sanitized)
            # Extract just the filename from the path and sanitize it
            original_filename = Path(img_fn).name
            sanitized_filename = original_filename.replace('/', '_').replace('\\', '_')

            # Handle potential filename conflicts by adding hash prefix
            file_content_hash = get_file_hash(original_img_path)
            hash_prefix = file_content_hash[:8]
            unique_filename = f"{hash_prefix}_{sanitized_filename}"

            # Check for duplicate images by content hash
            if file_content_hash in copied_image_hashes:
                stats['duplicate_images'] += 1
                # Use the existing filename for this duplicate
                unique_filename = copied_image_hashes[file_content_hash]
            else:
                # Copy the image with unique filename
                output_img_path = output_images_dir / unique_filename

                try:
                    shutil.copy2(original_img_path, output_img_path)
                    copied_image_hashes[file_content_hash] = unique_filename
                    stats['copied_images'] += 1
                except Exception as e:
                    print(f"❌ Error copying {img_fn}: {e}")
                    continue

            # Generate unique question ID
            question_id = generate_unique_question_id(used_question_ids)
            used_question_ids.add(question_id)

            # Get correct rationale (if available).
            # NB: VCR's rationale_label is independent of answer_label, so we
            # index `rationale_choices` by `rationale_label` to recover the
            # canonical correct rationale. (Earlier versions of this script
            # indexed by answer_label, which only matched in ~25% of records;
            # see data_prep/canonical/patch_rationales_to_label.py for the
            # one-shot fixer that retro-applies this correction to existing
            # cleaned JSONs without a full re-clean.)
            rationale_idx = entry.get('rationale_label', correct_idx)
            correct_rationale = ""
            if rationales and 0 <= rationale_idx < len(rationales):
                correct_rationale = rationales[rationale_idx]

            # Create clean entry
            clean_entry = {
                "split": split_name,
                "image_id": unique_filename,
                "question_id": question_id,
                "question": question,
                "choices": choices,
                "correct_choice_idx": correct_idx,
                "rationales": correct_rationale
            }

            clean_dataset.append(clean_entry)
            stats['valid_entries'] += 1

        except Exception as e:
            print(f"❌ Error processing entry {entry_idx}: {e}")
            stats['json_errors'] += 1
            continue

    # Save clean dataset as JSON
    output_json_path = output_path / output_json_name
    print(f"💾 Saving clean dataset to {output_json_path}")

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_dataset, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Error saving JSON file: {e}")
        return None

    # Save missing images list
    if missing_images:
        missing_file_path = output_path / "missing_images.txt"
        with open(missing_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Missing Images for {split_name} split:\n")
            f.write("=" * 50 + "\n")
            for img in missing_images:
                f.write(f"{img}\n")
        print(f"📝 Missing images list saved to {missing_file_path}")

    # Save processing summary
    summary_path = output_path / f"{split_name}_processing_summary.txt"
    save_processing_summary(summary_path, stats, split_name, jsonl_path, images_base_dir)

    # Print summary
    print("\n" + "="*70)
    print(f"🎉 CLEAN {split_name.upper()} DATASET CREATION COMPLETE!")
    print("="*70)
    print(f"📊 Total entries processed: {stats['total_entries']}")
    print(f"✅ Valid entries (with existing images): {stats['valid_entries']}")
    print(f"❌ Missing images: {stats['missing_images']}")
    print(f"📁 Unique images copied: {stats['copied_images']}")
    print(f"🔄 Duplicate images detected: {stats['duplicate_images']}")
    print(f"⚠️  JSON processing errors: {stats['json_errors']}")
    print(f"📈 Success rate: {(stats['valid_entries']/stats['total_entries']*100):.1f}%")
    print(f"📂 Output directory: {output_path.absolute()}")
    print(f"🖼️  Images directory: {output_images_dir.absolute()}")
    print(f"📄 Clean dataset JSON: {output_json_path.absolute()}")

    return {
        'stats': stats,
        'output_json_path': str(output_json_path),
        'output_images_dir': str(output_images_dir),
        'clean_dataset': clean_dataset
    }

def get_file_hash(file_path):
    """Generate MD5 hash of file content for duplicate detection."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return hashlib.md5(str(file_path).encode()).hexdigest()

def generate_unique_question_id(used_ids):
    """Generate a unique question ID."""
    while True:
        question_id = str(uuid.uuid4())[:8]  # Short UUID
        if question_id not in used_ids:
            return question_id

def save_processing_summary(summary_path, stats, split_name, jsonl_path, images_base_dir):
    """Save detailed processing summary."""
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"VCR DATASET CLEANING SUMMARY - {split_name.upper()} SPLIT\n")
        f.write("=" * 60 + "\n\n")

        f.write("INPUT CONFIGURATION:\n")
        f.write(f"  JSONL file: {jsonl_path}\n")
        f.write(f"  Images directory: {images_base_dir}\n")
        f.write(f"  Split name: {split_name}\n\n")

        f.write("PROCESSING STATISTICS:\n")
        for key, value in stats.items():
            f.write(f"  {key}: {value}\n")

        f.write(f"\nSUCCESS RATE: {(stats['valid_entries']/stats['total_entries']*100):.2f}%\n")

def process_multiple_splits(splits_config, images_base_dir, output_base_dir):
    """
    Process multiple VCR splits in batch.

    Args:
        splits_config: Dict with split_name -> jsonl_path mapping
        images_base_dir: Base directory containing images
        output_base_dir: Base output directory
    """
    all_results = {}

    for split_name, jsonl_path in splits_config.items():
        print(f"\n{'='*80}")
        print(f"PROCESSING {split_name.upper()} SPLIT")
        print(f"{'='*80}")

        split_output_dir = Path(output_base_dir) / split_name

        result = create_clean_vcr_dataset(
            jsonl_path=jsonl_path,
            images_base_dir=images_base_dir,
            output_dir=str(split_output_dir),
            split_name=split_name
        )

        all_results[split_name] = result

    return all_results

# At the bottom of your script, replace the example usage with:

if __name__ == "__main__":
    # Configuration for all splits
    base_images_dir = os.path.join(os.getcwd(), "data", "raw", "vcr1images")
    base_output_dir = os.path.join(os.getcwd(), "data", "processed", "vcr")
    
    # Define all splits with their JSONL file paths
    splits_config = {
        "train": os.path.join(os.getcwd(), "data","raw", "vcr1annots", "train.jsonl"), 
        "val": os.path.join(os.getcwd(), "data","raw", "vcr1annots", "val.jsonl"),       
        "test": os.path.join(os.getcwd(), "data","raw", "vcr1annots", "test.jsonl")    
    }
    
    print("🚀 PROCESSING ALL VCR SPLITS")
    print("=" * 80)
    
    # Process all splits
    all_results = process_multiple_splits(
        splits_config=splits_config,
        images_base_dir=base_images_dir,
        output_base_dir=base_output_dir
    )
    
    # Print overall summary
    print("\n" + "="*80)
    print("🎉 ALL SPLITS PROCESSING COMPLETE!")
    print("="*80)
    
    total_entries = 0
    total_valid = 0
    total_missing = 0
    
    for split_name, result in all_results.items():
        if result and 'stats' in result:
            stats = result['stats']
            print(f"\n📊 {split_name.upper()} SUMMARY:")
            print(f"   Processed: {stats['total_entries']}")
            print(f"   Valid: {stats['valid_entries']}")
            print(f"   Missing: {stats['missing_images']}")
            print(f"   Success: {(stats['valid_entries']/stats['total_entries']*100):.1f}%")
            
            total_entries += stats['total_entries']
            total_valid += stats['valid_entries']
            total_missing += stats['missing_images']
    
    print("\n🌟 OVERALL TOTALS:")
    print(f"   Total entries: {total_entries}")
    print(f"   Total valid: {total_valid}")
    print(f"   Total missing: {total_missing}")
    print(f"   Overall success: {(total_valid/total_entries*100):.1f}%")
