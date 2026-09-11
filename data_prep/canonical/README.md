# Canonical preparation code

These selected historical scripts prepare plain/polygon VCR inputs and frozen descriptions. Inputs and all generated outputs stay external and ignored. Read docs/DATA_ACCESS.md first. There is no broad dataset downloader or paper/model payload in this release.

Order: clean_vcr_dataset_generator.py creates plain inputs; vcr_dataset_clean4.py creates question-specific polygon renders; gpt4_split_and_merge.py and gpt4_generate_batched_requests.py construct batch requests; gpt4_submit_batches_openai.py and gpt4_monitor_all_batches.py submit/poll the external API; gpt4_incorporate_captions.py merges captions; patch_rationales_to_label.py aligns selected human rationales; data/subset_balanced_json.py constructs seeded balanced subsets. API scripts require locally set OPENAI_API_KEY and are never run by CI. Fresh API outputs are not guaranteed to match frozen descriptions.

Historical mains retain their relative data layout; run from the repository root and inspect each script's parameters before external preparation. The polygon script defaults to data/labeled while the frozen experiment consumed the corresponding data/final layout. Reconstruct and verify that layout locally; do not claim the relocation command was recovered. No corpus generation is performed by release packaging.
