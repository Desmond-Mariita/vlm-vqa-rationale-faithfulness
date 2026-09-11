# Frozen execution lineage

Stage 1 predicts an answer. Primary Stage 2 generates a rationale with the gold answer fixed. The four arms are plain, point, plain_desc and point_desc; image conditions are source, grey, mismatch, mask, noise and mirror. PRED-A is secondary.

The original hash-bound manifests are external. Their documentation stubs do not satisfy runner input requirements. Runners retain fixed dataset/manifest/checkpoint/prompt identities and durable resume checks. Public invocation replaces private transfer/approval mechanics with explicit local asset validation. No invocation downloads data or models.

Use configs/evaluation/final.yaml for final scoring settings, not historical train-time defaults. The primary generation budget is 256 tokens; training-time validation used 128. Final inference retained seed-only determinism. Exact future GPU byte identity is not promised.
