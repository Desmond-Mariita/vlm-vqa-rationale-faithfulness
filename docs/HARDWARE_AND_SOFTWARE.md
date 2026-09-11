# Hardware and software

Training used an NVIDIA RTX 3090 with NF4 4-bit loading and bf16. Stage 1 used batch 2, accumulation 6, three epochs, AdamW learning rate 1e-4 and weight decay 0.01. Stage 2 used batch 2, accumulation 2, six epochs, learning rate 5e-5 and weight decay 0. Both used seed 42, cosine scheduling with 0.05 warmup, LoRA rank 32 / alpha 64 / dropout 0.05. Exact arm-specific settings and prompts are retained in reports/runs/**/config.resolved.yaml; path fields are sanitised and their original hashes are preserved.

Final Stage 2 generation used RTX 5090 hardware with batch 1 and seed-only determinism. Device UUIDs are not public reproducibility requirements. Recorded GPU software includes torch 2.9.0+cu128, Transformers 4.57.1, accelerate 1.11.0, bitsandbytes 0.48.2, PEFT 0.16.0 and NumPy 2.2.6. Judge evidence uses Transformers 5.13.0 and is a separate environment; do not merge it blindly into the training environment. Environment files record recovered direct versions, not invented complete transitive locks.

CPU validation has no torch, GPU, paid API or model dependency. The optional plot environment is recorded separately. Exact model-output reproduction depends on the external artifacts and runtime; CPU checks establish numeric table reconstruction and release integrity only.
