# Model card

Research pipeline: Stage 1 predicts one of four VQA choices; Stage 2 generates reasoning/final rationale spans. Four input arms and matched image interventions probe answer sensitivity and rationale wording. Backbone: Qwen2.5-VL-3B-Instruct at the revision in MODEL_AND_REVISION_MANIFEST.md. Retained artifacts are external; no weights are supplied.

Training uses the documented VCR subsets, LoRA and separate answer/rationale stages. Primary Stage 2 receives gold answers and retains epoch 6 of a fixed budget. PRED-A is secondary. Loss covers all non-padding prompt and target tokens in the original Stage 2 implementation; the target split is a formatting heuristic over a human rationale, not separately annotated reasoning.

Intended use is research reproduction with appropriately licensed inputs. Results do not prove internal causal faithfulness or establish reliability for deployment. VCR-derived biases, fixed descriptions, parser recovery and external model/judge limitations apply. Do not infer redistribution rights for adapters from availability of source code.
