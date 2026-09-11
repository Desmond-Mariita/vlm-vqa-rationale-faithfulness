# Model and revision manifest

| Role | Identifier | Frozen revision |
|---|---|---|
| Stage 1 / Stage 2 backbone and processor | Qwen/Qwen2.5-VL-3B-Instruct | 66285546d2b821cf421d4f5eb2576359d3770cd3 |
| BERTScore | roberta-large | 722cf37b1afa9454edce342e7895e588b6ff1d59 |
| CMC text-only judge | Qwen/Qwen2.5-3B-Instruct | aa8e72537993ba99e69dfaafa59ed015b17504d1 |
| Frozen description generation | gpt-4o | mutable API alias; no immutable revision recovered |

No weights or adapters are included. Eight retained checkpoints and eight transport artifacts are represented by external stubs and original hashes in data/manifests/external_sources.json. Base Qwen downloads do not provide the thesis-trained adapters. Stage 1 retained epoch 2; Stage 2 retained epoch 6 after a fixed six-epoch budget. PRED-A's supplied answers came from historical Stage 1 epoch 3.

BERTScore 0.3.12 uses layer 17, slow RobertaTokenizer, no IDF and no baseline rescaling for the primary measure. The CMC judge sees only the final rationale span, with no image, arm identity or supplied answer, input capped at 2,048 tokens and output constrained to one digit 0–3. This is secondary PRED-A evaluation, not primary RQ2 Drift.

Consult each upstream model's licence at the stated revision before acquisition or redistribution. Qwen 3B licences are separate from any future author code licence; adapter publication is not authorised by their presence in this manifest.
