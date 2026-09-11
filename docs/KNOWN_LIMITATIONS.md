# Known limitations

The exact approved PDF and two title/declaration assets are withheld. The active source is preserved verbatim and cannot yet reproduce those pages. No substitute PDF is generated.

Numeric Level B reproduces displayed tables and result plots from frozen outputs, not experiments. Public per-record files replace record/frame IDs with shared integer codes and therefore have new hashes. Original hash-bound scoring/inference verifiers need external originals. Some historical analysis/build scripts also need licensed qualitative records, frozen rationales and source manifests. These are not part of the CPU path.

Four arms and six image conditions are fixed. BERTScore Drift measures rationale-text sensitivity and does not establish causal faithfulness. Primary Stage 2 conditions on gold answers. PRED-A is separate, secondary and uses a different historical Stage 1 prediction lineage. Interim and final Stage 2 BLEU populations were different; they are not a comparable checkpoint-selection curve.

Exact GPT-4o caption regeneration cannot be guaranteed. No checkpoint or adapter download is promised. Full GPU evaluation/retraining and Docker image build were not exercised by local CPU validation. Environment pin files capture recovered direct versions rather than fully resolved transitive locks. The transparent secret-pattern scanner is bounded and cannot prove the absence of every possible secret format.
