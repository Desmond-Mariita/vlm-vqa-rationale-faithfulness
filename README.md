# Evaluating the Faithfulness of Chain-of-Thought Rationales in Vision-Language Models for Visual Question Answering

## Thesis summary

In this thesis, I study whether natural-language rationales from a vision-language model respond to the visual evidence that affects its answers. I use Qwen2.5-VL-3B-Instruct on Visual Commonsense Reasoning (VCR) in a two-stage pipeline: Stage 1 predicts one of four answers, and Stage 2 generates a rationale for a supplied answer. I compare four input configurations—`plain`, `point`, `plain_desc` and `point_desc`—under six image conditions: `source`, `grey`, `mismatch`, `mask`, `noise` and `mirror`.

Both answer predictions and rationale text respond systematically to changes in visual evidence. Configurations with descriptions are more robust when images are degraded, while polygon overlays show no consistent independent advantage. Changing the supplied answer also changes the rationale text. I interpret these patterns as behavioural signals relevant to faithfulness. They describe how outputs change under controlled interventions; they do not reveal the model's hidden causal computation or establish whether an individual rationale is faithful or unfaithful. Greater robustness with descriptions also does not establish stronger visual grounding.

## Experimental setup

Stage 1 selects an answer from four choices. Stage 2 writes a rationale conditioned on a supplied answer. In my primary Stage 2 analysis, I keep the gold answer fixed while changing the image condition. I report **PRED-A** separately as a secondary analysis using historical Stage 1 predictions.

Each input arm has its own trained adaptation:

| Arm | Image | Fixed textual description |
|---|---|---|
| `plain` | Plain frame | No |
| `point` | Frame with polygon overlay | No |
| `plain_desc` | Plain frame | Yes |
| `point_desc` | Frame with polygon overlay | Yes |

The six image conditions are applied within these four arms.

## What is in this repository

This repository contains the code and supporting material for my thesis: training and evaluation code, YAML configurations, scripts for generating image interventions and analysing outputs, and parsers and evaluation utilities. It also includes frozen numeric evidence, scripts to reproduce the reported tables and numeric figures, selected thesis source, and CPU release tests.

## Reproducing the reported tables

Use Python 3.12 or later and run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

make check
python scripts/build_public_tables.py --output-dir reports/generated/tables --check
```

These checks reproduce tables from the included numeric evidence and compare all 12 active numeric table sources byte-for-byte with the thesis versions. No GPU, model download or VCR download is required.

With Matplotlib and the required system LaTeX packages installed, reproduce and check the three numeric result plots with:

```bash
make figures
```

## Data and visual assets

VCR data and imagery are not redistributed here. Please obtain VCR separately under its own terms. Reproducing the thesis figures that use VCR images requires reconstructing their visual assets locally from separately acquired material; see [Data access](docs/DATA_ACCESS.md) for the requirements.

The University logo, my signature and the final submission PDF are not included in this public repository.

## Full experiments

Running the full experiments requires separately acquired VCR data, the recorded subsets and manifests, frozen descriptions, retained checkpoints, pinned model revisions, and suitable NVIDIA hardware. I describe the commands and external dependencies in [Reproducibility](docs/REPRODUCIBILITY.md).

The descriptions were generated through a GPT-4o API alias that was not pinned to a dated model snapshot, so generating them again is not guaranteed to reproduce the same text.

## Thesis and citation

Desmond Mariita (2026). *Evaluating the Faithfulness of Chain-of-Thought Rationales in Vision-Language Models for Visual Question Answering*. Master's thesis, University of Potsdam.

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## Repository mirrors

These repositories are mirrors of the same publication history:

- [GitHub](https://github.com/Desmond-Mariita/vlm-vqa-rationale-faithfulness)
- [University of Potsdam GitUP](https://gitup.uni-potsdam.de/mariita/vlm-vqa-rationale-faithfulness)

## Licence and third-party material

I have not yet assigned a general open-source licence to the code or thesis material I own. Public visibility does not imply blanket permission to reuse it. Third-party datasets, models and software remain subject to their own terms. See [NOTICE](NOTICE) and [LICENSE_PENDING.md](LICENSE_PENDING.md).
