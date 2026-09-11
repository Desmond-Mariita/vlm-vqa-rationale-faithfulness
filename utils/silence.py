"""Suppress noisy warnings / info messages emitted during training and eval.

Import this module FIRST in any script that calls torch/transformers/TF, e.g.:

    from utils import silence  # noqa: F401

Sets a handful of environment variables that must be in place before the
third-party libraries are imported (TF CPU feature logging, HF tokenizer
parallelism, the renamed PYTORCH_ALLOC_CONF), and installs targeted
``warnings.filterwarnings`` rules for the specific library warnings that spam
training logs without changing behavior. Suppressions are message-scoped so
real warnings still surface.
"""

from __future__ import annotations

import os
import warnings

# ---- env vars (must be set before the corresponding libs are imported) ----
# Silence TensorFlow CPU-feature info log lines from BLEURT.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Quiet HF fast-tokenizer fork warnings.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# torch 2.9 renamed PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF. If the
# caller set the old one, mirror it into the new one so torch stops warning.
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ and "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]

# ---- warnings.filterwarnings rules (message-scoped) ----
_FILTERS = [
    # transformers deprecation chatter
    (".*AutoModelForVision2Seq.*deprecated.*", FutureWarning),
    (".*`torch_dtype`.*deprecated.*", FutureWarning),
    (".*`torch_dtype`.*deprecated.*", UserWarning),
    # bitsandbytes informational cast notice
    (".*MatMul8bitLt.*will be cast.*", UserWarning),
    # gradient checkpointing on a no-grad eval pass
    (".*None of the inputs have requires_grad=True.*", UserWarning),
    # torch 2.9 TF32 API rename
    (".*torch\\.backends.*fp32_precision.*", UserWarning),
    (".*allow_tf32.*deprecated.*", UserWarning),
    # matplotlib CJK / full-width punctuation in VCR captions
    (".*missing from font.*", UserWarning),
    # peft/accelerate misc chatter that's purely informational
    (".*use_reentrant.*", UserWarning),
]

for _msg, _cat in _FILTERS:
    warnings.filterwarnings("ignore", message=_msg, category=_cat)

# Per-module blanket filters for the worst offenders (kept narrow).
warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers\..*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"peft\..*")


def install() -> None:
    """No-op hook so callers can express intent as `silence.install()`.

    Importing the module is sufficient; this just makes the intent explicit
    at call sites that prefer a function call over an import-with-side-effect.
    """
    return None
