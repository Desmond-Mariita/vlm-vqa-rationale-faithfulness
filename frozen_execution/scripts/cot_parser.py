"""Malformation-tolerant parser for ``<reasoning>``/``<final>`` CoT spans.

Motivation (see docs/KNOWN_LIMITATIONS.md): the Stage-2 models emit
the schema tags with several malformation modes:

  * whitespace drift           -- ``<reasoning >``, ``</reasoning >``
  * tag-name misspelling        -- ``<reasonning>`` (doubled n)
  * stray punctuation in a tag  -- ``<final'>``
  * an unclosed closing tag      -- ``</reasonning <final'>`` (no ``>``)
  * no tags at all               -- a bare final sentence

The original ``_parse_cot`` in ``train_rationale_qwen.py`` only tolerates
whitespace, so misspelled/punctuation-corrupted tags fall through to a
"treat the whole string as <final>" fallback that leaves literal tag debris
fused with the reasoning text in the scored span. This module recovers the
two spans across all of the above modes and, failing that, strips every
tag-like fragment and splits on the final sentence.

This is the single shared implementation §5.8.1 recommends; both the
write-time parser and any re-scoring tool should import ``parse_cot`` from
here rather than re-implementing the regex.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# A tag whose name is a fuzzy variant of "reasoning"/"final". ``\w*`` absorbs
# misspellings (``reasonning``); ``[^\s<>]*`` absorbs only stray *non-space*
# punctuation immediately after the name (``<final)``, ``<final'>``) and then an
# optional ``\s*>`` — crucially it does NOT cross whitespace, so when a tag has
# no closing ``>`` (``<final) person_2 ...``) the following words stay as
# content instead of being swallowed into the tag.
_TAG = re.compile(r"<\s*(/?)\s*(reason\w*|final\w*)[^\s<>]*\s*>?", re.IGNORECASE)

# Detect whether a string still carries schema tag debris (reason/final only,
# so a stray "<" in ordinary text is not treated as debris).
_DEBRIS = re.compile(r"<\s*/?\s*(reason|final)", re.IGNORECASE)

# Any residual tag-like fragment, for final cleanup of a span.
_ANY_TAGISH = re.compile(r"<\s*/?\s*\w[^<>]*>?")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def has_tag_debris(text: str) -> bool:
    """True if ``text`` still contains reasoning/final tag fragments."""
    return isinstance(text, str) and bool(_DEBRIS.search(text))


def _kind(slash: str, name: str) -> str:
    base = "reason" if name.lower().startswith("reason") else "final"
    return ("close_" if slash else "open_") + base


def _strip_tagish(s: str) -> str:
    s = _TAG.sub(" ", s)
    s = _ANY_TAGISH.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_cot(text: str) -> Tuple[str, str]:
    """Return ``(reasoning, final)`` extracted from a possibly-malformed CoT.

    Tolerates whitespace drift, tag-name misspellings, stray in-tag
    punctuation, and unclosed closing tags. If no usable reasoning/final tags
    are present, strips any tag debris and splits on the last sentence
    (reasoning = all but the last sentence, final = the last sentence).
    """
    if not isinstance(text, str):
        return "", ""
    s = text.strip()
    if not s:
        return "", ""

    tags = [(_kind(m.group(1), m.group(2)), m.start(), m.end())
            for m in _TAG.finditer(s)]

    reasoning = ""
    final = ""

    # reasoning span: first open_reason -> next close_reason or open_final
    for i, (k, _st, en) in enumerate(tags):
        if k == "open_reason":
            end_pos = len(s)
            for k2, st2, _en2 in tags[i + 1:]:
                if k2 in ("close_reason", "open_final"):
                    end_pos = st2
                    break
            reasoning = _strip_tagish(s[en:end_pos])
            break

    # final span: first open_final -> next close_final or end of string
    for i, (k, _st, en) in enumerate(tags):
        if k == "open_final":
            end_pos = len(s)
            for k2, st2, _en2 in tags[i + 1:]:
                if k2 == "close_final":
                    end_pos = st2
                    break
            final = _strip_tagish(s[en:end_pos])
            break

    if not reasoning and not final:
        clean = _strip_tagish(s)
        sents: List[str] = [x.strip() for x in _SENT_SPLIT.split(clean) if x.strip()]
        if len(sents) >= 2:
            final = sents[-1]
            reasoning = " ".join(sents[:-1])
        else:
            final = clean

    return reasoning, final


def _selftest() -> None:
    cases = [
        # (raw, expected_reasoning_substr, expected_final_substr)
        ("<reasoning> A is true . </reasoning> <final> B follows . </final>",
         "A is true", "B follows"),
        # whitespace drift
        ("<reasonING > A is true . </reasonING > <final > B follows . </final >",
         "A is true", "B follows"),
        # misspelled + apostrophe + unclosed close tag (the plain_desc mode)
        ("<reasonning> the blue square is on the right . </reasonning <final'> the red circle is on the left .",
         "the blue square is on the right", "the red circle is on the left"),
        # final tag with NO closing '>' followed directly by content (point_desc mode)
        ("<reasonING> A is a red circle . </reasonING <final) B is a blue square .",
         "A is a red circle", "B is a blue square"),
        # final marker as a bare word, no bracket at all
        ("<reasonING> The circle is red . </reasonING <final The square is blue .",
         "The circle is red", "The square is blue"),
        # bare final sentence, no tags
        ("the red circle is left of the square .",
         "", "the red circle is left"),
        # two bare sentences -> split
        ("The shape is round . It is a circle .",
         "The shape is round", "It is a circle"),
        # empty
        ("", "", ""),
    ]
    for raw, exp_r, exp_f in cases:
        r, f = parse_cot(raw)
        assert exp_r in r, f"reasoning mismatch for {raw!r}: got {r!r}, want substr {exp_r!r}"
        assert exp_f in f, f"final mismatch for {raw!r}: got {f!r}, want substr {exp_f!r}"
        # cleaned spans must carry no residual tag debris
        assert not has_tag_debris(r) and not has_tag_debris(f), \
            f"residual debris for {raw!r}: r={r!r} f={f!r}"
    print("cot_parser self-test: all cases passed")


if __name__ == "__main__":
    _selftest()
