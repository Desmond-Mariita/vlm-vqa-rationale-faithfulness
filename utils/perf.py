"""Resume-surviving pipeline timing + record-level resume helpers.

Two utilities shared across the training and eval pipeline:

* ``StageTimer`` records wall-clock per named stage into ONE json file
  (``pipeline_timing.json`` in the run dir by default), ACCUMULATING across
  process restarts. A stage timed per-segment (e.g. per epoch, per eval) keeps
  the wall-clock of every *completed* segment, so a run interrupted by a
  power-off and resumed reports a correct cumulative total at segment
  granularity. A pipeline-level ``total_s`` is maintained automatically.

* ``load_done_keys`` / ``ResumeJSONL`` give eval scripts record-level resume:
  a per-record JSONL is appended as each record is computed; on resume the set
  of already-finished keys is read back so only the remaining records run.

Neither helper touches the trainers' own checkpoint/optimizer resume (which
already exists); this is purely additive instrumentation.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set


def fmt_duration(seconds: float) -> str:
    """Seconds -> compact human string, e.g. 5025.0 -> '1h 23m 45s'."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    out = []
    if h:
        out.append(f"{h}h")
    if h or m:
        out.append(f"{m}m")
    out.append(f"{sec}s")
    return " ".join(out)


class StageTimer:
    """Accumulating, resume-surviving per-stage wall-clock timer.

    Writes a single json (atomic) holding per-stage cumulative seconds, a
    segment count, a human string, and a pipeline total. Safe to construct
    repeatedly across process restarts pointed at the same path: it loads and
    extends the existing record rather than overwriting it.
    """

    def __init__(self, timing_path: os.PathLike | str):
        self.path = Path(timing_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                pass
        return {"stages": {}, "total_s": 0.0, "total_human": "0s"}

    def _save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        os.replace(tmp, self.path)  # atomic

    def add(self, stage: str, seconds: float, meta: Optional[dict] = None) -> None:
        """Add ``seconds`` to ``stage`` and refresh the pipeline total."""
        st = self.data["stages"].setdefault(stage, {"seconds": 0.0, "segments": 0})
        st["seconds"] = round(st["seconds"] + float(seconds), 2)
        st["segments"] = int(st.get("segments", 0)) + 1
        st["human"] = fmt_duration(st["seconds"])
        if meta:
            st.setdefault("meta", {}).update(meta)
        self.data["total_s"] = round(
            sum(float(v["seconds"]) for v in self.data["stages"].values()), 2
        )
        self.data["total_human"] = fmt_duration(self.data["total_s"])
        self._save()

    @contextmanager
    def stage(self, name: str, meta: Optional[dict] = None):
        """Context manager: time the block and add it to ``name`` on exit.

        Note: if the process is killed mid-block (power-off), that segment's
        time is not recorded -- only completed segments accumulate. Time at
        epoch/eval granularity (call inside the loop) so a kill loses at most
        the current segment.
        """
        t0 = time.time()
        try:
            yield
        finally:
            self.add(name, time.time() - t0, meta)

    def summary(self) -> str:
        lines = [f"  {n:<22} {v['human']:>12}  ({v['segments']} seg)"
                 for n, v in self.data["stages"].items()]
        lines.append(f"  {'TOTAL':<22} {self.data.get('total_human', '0s'):>12}")
        return "\n".join(lines)


def time_main(fn, label: str = "script"):
    """Run ``fn`` (a script's main()), print its total wall-clock, and return its
    value. If PIPELINE_TIMING_PATH is set (i.e. run under the pipeline driver),
    also record the duration into that shared timing file. Always prints, even on
    error, so a standalone run is timed without any other setup.
    """
    t0 = time.time()
    try:
        return fn()
    finally:
        dt = time.time() - t0
        print(f"[timing] {label} total wall-clock: {fmt_duration(dt)} ({dt:.1f}s)", flush=True)
        p = os.environ.get("PIPELINE_TIMING_PATH")
        if p:
            try:
                StageTimer(p).add(label, dt)
            except Exception:
                pass


def load_done_keys(jsonl_path: os.PathLike | str, key: str = "key") -> Set[str]:
    """Read an append-as-you-go per-record JSONL and return the finished keys.

    Tolerant of a truncated final line (e.g. killed mid-write): a line that
    fails to parse is skipped rather than aborting resume.
    """
    done: Set[str] = set()
    p = Path(jsonl_path)
    if not p.exists():
        return done
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue  # truncated tail line
            if key in rec and rec[key] is not None:
                done.add(str(rec[key]))
    return done


class ResumeJSONL:
    """Append-mode per-record sink with record-level resume.

    Open with ``key`` naming the unique id field. ``done`` holds the keys
    already present (so callers can skip them). ``write(record)`` appends and
    flushes immediately, so an interrupted run keeps every completed record.
    """

    def __init__(self, jsonl_path: os.PathLike | str, key: str = "key"):
        self.path = Path(jsonl_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key = key
        self.done = load_done_keys(self.path, key)
        self._f = self.path.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())
        k = record.get(self.key)
        if k is not None:
            self.done.add(str(k))

    def records(self) -> Iterable[dict]:
        """Re-read all persisted records (for end-of-run summarisation)."""
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
