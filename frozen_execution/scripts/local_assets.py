"""Validate caller-supplied local assets without network access."""
from pathlib import Path
import hashlib
import json

def validate_local_assets(path):
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not document.get("files"):
        raise ValueError("local asset manifest must have schema_version=1 and a nonempty files list")
    seen = set()
    for row in document["files"]:
        target = Path(row["path"])
        if not target.is_absolute():
            target = path.parent / target
        target = target.resolve()
        if target in seen or not target.is_file():
            raise ValueError("duplicate or missing local asset")
        seen.add(target)
        h = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                h.update(block)
        if h.hexdigest() != row["sha256"]:
            raise ValueError("local asset SHA-256 mismatch")
        if "bytes" in row and target.stat().st_size != row["bytes"]:
            raise ValueError("local asset size mismatch")
    return hashlib.sha256(path.read_bytes()).hexdigest()
