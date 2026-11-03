"""Atomic JSON registry for sequential training progress."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = Path("training_registry.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    if not path.exists():
        return {"experiments": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Training registry is unreadable: {path}: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("experiments"), dict):
        raise RuntimeError(f"Training registry has an invalid format: {path}")
    return data


def update_model(model: str, **fields: Any) -> None:
    path = DEFAULT_REGISTRY
    data = read_registry(path)
    previous = data["experiments"].get(model, {})
    previous.update(fields)
    previous["model"] = model
    previous["updated_at"] = _now()
    data["experiments"][model] = previous
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     delete=False, suffix=".tmp") as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
