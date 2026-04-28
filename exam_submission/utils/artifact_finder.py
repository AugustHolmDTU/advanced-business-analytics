from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def submission_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return submission_root() / "data"


def manifest_path() -> Path:
    return data_root() / "artifact_manifest.json"


def load_manifest() -> dict[str, Any]:
    path = manifest_path()
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(relative_path: str) -> Path:
    return submission_root() / relative_path


def selected_artifacts() -> list[dict[str, Any]]:
    return list(load_manifest().get("selected_artifacts", []))


def missing_notes() -> list[str]:
    return list(load_manifest().get("missing_or_incompatible", []))


def artifact_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in selected_artifacts():
        path = resolve(item["path"])
        rows.append(
            {
                "path": item["path"],
                "type": item.get("type", ""),
                "exists": path.exists(),
                "note": item.get("note", ""),
            }
        )
    return rows

