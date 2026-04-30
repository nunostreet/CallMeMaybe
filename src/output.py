"""Utilities for writing output files."""
import json
from pathlib import Path
from typing import Any


def write_json_file(data: Any, output_path: Path) -> None:
    """Write JSON-serializable data to disk, creating parent dirs if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
