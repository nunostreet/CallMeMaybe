import json
from pathlib import Path
from typing import Any


def write_json_file(data: Any, output_path: Path) -> None:
    """Write JSON-serializable data to disk, creating parent dirs if needed."""
    # Creates data/output/ if it doesn't exist yet — avoids FileNotFoundError
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        # indent=2 produces human-readable JSON
        # ensure_ascii=False keeps characters like é, ñ as-is instead of \uXXXX
        json.dump(data, file, indent=2, ensure_ascii=False)
