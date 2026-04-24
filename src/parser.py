import json
from pathlib import Path


def _load_json_file(path: str | Path) -> object:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file: {path}") from exc


def load_prompts(path):

    raw_data = _load_json_file(path)

    if not isinstance(raw_data, list):
        raise ValueError("Prompt must be in a valid JSON format.")
    
    for item in raw_data:
        

    # Pydantic to validate if there
