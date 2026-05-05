import json
from pathlib import Path
from pydantic import ValidationError
from .schema import PromptRequest, FunctionSpec


def _load_json_file(path: str | Path) -> object:
    """Read a JSON file and return its parsed content."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file: {path} (line {exc.lineno}, col {exc.colno})") from exc


def load_prompts(path: str | Path) -> list[PromptRequest]:
    """Load a list of user prompts from a JSON file and validate each entry."""
    raw_data = _load_json_file(path)

    if not isinstance(raw_data, list):
        raise ValueError("Prompt must be in a valid JSON format.")

    valid = []
    for i, item in enumerate(raw_data):
        try:
            valid.append(PromptRequest.model_validate(item))
        except ValidationError as exc:
            print(f"Skipping invalid prompt at index {i}: {exc.errors()[0]['msg']} ({item!r})")
    return valid


def load_functions(path: str | Path) -> list[FunctionSpec]:
    """Load function definitions from a JSON file and validate each entry."""
    raw_data = _load_json_file(path)

    if not isinstance(raw_data, list):
        raise ValueError("Function def. file must contain a JSON array.")

    try:
        return [FunctionSpec.model_validate(item) for item in raw_data]
    except ValidationError as exc:
        raise ValueError(
            "Function definitions do not match the expected schema."
        ) from exc
