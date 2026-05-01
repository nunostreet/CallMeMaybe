"""Tests for src/parser.py — loading and validating input JSON files.

Two functions are tested: load_prompts and load_functions.
Both share the same error-handling logic so similar edge cases
are tested for each.
"""
import pytest
from pathlib import Path

from src.parser import load_prompts, load_functions


class TestLoadPrompts:
    """load_prompts reads a JSON array of {prompt: str} objects."""

    def test_loads_real_file(self):
        """The real input file loads with the expected number of prompts."""
        prompts = load_prompts("data/input/function_calling_tests.json")
        assert len(prompts) == 11
        assert prompts[0].prompt == "What is the sum of 2 and 3?"

    def test_all_items_are_non_empty(self):
        """Every prompt in the real file passes the min_length=1 check."""
        prompts = load_prompts("data/input/function_calling_tests.json")
        for p in prompts:
            assert len(p.prompt) > 0

    def test_file_not_found(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_prompts("data/input/does_not_exist.json")

    def test_invalid_json(self, tmp_path):
        """Malformed JSON raises ValueError mentioning 'Invalid JSON'."""
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all {{{")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_prompts(bad)

    def test_non_list_json(self, tmp_path):
        """A JSON object at root level (not an array) raises ValueError."""
        bad = tmp_path / "bad.json"
        bad.write_text('{"prompt": "hello"}')
        with pytest.raises(ValueError, match="valid JSON format"):
            load_prompts(bad)

    def test_invalid_item_schema(self, tmp_path):
        """Items missing the 'prompt' key fail Pydantic validation."""
        bad = tmp_path / "bad.json"
        bad.write_text('[{"not_prompt": "hello"}]')
        with pytest.raises(ValueError, match="expected schema"):
            load_prompts(bad)

    def test_empty_prompt_rejected(self, tmp_path):
        """An empty string fails min_length=1 and raises ValueError."""
        bad = tmp_path / "bad.json"
        bad.write_text('[{"prompt": ""}]')
        with pytest.raises(ValueError):
            load_prompts(bad)

    def test_accepts_path_object(self):
        """Both str and pathlib.Path are accepted as the path argument."""
        prompts = load_prompts(
            Path("data/input/function_calling_tests.json")
        )
        assert len(prompts) > 0


class TestLoadFunctions:
    """load_functions reads a JSON array of function definition objects."""

    def test_loads_real_file(self):
        """The real functions file loads correctly and has 5 functions."""
        functions = load_functions("data/input/functions_definition.json")
        assert len(functions) == 5

    def test_function_names(self):
        """All five expected function names are present."""
        functions = load_functions("data/input/functions_definition.json")
        names = [f.name for f in functions]
        assert "fn_add_numbers" in names
        assert "fn_greet" in names
        assert "fn_reverse_string" in names
        assert "fn_get_square_root" in names
        assert "fn_substitute_string_with_regex" in names

    def test_parameter_types_are_valid(self):
        """Every parameter in the real file has type 'number' or 'string'."""
        functions = load_functions("data/input/functions_definition.json")
        for fn in functions:
            for param in fn.parameters.values():
                assert param.type in ("number", "string")

    def test_file_not_found(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_functions("data/input/does_not_exist.json")

    def test_invalid_json(self, tmp_path):
        """Malformed JSON raises ValueError."""
        bad = tmp_path / "bad.json"
        bad.write_text("{{broken")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_functions(bad)

    def test_non_list_json(self, tmp_path):
        """A JSON object at root level raises ValueError."""
        bad = tmp_path / "bad.json"
        bad.write_text('{"name": "fn_test"}')
        with pytest.raises(ValueError, match="JSON array"):
            load_functions(bad)

    def test_missing_required_field(self, tmp_path):
        """A function definition missing required fields fails validation."""
        bad = tmp_path / "bad.json"
        bad.write_text('[{"name": "fn_test", "parameters": {}}]')
        with pytest.raises(ValueError, match="expected schema"):
            load_functions(bad)

    def test_substitute_has_three_params(self):
        """fn_substitute_string_with_regex must have exactly three params."""
        functions = load_functions("data/input/functions_definition.json")
        sub = next(
            f for f in functions
            if f.name == "fn_substitute_string_with_regex"
        )
        assert set(sub.parameters.keys()) == {
            "source_string", "regex", "replacement"
        }
