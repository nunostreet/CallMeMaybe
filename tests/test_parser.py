"""Tests for src/parser.py — loading and validating input JSON files."""
import pytest

from src.parser import load_prompts, load_functions


class TestLoadPrompts:

    def test_loads_real_file(self):
        prompts = load_prompts("data/input/function_calling_tests.json")
        assert len(prompts) == 11
        assert prompts[0].prompt == "What is the sum of 2 and 3?"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_prompts("data/input/does_not_exist.json")

    def test_invalid_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all {{{")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_prompts(bad)

    def test_invalid_item_schema(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('[{"not_prompt": "hello"}]')
        result = load_prompts(bad)
        assert result == []


class TestLoadFunctions:

    def test_loads_real_file(self):
        functions = load_functions("data/input/functions_definition.json")
        assert len(functions) == 5

    def test_missing_required_field(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('[{"name": "fn_test", "parameters": {}}]')
        with pytest.raises(ValueError, match="expected schema"):
            load_functions(bad)
