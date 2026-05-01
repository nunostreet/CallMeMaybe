"""Tests that validate the generated output file.

These tests are skipped automatically if the output file does not exist.
Run `make run` first to generate it, then `make test` to validate.

Two things are checked:
1. Structure: the JSON has the right shape (11 items, required keys).
2. Correctness: function names match, numeric args match, and each
   regex actually produces the expected substitution.
"""
import re
import json
import pytest
from pathlib import Path


EXPECTED_PATH = Path("data/output/function_calling_results.json")

# Ground truth for the 11 standard test prompts.
# Regex string values are NOT compared directly — instead the regex is
# applied to the source string and the result is checked (see below).
EXPECTED_RESULTS = [
    {"prompt": "What is the sum of 2 and 3?",
     "name": "fn_add_numbers",
     "parameters": {"a": 2.0, "b": 3.0}},
    {"prompt": "What is the sum of 265 and 345?",
     "name": "fn_add_numbers",
     "parameters": {"a": 265.0, "b": 345.0}},
    {"prompt": "Greet shrek",
     "name": "fn_greet",
     "parameters": {"name": "shrek"}},
    {"prompt": "Greet john",
     "name": "fn_greet",
     "parameters": {"name": "john"}},
    {"prompt": "Reverse the string 'hello'",
     "name": "fn_reverse_string",
     "parameters": {"s": "hello"}},
    {"prompt": "Reverse the string 'world'",
     "name": "fn_reverse_string",
     "parameters": {"s": "world"}},
    {"prompt": "What is the square root of 16?",
     "name": "fn_get_square_root",
     "parameters": {"a": 16.0}},
    {"prompt": "Calculate the square root of 144",
     "name": "fn_get_square_root",
     "parameters": {"a": 144.0}},
    {"prompt": (
        "Replace all numbers in "
        "\"Hello 34 I'm 233 years old\" with NUMBERS"
     ),
     "name": "fn_substitute_string_with_regex",
     "parameters": {
         "source_string": "Hello 34 I'm 233 years old",
         "regex": "[0-9]+",
         "replacement": "NUMBERS",
     }},
    {"prompt": "Replace all vowels in 'Programming is fun' with asterisks",
     "name": "fn_substitute_string_with_regex",
     "parameters": {
         "source_string": "Programming is fun",
         "regex": "[aeiouAEIOU]",
         "replacement": "*",
     }},
    {"prompt": (
        "Substitute the word 'cat' with 'dog' in "
        "'The cat sat on the mat with another cat'"
     ),
     "name": "fn_substitute_string_with_regex",
     "parameters": {
         "source_string": "The cat sat on the mat with another cat",
         "regex": "cat",
         "replacement": "dog",
     }},
]


@pytest.fixture
def output():
    """Load the generated output file, or skip if it does not exist."""
    if not EXPECTED_PATH.exists():
        pytest.skip("Output file not generated yet — run `make run` first")
    return json.loads(EXPECTED_PATH.read_text())


class TestOutputStructure:
    """Basic shape checks on the output file."""

    def test_has_eleven_results(self, output):
        """One result per prompt — 11 prompts, 11 results."""
        assert len(output) == 11

    def test_each_result_has_required_keys(self, output):
        """Every result must have prompt, name, and parameters."""
        for item in output:
            assert "prompt" in item
            assert "name" in item
            assert "parameters" in item

    def test_function_names_are_valid(self, output):
        """Every selected function must be one of the five defined ones."""
        valid_names = {
            "fn_add_numbers", "fn_greet", "fn_reverse_string",
            "fn_get_square_root", "fn_substitute_string_with_regex",
        }
        for item in output:
            assert item["name"] in valid_names


class TestOutputCorrectness:
    """Content checks — right functions selected with right arguments."""

    def test_function_selection_all_correct(self, output):
        """Every prompt must map to the expected function name.

        Errors are collected together so all failures are visible at once.
        """
        errors = []
        for actual, expected in zip(output, EXPECTED_RESULTS):
            if actual["name"] != expected["name"]:
                errors.append(
                    f"Prompt: '{expected['prompt']}'\n"
                    f"  Expected: {expected['name']}\n"
                    f"  Got:      {actual['name']}"
                )
        assert not errors, "Wrong function:\n" + "\n".join(errors)

    def test_numeric_parameters_correct(self, output):
        """All numeric arguments must match exactly."""
        errors = []
        for actual, expected in zip(output, EXPECTED_RESULTS):
            for key, val in expected["parameters"].items():
                if isinstance(val, float):
                    actual_val = actual["parameters"].get(key)
                    if actual_val != val:
                        errors.append(
                            f"Prompt: '{expected['prompt']}'\n"
                            f"  '{key}': expected {val}, got {actual_val}"
                        )
        assert not errors, "Wrong param:\n" + "\n".join(errors)

    def test_greet_name_correct(self, output):
        """Both greeting prompts must extract the correct name."""
        greet_results = [r for r in output if r["name"] == "fn_greet"]
        names = {r["parameters"].get("name") for r in greet_results}
        assert "shrek" in names
        assert "john" in names

    def test_reverse_string_correct(self, output):
        """Both reversal prompts must extract the correct input string."""
        reverse_results = [
            r for r in output if r["name"] == "fn_reverse_string"
        ]
        strings = {r["parameters"].get("s") for r in reverse_results}
        assert "hello" in strings
        assert "world" in strings

    def test_substitute_source_strings_correct(self, output):
        """All three substitution prompts must extract the source string."""
        sub = [
            r for r in output
            if r["name"] == "fn_substitute_string_with_regex"
        ]
        assert len(sub) == 3
        sources = [r["parameters"].get("source_string") for r in sub]
        assert "Hello 34 I'm 233 years old" in sources
        assert "Programming is fun" in sources
        assert "The cat sat on the mat with another cat" in sources

    def test_substitute_replacements_correct(self, output):
        """All three replacement strings must be extracted correctly."""
        sub = [
            r for r in output
            if r["name"] == "fn_substitute_string_with_regex"
        ]
        replacements = [r["parameters"].get("replacement") for r in sub]
        assert "NUMBERS" in replacements
        assert "*" in replacements
        assert "dog" in replacements

    def test_regex_patterns_are_valid_regex(self, output):
        """Every generated regex must compile without error."""
        sub = [
            r for r in output
            if r["name"] == "fn_substitute_string_with_regex"
        ]
        for result in sub:
            pattern = result["parameters"].get("regex", "")
            try:
                re.compile(pattern)
            except re.error as e:
                pytest.fail(f"Invalid regex '{pattern}': {e}")

    def test_regex_numbers_pattern_works(self, output):
        """The numbers regex must replace all digit sequences."""
        sub = [
            r for r in output
            if r["name"] == "fn_substitute_string_with_regex"
        ]
        result = next(
            r for r in sub
            if r["parameters"].get("source_string")
            == "Hello 34 I'm 233 years old"
        )
        pattern = result["parameters"]["regex"]
        replacement = result["parameters"]["replacement"]
        out = re.sub(pattern, replacement, "Hello 34 I'm 233 years old")
        assert out == "Hello NUMBERS I'm NUMBERS years old"

    def test_regex_vowels_pattern_works(self, output):
        """The vowels regex must replace all vowels with '*'."""
        sub = [
            r for r in output
            if r["name"] == "fn_substitute_string_with_regex"
        ]
        result = next(
            r for r in sub
            if r["parameters"].get("source_string") == "Programming is fun"
        )
        pattern = result["parameters"]["regex"]
        replacement = result["parameters"]["replacement"]
        out = re.sub(pattern, replacement, "Programming is fun")
        assert out == "Pr*gr*mm*ng *s f*n"

    def test_regex_cat_pattern_works(self, output):
        """The cat regex must replace all occurrences of 'cat' with 'dog'."""
        sub = [
            r for r in output
            if r["name"] == "fn_substitute_string_with_regex"
        ]
        source = "The cat sat on the mat with another cat"
        result = next(
            r for r in sub
            if r["parameters"].get("source_string") == source
        )
        pattern = result["parameters"]["regex"]
        replacement = result["parameters"]["replacement"]
        out = re.sub(pattern, replacement, source)
        assert out == "The dog sat on the mat with another dog"
