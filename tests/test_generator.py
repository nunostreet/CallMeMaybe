"""Tests for pure functions in src/generator.py.

The real model is never loaded — all model interactions go through
the mock_model fixture from conftest.py. This keeps tests fast
and runnable without GPU or internet access.

Functions tested:
- _get_valid_tokens: the constrained decoding filter
- _build_prompt: the instruction prompt builder
- generate_constrained: the masked argmax step
- extract_arguments: number regex + string generation
"""
import pytest
import numpy as np

from src.generator import (
    _get_valid_tokens,
    _build_prompt,
    generate_constrained,
    extract_arguments,
    _extract_string,
)


class TestGetValidTokens:
    """_get_valid_tokens returns token IDs that keep the prefix on track.

    A token is valid if (generated_prefix + token_str) is still a prefix
    of at least one target string of the form 'fn_name"'.
    """

    def test_empty_prefix_returns_tokens_matching_any_function(
        self, simple_vocab
    ):
        """With no prefix, any token starting a valid name is allowed."""
        fn_names = ["fn_add_numbers", "fn_greet"]
        valid = _get_valid_tokens(simple_vocab, fn_names, "")
        valid_strs = [simple_vocab[t] for t in valid]
        assert "fn_" in valid_strs

    def test_partial_prefix_narrows_candidates(self, simple_vocab):
        """As the prefix grows, fewer tokens remain valid."""
        fn_names = ["fn_add_numbers", "fn_greet"]
        valid_empty = _get_valid_tokens(simple_vocab, fn_names, "")
        valid_fn = _get_valid_tokens(simple_vocab, fn_names, "fn_")
        assert len(valid_fn) <= len(valid_empty)

    def test_wrong_prefix_returns_empty(self, simple_vocab):
        """A prefix matching no function name yields an empty list."""
        fn_names = ["fn_add_numbers"]
        valid = _get_valid_tokens(simple_vocab, fn_names, "zzz")
        assert valid == []

    def test_closing_quote_included_when_name_complete(self):
        """Once the full name is generated, the closing quote is valid."""
        vocab = {0: "fn_add_numbers", 1: '"', 2: "other"}
        fn_names = ["fn_add_numbers"]
        valid = _get_valid_tokens(vocab, fn_names, "fn_add_numbers")
        # token 1 ('"') must be valid so the loop can terminate
        assert 1 in valid

    def test_single_function_only_accepts_its_tokens(self):
        """Tokens for other function names are rejected."""
        vocab = {0: "fn_add", 1: "fn_greet", 2: '"'}
        fn_names = ["fn_add_numbers"]
        valid = _get_valid_tokens(vocab, fn_names, "")
        valid_strs = [vocab[t] for t in valid]
        assert "fn_greet" not in valid_strs

    def test_both_functions_valid_at_shared_prefix(self):
        """When two functions share a prefix, both continuations are valid."""
        vocab = {0: "fn_", 1: "add", 2: "greet", 3: '"'}
        fn_names = ["fn_add_numbers", "fn_greet"]
        valid = _get_valid_tokens(vocab, fn_names, "fn_")
        valid_strs = [vocab[t] for t in valid]
        assert "add" in valid_strs
        assert "greet" in valid_strs


class TestBuildPrompt:
    """_build_prompt assembles the instruction text sent to the model."""

    def test_contains_user_prompt(self, fn_add):
        """The user's question must appear in the prompt."""
        prompt = _build_prompt("Add 2 and 3", [fn_add])
        assert "Add 2 and 3" in prompt

    def test_contains_function_name(self, fn_add):
        """Each function name must be listed."""
        prompt = _build_prompt("Add 2 and 3", [fn_add])
        assert "fn_add_numbers" in prompt

    def test_contains_function_description(self, fn_add):
        """The description helps the model choose the right function."""
        prompt = _build_prompt("Add 2 and 3", [fn_add])
        assert "Add two numbers" in prompt

    def test_lists_all_functions(self, fn_add, fn_greet):
        """All available functions must appear, not just the first."""
        prompt = _build_prompt("hello", [fn_add, fn_greet])
        assert "fn_add_numbers" in prompt
        assert "fn_greet" in prompt

    def test_numbered_list(self, fn_add, fn_greet):
        """Functions are numbered so the model can refer to them."""
        prompt = _build_prompt("hello", [fn_add, fn_greet])
        assert "1." in prompt
        assert "2." in prompt

    def test_param_types_included(self, fn_add):
        """Parameter types appear to inform argument extraction."""
        prompt = _build_prompt("hello", [fn_add])
        assert "number" in prompt


class TestGenerateConstrained:
    """generate_constrained masks invalid tokens to -inf then picks argmax."""

    def test_returns_highest_logit_among_valid(self, mock_model):
        """Token 1 has logit 5.0 and is valid — it should win."""
        logits = np.array([0.1, 5.0, 0.3, 0.2])
        mock_model.get_logits_from_input_ids.return_value = logits
        result = generate_constrained(
            mock_model, [1, 2], valid_token_ids=[0, 1, 2]
        )
        assert result == 1

    def test_ignores_invalid_tokens_even_if_highest(self, mock_model):
        """Token 1 has logit 99.0 but is NOT valid — token 2 wins."""
        logits = np.array([0.1, 99.0, 0.3, 0.2])
        mock_model.get_logits_from_input_ids.return_value = logits
        result = generate_constrained(
            mock_model, [1, 2], valid_token_ids=[0, 2, 3]
        )
        assert result == 2

    def test_single_valid_token_always_selected(self, mock_model):
        """When only one token is valid, it is always chosen."""
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        mock_model.get_logits_from_input_ids.return_value = logits
        result = generate_constrained(
            mock_model, [1, 2], valid_token_ids=[0]
        )
        assert result == 0


class TestExtractArguments:
    """extract_arguments fills a dict of parameter values from the prompt."""

    def test_extracts_two_numbers(self, mock_model, fn_add):
        """Both numbers are extracted from the prompt in order."""
        result = extract_arguments(mock_model, {}, "Add 2 and 3", fn_add)
        assert result == {"a": 2.0, "b": 3.0}

    def test_extracts_larger_numbers(self, mock_model, fn_add):
        """Larger numbers are extracted correctly."""
        result = extract_arguments(
            mock_model, {}, "Sum of 265 and 345", fn_add
        )
        assert result == {"a": 265.0, "b": 345.0}

    def test_string_param_calls_model(self, mock_model, fn_greet):
        """String params trigger model generation — key must be in result."""
        # Generation stops on the first token that contains a quote.
        logits = np.zeros(100)
        logits[34] = 1.0
        mock_model.get_logits_from_input_ids.return_value = logits
        vocab = {i: f"tok{i}" for i in range(100)}
        vocab[34] = '"'  # token 34 contains '"' — generation stops here

        result = extract_arguments(mock_model, vocab, "Greet shrek", fn_greet)
        assert "name" in result

    def test_number_extracted_as_float(self, mock_model, fn_add):
        """Numbers are always stored as float, not int."""
        result = extract_arguments(mock_model, {}, "16 plus 4", fn_add)
        assert isinstance(result["a"], float)
        assert isinstance(result["b"], float)

    def test_decimal_number_extracted(self, mock_model):
        """Decimal numbers (e.g. 3.14) are parsed correctly."""
        from src.schema import FunctionSpec, TypeSpec
        fn = FunctionSpec(
            name="fn_test",
            description="Test",
            parameters={"x": TypeSpec(type="number")},
            returns=TypeSpec(type="number"),
        )
        result = extract_arguments(
            mock_model, {}, "Calculate 3.14 area", fn
        )
        assert result["x"] == pytest.approx(3.14)

    def test_negative_number_extracted(self, mock_model, fn_add):
        """Negative numbers (with leading minus) are parsed correctly."""
        result = extract_arguments(
            mock_model, {}, "Add -5 and 10", fn_add
        )
        assert result["a"] == -5.0
        assert result["b"] == 10.0


class TestExtractStringSpecialParams:
    """_extract_string extracts path and template directly from the prompt."""

    def test_unix_path_extracted_from_prompt(self, mock_model):
        """Unix paths are extracted by regex without calling the model."""
        result = _extract_string(
            mock_model, {}, "Read the file at /home/user/data.json with utf-8 encoding",
            "fn_read_file", "path",
        )
        assert result == "/home/user/data.json"

    def test_windows_path_extracted_from_prompt(self, mock_model):
        """Windows paths are extracted by regex without calling the model."""
        result = _extract_string(
            mock_model, {}, r"Read C:\Users\john\config.ini with latin-1 encoding",
            "fn_read_file", "path",
        )
        assert result == r"C:\Users\john\config.ini"

    def test_template_extracted_from_prompt(self, mock_model):
        """Template content after 'Format template:' is extracted directly."""
        result = _extract_string(
            mock_model, {}, "Format template: Hello {user}'s profile!",
            "fn_format_template", "template",
        )
        assert result == "Hello {user}'s profile!"

    def test_template_with_embedded_quotes(self, mock_model):
        """Templates containing double quotes are extracted without truncation."""
        result = _extract_string(
            mock_model, {}, 'Format template: Say "hello" to {name}',
            "fn_format_template", "template",
        )
        assert result == 'Say "hello" to {name}'

    def test_path_extraction_does_not_call_model(self, mock_model):
        """Model generation is never triggered when path is in the prompt."""
        _extract_string(
            mock_model, {}, "Read /tmp/file.txt with utf-8",
            "fn_read_file", "path",
        )
        mock_model.get_logits_from_input_ids.assert_not_called()

    def test_template_extraction_does_not_call_model(self, mock_model):
        """Model generation is never triggered when template is in the prompt."""
        _extract_string(
            mock_model, {}, "Format template: Hello world",
            "fn_format_template", "template",
        )
        mock_model.get_logits_from_input_ids.assert_not_called()
