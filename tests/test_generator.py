"""Tests for pure functions in src/generator.py."""
import numpy as np

from src.generator import (
    _get_valid_tokens,
    _build_prompt,
    generate_constrained,
    extract_arguments,
    _extract_string,
)


class TestGetValidTokens:

    def test_empty_prefix_returns_tokens_matching_any_function(
        self, simple_vocab
    ):
        fn_names = ["fn_add_numbers", "fn_greet"]
        valid = _get_valid_tokens(simple_vocab, fn_names, "")
        valid_strs = [simple_vocab[t] for t in valid]
        assert "fn_" in valid_strs

    def test_wrong_prefix_returns_empty(self, simple_vocab):
        fn_names = ["fn_add_numbers"]
        valid = _get_valid_tokens(simple_vocab, fn_names, "zzz")
        assert valid == []

    def test_closing_quote_included_when_name_complete(self):
        vocab = {0: "fn_add_numbers", 1: '"', 2: "other"}
        fn_names = ["fn_add_numbers"]
        valid = _get_valid_tokens(vocab, fn_names, "fn_add_numbers")
        assert 1 in valid


class TestBuildPrompt:

    def test_contains_user_prompt(self, fn_add):
        prompt = _build_prompt("Add 2 and 3", [fn_add])
        assert "Add 2 and 3" in prompt

    def test_lists_all_functions(self, fn_add, fn_greet):
        prompt = _build_prompt("hello", [fn_add, fn_greet])
        assert "fn_add_numbers" in prompt
        assert "fn_greet" in prompt


class TestGenerateConstrained:

    def test_returns_highest_logit_among_valid(self, mock_model):
        logits = np.array([0.1, 5.0, 0.3, 0.2])
        mock_model.get_logits_from_input_ids.return_value = logits
        result = generate_constrained(
            mock_model, [1, 2], valid_token_ids=[0, 1, 2]
        )
        assert result == 1

    def test_ignores_invalid_tokens_even_if_highest(self, mock_model):
        logits = np.array([0.1, 99.0, 0.3, 0.2])
        mock_model.get_logits_from_input_ids.return_value = logits
        result = generate_constrained(
            mock_model, [1, 2], valid_token_ids=[0, 2, 3]
        )
        assert result == 2


class TestExtractArguments:

    def test_extracts_two_numbers(self, mock_model, fn_add):
        result = extract_arguments(mock_model, {}, "Add 2 and 3", fn_add)
        assert result == {"a": 2.0, "b": 3.0}

    def test_string_param_calls_model(self, mock_model, fn_greet):
        logits = np.zeros(100)
        logits[34] = 1.0
        mock_model.get_logits_from_input_ids.return_value = logits
        vocab = {i: f"tok{i}" for i in range(100)}
        vocab[34] = '"'

        result = extract_arguments(mock_model, vocab, "Greet shrek", fn_greet)
        assert "name" in result


class TestExtractStringSpecialParams:

    def test_unix_path_extracted_from_prompt(self, mock_model):
        result = _extract_string(
            mock_model, {},
            "Read the file at /home/user/data.json with utf-8 encoding",
            "fn_read_file", "path",
        )
        assert result == "/home/user/data.json"

    def test_path_extraction_does_not_call_model(self, mock_model):
        _extract_string(
            mock_model, {}, "Read /tmp/file.txt with utf-8",
            "fn_read_file", "path",
        )
        mock_model.get_logits_from_input_ids.assert_not_called()
