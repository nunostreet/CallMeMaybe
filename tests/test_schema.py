"""Tests for Pydantic schema models in src/schema.py."""
import pytest
from pydantic import ValidationError

from src.schema import PromptRequest, TypeSpec, FunctionCall


class TestPromptRequest:

    def test_valid(self):
        pr = PromptRequest(prompt="What is 2 + 2?")
        assert pr.prompt == "What is 2 + 2?"

    def test_empty_string_rejected(self):
        with pytest.raises(ValidationError):
            PromptRequest.model_validate({"prompt": ""})


class TestTypeSpec:

    def test_valid(self):
        ts = TypeSpec(type="number")
        assert ts.type == "number"


class TestFunctionSpec:

    def test_valid(self, fn_add):
        assert fn_add.name == "fn_add_numbers"
        assert fn_add.parameters["a"].type == "number"
        assert fn_add.returns.type == "number"


class TestFunctionCall:

    def test_valid_with_numbers(self):
        fc = FunctionCall(
            prompt="Add 2 and 3",
            name="fn_add_numbers",
            parameters={"a": 2.0, "b": 3.0},
        )
        assert fc.name == "fn_add_numbers"
        assert fc.parameters["a"] == 2.0
