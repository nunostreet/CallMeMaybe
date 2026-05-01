"""Tests for Pydantic schema models in src/schema.py.

Each test class covers one model. The main things verified are:
- valid inputs are accepted and stored correctly
- invalid inputs raise ValidationError
- model_dump() produces the expected output structure
"""
import pytest
from pydantic import ValidationError

from src.schema import PromptRequest, TypeSpec, FunctionSpec, FunctionCall


class TestPromptRequest:
    """PromptRequest wraps a single user prompt string."""

    def test_valid(self):
        """A non-empty string is accepted."""
        pr = PromptRequest(prompt="What is 2 + 2?")
        assert pr.prompt == "What is 2 + 2?"

    def test_empty_string_rejected(self):
        """min_length=1 in the schema rejects empty strings."""
        with pytest.raises(ValidationError):
            PromptRequest.model_validate({"prompt": ""})

    def test_extra_fields_rejected(self):
        """extra='forbid' in CheckModel blocks unknown fields."""
        with pytest.raises(ValidationError):
            PromptRequest.model_validate({"prompt": "hello", "unexpected": "field"})

    def test_missing_prompt_rejected(self):
        """The prompt field is required — omitting it raises an error."""
        with pytest.raises(ValidationError):
            PromptRequest.model_validate({})


class TestTypeSpec:
    """TypeSpec holds the type string for a parameter or return value."""

    def test_valid(self):
        """'number' is a valid type string."""
        ts = TypeSpec(type="number")
        assert ts.type == "number"

    def test_string_type(self):
        """'string' is a valid type string."""
        ts = TypeSpec(type="string")
        assert ts.type == "string"

    def test_extra_fields_rejected(self):
        """TypeSpec inherits extra='forbid' from CheckModel."""
        with pytest.raises(ValidationError):
            TypeSpec.model_validate({"type": "number", "extra": "nope"})


class TestFunctionSpec:
    """FunctionSpec describes a callable function."""

    def test_valid(self, fn_add):
        """A fully specified function is accepted and fields are accessible."""
        assert fn_add.name == "fn_add_numbers"
        assert fn_add.parameters["a"].type == "number"
        assert fn_add.returns.type == "number"

    def test_missing_name_rejected(self):
        """name is a required field."""
        with pytest.raises(ValidationError):
            FunctionSpec.model_validate({
                "description": "desc",
                "parameters": {},
                "returns": {"type": "number"},
            })

    def test_missing_returns_rejected(self):
        """returns is a required field."""
        with pytest.raises(ValidationError):
            FunctionSpec.model_validate({
                "name": "fn_test",
                "description": "desc",
                "parameters": {},
            })

    def test_extra_fields_rejected(self):
        """Unknown fields are blocked by extra='forbid'."""
        with pytest.raises(ValidationError):
            FunctionSpec.model_validate({
                "name": "fn_test",
                "description": "desc",
                "parameters": {},
                "returns": {"type": "number"},
                "unexpected": "field",
            })

    def test_empty_parameters_allowed(self):
        """A function with no parameters is valid."""
        fn = FunctionSpec(
            name="fn_no_args",
            description="No parameters.",
            parameters={},
            returns=TypeSpec(type="string"),
        )
        assert fn.parameters == {}


class TestFunctionCall:
    """FunctionCall is the output: prompt + selected function + args."""

    def test_valid_with_numbers(self):
        """Numeric parameters are stored as-is."""
        fc = FunctionCall(
            prompt="Add 2 and 3",
            name="fn_add_numbers",
            parameters={"a": 2.0, "b": 3.0},
        )
        assert fc.name == "fn_add_numbers"
        assert fc.parameters["a"] == 2.0

    def test_valid_with_string(self):
        """String parameters are stored as-is."""
        fc = FunctionCall(
            prompt="Greet shrek",
            name="fn_greet",
            parameters={"name": "shrek"},
        )
        assert fc.parameters["name"] == "shrek"

    def test_model_dump_structure(self):
        """model_dump() must produce exactly prompt, name, parameters."""
        fc = FunctionCall(
            prompt="Add 1 and 2",
            name="fn_add_numbers",
            parameters={"a": 1.0, "b": 2.0},
        )
        dumped = fc.model_dump()
        assert set(dumped.keys()) == {"prompt", "name", "parameters"}
