"""Shared pytest fixtures for all test modules.

llm_sdk is injected as a MagicMock before any src import so the real model
(which requires GPU and large downloads) is never loaded during testing.
"""
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# Replace llm_sdk with a fake before src modules are imported.
# Without this, importing src.generator triggers the real model load.
_fake_llm_sdk = MagicMock()
_fake_llm_sdk.Small_LLM_Model = MagicMock
sys.modules.setdefault("llm_sdk", _fake_llm_sdk)

from src.schema import FunctionSpec, TypeSpec  # noqa: E402


FUNCTIONS_PATH = Path("data/input/functions_definition.json")


@pytest.fixture
def functions() -> list[FunctionSpec]:
    """Load the real function definitions from disk."""
    raw = json.loads(FUNCTIONS_PATH.read_text())
    return [FunctionSpec.model_validate(f) for f in raw]


@pytest.fixture
def fn_add() -> FunctionSpec:
    """A simple two-number addition function for use in tests."""
    return FunctionSpec(
        name="fn_add_numbers",
        description="Add two numbers together.",
        parameters={
            "a": TypeSpec(type="number"),
            "b": TypeSpec(type="number"),
        },
        returns=TypeSpec(type="number"),
    )


@pytest.fixture
def fn_greet() -> FunctionSpec:
    """A single string-parameter greeting function for use in tests."""
    return FunctionSpec(
        name="fn_greet",
        description="Greet a person by name.",
        parameters={"name": TypeSpec(type="string")},
        returns=TypeSpec(type="string"),
    )


@pytest.fixture
def fn_substitute() -> FunctionSpec:
    """A three-string-parameter regex substitution function."""
    return FunctionSpec(
        name="fn_substitute_string_with_regex",
        description="Replace occurrences of a regex pattern.",
        parameters={
            "source_string": TypeSpec(type="string"),
            "regex": TypeSpec(type="string"),
            "replacement": TypeSpec(type="string"),
        },
        returns=TypeSpec(type="string"),
    )


@pytest.fixture
def simple_vocab() -> dict[int, str]:
    """Minimal vocabulary for testing token logic without a real model.

    Contains just enough tokens to cover fn_add_numbers and fn_greet paths.
    """
    return {
        0: "fn_",
        1: "add",
        2: "_numbers",
        3: "greet",
        4: '"',
        5: "xyz",
        6: "fn_add",
        7: "_greet",
    }


@pytest.fixture
def mock_model():
    """A MagicMock that mimics Small_LLM_Model for unit tests.

    encode() returns a fixed list of token IDs [1, 2, 3].
    decode() returns "shrek" by default (overridable per test).
    """
    model = MagicMock()
    model.encode.return_value = MagicMock(
        tolist=MagicMock(return_value=[[1, 2, 3]])
    )
    model.decode.return_value = "shrek"
    return model
