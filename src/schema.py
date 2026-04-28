from pydantic import BaseModel, Field, ConfigDict
from typing import Any


class CheckModel(BaseModel):
    """Base model that restricts unexpected extra fields."""

    model_config = ConfigDict(extra='forbid')


class PromptRequest(CheckModel):
    """Checks prompt entry from input file."""

    prompt: str = Field(min_length=1)


class TypeSpec(CheckModel):
    """Checks type format for parameters and return values."""

    type: str


class FunctionSpec(CheckModel):
    """Checks function format."""

    name: str
    description: str
    parameters: dict[str, TypeSpec]
    returns: TypeSpec


class FunctionCall(CheckModel):
    """Output as requested by the subject."""

    prompt: str
    name: str
    parameters: dict[str, Any]
