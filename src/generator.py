from llm_sdk import Small_LLM_Model
from .schema import FunctionSpec, FunctionCall
import json
import re
import numpy as np


def load_vocab(model: Small_LLM_Model) -> dict[int, str]:
    path = model.get_path_to_tokenizer_file()
    with open(path, encoding="utf-8") as file:
        tok = json.load(file)
    return {v: k for k, v in tok["model"]["vocab"].items()}


def generate_constrained(
    model: Small_LLM_Model,
    input_ids: list[int],
    valid_token_ids: list[int],
) -> int:
    logits = model.get_logits_from_input_ids(input_ids)
    mask = np.full(len(logits), -np.inf)
    for token_id in valid_token_ids:
        mask[token_id] = logits[token_id]
    return int(np.argmax(mask))


def select_function(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    user_prompt: str,
    functions: list[FunctionSpec],
) -> FunctionSpec:
    prompt = _build_prompt(user_prompt, functions)
    input_ids = model.encode(prompt).tolist()[0]
    prefix_ids = model.encode('{"name":"').tolist()[0]
    input_ids = input_ids + prefix_ids

    fn_names = [f.name for f in functions]
    generated_prefix = ""

    while True:
        valid = _get_valid_tokens(vocab, fn_names, generated_prefix)
        next_token = generate_constrained(model, input_ids, valid)
        token_str = vocab[next_token]
        generated_prefix += token_str
        input_ids.append(next_token)
        if generated_prefix.endswith('"'):
            generated_prefix = generated_prefix[:-1]
            break

    return next(f for f in functions if f.name == generated_prefix)


def extract_arguments(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    user_prompt: str,
    function: FunctionSpec,
) -> dict:
    result: dict[str, float | str] = {}
    numbers = re.findall(r'-?\d+(?:\.\d+)?', user_prompt)
    number_idx = 0

    for param_name, param_spec in function.parameters.items():
        if param_spec.type == "number":
            result[param_name] = float(numbers[number_idx])
            number_idx += 1
        elif param_spec.type == "string":
            result[param_name] = _extract_string(
                model, vocab, user_prompt, function.name, param_name
            )
    return result


def generate(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    functions: list[FunctionSpec],
) -> FunctionCall:
    function = select_function(model, vocab, prompt, functions)
    parameters = extract_arguments(model, vocab, prompt, function)
    return FunctionCall(
        prompt=prompt,
        name=function.name,
        parameters=parameters)


# --- helpers -----------------------------------------------------------------

def _build_prompt(user_prompt: str, functions: list[FunctionSpec]) -> str:
    lines = []
    for i, f in enumerate(functions, 1):
        params = ", ".join(
            f"{name}: {spec.type}"
            for name, spec in f.parameters.items()
        )
        lines.append(f"{i}. {f.name} | {f.description} | params: {params}")

    return (
        "You are a function-calling machine.\n"
        "Do not answer the user directly.\n"
        "Choose one of the following functions:\n"
        + "\n".join(lines)
        + f'\n\nUser prompt: "{user_prompt}"\n'
        'Function: "'
    )


def _get_valid_tokens(
    vocab: dict[int, str],
    fn_names: list[str],
    generated_prefix: str,
) -> list[int]:
    targets = [f'{name}"' for name in fn_names]
    return [
        token_id for token_id, token_str in vocab.items()
        if any(t.startswith(generated_prefix + token_str) for t in targets)
    ]


def _extract_string(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    user_prompt: str,
    fn_name: str,
    param_name: str,
) -> str:
    prompt = (
        f'User said: "{user_prompt}"\n'
        f'Function {fn_name} needs string argument "{param_name}".\n'
        f'{param_name} = "'
    )
    input_ids = model.encode(prompt).tolist()[0]

    string_tokens = [
        token_id for token_id, token_str in vocab.items()
        if token_str and '"' not in token_str
    ]

    generated_ids = []
    for _ in range(80):
        next_token = generate_constrained(model, input_ids, string_tokens)
        generated_ids.append(next_token)
        input_ids.append(next_token)

    return model.decode(generated_ids)
