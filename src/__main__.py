import argparse
from pathlib import Path
from llm_sdk import Small_LLM_Model
from .parser import load_functions, load_prompts
from .generator import generate, load_vocab
from .output import write_json_file


def main() -> None:
    """Entry point for the function calling tool."""
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
    )
    args = parser.parse_args()

    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)

    model = Small_LLM_Model()
    vocab = load_vocab(model)

    results = []
    for i, prompt_request in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {prompt_request.prompt[:50]}...")
        try:
            function_call = generate(
                model, vocab, prompt_request.prompt, functions
                )
            results.append(function_call.model_dump())
        except Exception as e:
            print(f"Error processing prompt '{prompt_request.prompt}': {e}")

    write_json_file(results, Path(args.output))
    print(f"Written {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
