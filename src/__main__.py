import argparse
from .parser import load_functions, load_prompts


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

    print(f"Loaded {len(functions)} functions and {len(prompts)} prompts.")


if __name__ == "__main__":
    main()
