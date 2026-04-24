from .parser import _load_json_file


def main():
    path = "data/input/function_calling_tests.json"
    data = _load_json_file(path)
    print(data)


if __name__ == "__main__":
    main()
