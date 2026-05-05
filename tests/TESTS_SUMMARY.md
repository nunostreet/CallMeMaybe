# Test Suite Summary

71 tests across 4 files. All run without loading the real model (<0.2s).

---

## test_schema.py — Pydantic model validation (15 tests)

### TestPromptRequest
| Test | What it checks |
|---|---|
| `test_valid` | A non-empty string is accepted |
| `test_empty_string_rejected` | Empty string fails `min_length=1` |
| `test_extra_fields_rejected` | Unknown fields are blocked by `extra='forbid'` |
| `test_missing_prompt_rejected` | `prompt` field is required |

### TestTypeSpec
| Test | What it checks |
|---|---|
| `test_valid` | `"number"` is a valid type string |
| `test_string_type` | `"string"` is a valid type string |
| `test_extra_fields_rejected` | Unknown fields are blocked |

### TestFunctionSpec
| Test | What it checks |
|---|---|
| `test_valid` | A fully specified function is accepted |
| `test_missing_name_rejected` | `name` field is required |
| `test_missing_returns_rejected` | `returns` field is required |
| `test_extra_fields_rejected` | Unknown fields are blocked |
| `test_empty_parameters_allowed` | A function with no parameters is valid |

### TestFunctionCall
| Test | What it checks |
|---|---|
| `test_valid_with_numbers` | Numeric parameters are stored correctly |
| `test_valid_with_string` | String parameters are stored correctly |
| `test_model_dump_structure` | `model_dump()` produces exactly `prompt`, `name`, `parameters` |

---

## test_parser.py — Input file loading (16 tests)

### TestLoadPrompts
| Test | What it checks |
|---|---|
| `test_loads_real_file` | Real file loads with 11 prompts |
| `test_all_items_are_non_empty` | Every prompt passes `min_length=1` |
| `test_file_not_found` | Missing file raises `FileNotFoundError` |
| `test_invalid_json` | Malformed JSON raises `ValueError` |
| `test_non_list_json` | JSON object at root (not array) raises `ValueError` |
| `test_invalid_item_schema` | Items missing `prompt` key fail validation |
| `test_empty_prompt_rejected` | Empty string fails validation |
| `test_accepts_path_object` | Both `str` and `Path` are accepted |

### TestLoadFunctions
| Test | What it checks |
|---|---|
| `test_loads_real_file` | Real file loads with 5 functions |
| `test_function_names` | All 5 expected function names are present |
| `test_parameter_types_are_valid` | Every parameter has type `number` or `string` |
| `test_file_not_found` | Missing file raises `FileNotFoundError` |
| `test_invalid_json` | Malformed JSON raises `ValueError` |
| `test_non_list_json` | JSON object at root raises `ValueError` |
| `test_missing_required_field` | Function definition missing required fields fails |
| `test_substitute_has_three_params` | `fn_substitute_string_with_regex` has exactly 3 params |

---

## test_generator.py — Constrained decoding logic (27 tests)

### TestGetValidTokens
| Test | What it checks |
|---|---|
| `test_empty_prefix_returns_tokens_matching_any_function` | With no prefix, any token starting a valid name is allowed |
| `test_partial_prefix_narrows_candidates` | As the prefix grows, fewer tokens remain valid |
| `test_wrong_prefix_returns_empty` | A prefix matching no function yields an empty list |
| `test_closing_quote_included_when_name_complete` | The closing `"` is valid once the full name is generated |
| `test_single_function_only_accepts_its_tokens` | Tokens for other function names are rejected |
| `test_both_functions_valid_at_shared_prefix` | When two functions share a prefix, both continuations are valid |

### TestBuildPrompt
| Test | What it checks |
|---|---|
| `test_contains_user_prompt` | The user's question appears in the prompt |
| `test_contains_function_name` | Each function name is listed |
| `test_contains_function_description` | The description is included |
| `test_lists_all_functions` | All available functions appear |
| `test_numbered_list` | Functions are numbered |
| `test_param_types_included` | Parameter types appear in the prompt |

### TestGenerateConstrained
| Test | What it checks |
|---|---|
| `test_returns_highest_logit_among_valid` | The valid token with highest logit is chosen |
| `test_ignores_invalid_tokens_even_if_highest` | Invalid tokens are never chosen even with highest logit |
| `test_single_valid_token_always_selected` | When only one token is valid, it is always chosen |

### TestExtractArguments
| Test | What it checks |
|---|---|
| `test_extracts_two_numbers` | Both numbers are extracted from the prompt in order |
| `test_extracts_larger_numbers` | Larger numbers are extracted correctly |
| `test_string_param_calls_model` | String params trigger model generation |
| `test_number_extracted_as_float` | Numbers are stored as `float` |
| `test_decimal_number_extracted` | Decimal numbers are parsed correctly |
| `test_negative_number_extracted` | Negative numbers with leading minus are parsed correctly |

### TestExtractStringSpecialParams
| Test | What it checks |
|---|---|
| `test_unix_path_extracted_from_prompt` | Unix paths extracted by regex without calling the model |
| `test_windows_path_extracted_from_prompt` | Windows paths extracted by regex without calling the model |
| `test_template_extracted_from_prompt` | Template content after `Format template:` extracted directly |
| `test_template_with_embedded_quotes` | Templates with embedded `"` are extracted without truncation |
| `test_path_extraction_does_not_call_model` | Model is never called when path is in the prompt |
| `test_template_extraction_does_not_call_model` | Model is never called when template is in the prompt |

---

## test_output.py — End-to-end output validation (13 tests, skipped if output not generated)

> Run `make run` first to generate `data/output/function_calling_results.json`, then `make test`.

### TestOutputStructure
| Test | What it checks |
|---|---|
| `test_has_eleven_results` | Output has exactly 11 results |
| `test_each_result_has_required_keys` | Every result has `prompt`, `name`, `parameters` |
| `test_function_names_are_valid` | Every selected function is one of the 5 defined ones |

### TestOutputCorrectness
| Test | What it checks |
|---|---|
| `test_function_selection_all_correct` | Every prompt maps to the expected function name |
| `test_numeric_parameters_correct` | All numeric arguments match exactly |
| `test_greet_name_correct` | Both greeting prompts extract the correct name |
| `test_reverse_string_correct` | Both reversal prompts extract the correct input string |
| `test_substitute_source_strings_correct` | All 3 substitution prompts extract the correct source string |
| `test_substitute_replacements_correct` | All 3 replacement strings are extracted correctly |
| `test_regex_patterns_are_valid_regex` | Every generated regex compiles without error |
| `test_regex_numbers_pattern_works` | Numbers regex replaces all digit sequences correctly |
| `test_regex_vowels_pattern_works` | Vowels regex replaces all vowels with `*` correctly |
| `test_regex_cat_pattern_works` | Cat regex replaces all occurrences of `cat` with `dog` |
