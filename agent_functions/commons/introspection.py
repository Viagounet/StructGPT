import ast
import inspect
import re


def retrieve_function_details(cls, method_name):
    method = getattr(cls, method_name, None)
    if method is None:
        return "Method not found"

    sig = inspect.signature(method)
    params = sig.parameters
    return_annotation = sig.return_annotation

    # Formatting the arguments and their types, excluding 'self'
    arg_list = [
        f"{name}: {param.annotation.__name__}"
        for name, param in params.items()
        if name != "self"
    ]
    args = ", ".join(arg_list)

    # Formatting the return type
    ret_type = (
        return_annotation.__name__
        if return_annotation is not inspect._empty
        else "None"
    )

    return f"{method_name}({args}) -> {ret_type}"


def parse_function_string(func_str):
    # Regular expression pattern to match the function call structure
    pattern = r"(\w+)\((.*)\)"
    match = re.match(pattern, func_str)

    if match:
        func_name = match.group(1)
        args_str = match.group(2)

        try:
            # Using ast.literal_eval to safely evaluate the argument string
            args = ast.literal_eval(f"[{args_str}]")
        except Exception as e:
            return f"Error parsing arguments: {e}", []

        return func_name, args
    else:
        return "Invalid function string", []
