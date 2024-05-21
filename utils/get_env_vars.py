import json

class JSONConversionError(Exception):
    """Custom exception for JSON conversion errors."""
    pass

def get_env_vars(file_path):
    """
    Reads a JSON file and returns its contents as a Python dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The JSON data converted to a Python dictionary.

    Raises:
        JSONConversionError: If the file is not found or the JSON is invalid.
    """
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        raise JSONConversionError(f"File not found: '{file_path}'")
    except json.JSONDecodeError as e:
        raise JSONConversionError(f"Invalid JSON: '{file_path}' - {e}")

