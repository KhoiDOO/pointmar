import json
import argparse
from typing import Union, Dict, Any

def save_args_to_json(args: argparse.Namespace, filepath: str) -> None:
    """
    Saves command-line arguments from an argparse.Namespace object to a JSON file.

    Args:
        args (argparse.Namespace): The namespace object containing the arguments.
        filepath (str): The path to the JSON file where arguments will be saved.
    """
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4)

def load_args_from_json(filepath: str) -> argparse.Namespace:
    """
    Loads arguments from a JSON file and returns them as an argparse.Namespace object.

    Args:
        filepath (str): The path to the JSON file to load arguments from.

    Returns:
        argparse.Namespace: The namespace object containing the loaded arguments.
    """
    with open(filepath, 'r') as f:
        args_dict: Dict[str, Any] = json.load(f)
    return argparse.Namespace(**args_dict)
