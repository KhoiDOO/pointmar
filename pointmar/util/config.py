import json
import argparse
from typing import Union, Dict, Any

def save_args_to_json(args: argparse.Namespace, filepath: str) -> None:
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4)

def load_args_from_json(filepath: str) -> argparse.Namespace:
    with open(filepath, 'r') as f:
        args_dict: Dict[str, Any] = json.load(f)
    return argparse.Namespace(**args_dict)
