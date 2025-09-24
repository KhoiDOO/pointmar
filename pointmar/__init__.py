"""
PointMAR: Point cloud Masked Autoencoder Reconstruction
"""

import os
import sys
from pathlib import Path

def _get_version():
    """Read version from pyproject.toml file."""
    try:
        # Get the directory where this __init__.py file is located
        current_dir = Path(__file__).parent
        # Go up one level to find pyproject.toml
        pyproject_path = current_dir.parent / "pyproject.toml"
        
        if pyproject_path.exists():
            with open(pyproject_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith('version ='):
                        # Extract version from line like: version = "0.0.4"
                        version = line.split('=')[1].strip().strip('"').strip("'")
                        return version
        
        # Fallback version if pyproject.toml not found or version not found
        return "0.0.4"
    except Exception:
        # Fallback version if any error occurs
        return "0.0.4"

__version__ = _get_version()

# Import main modules
from . import data
from . import diffusion
from . import models
from . import util

__all__ = [
    "data",
    "diffusion", 
    "models",
    "util"
]