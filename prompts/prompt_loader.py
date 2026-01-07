"""
Prompt Loader Utility

This module provides functions to load prompts from external files.
Supports multiple formats: .py, .json, .txt

Users choose which format to use 

Usage:
    from prompts.prompt_loader import load_screening_prompts, load_extraction_prompts

    # Default: uses .py files (prompts as code)
    screening_prompts = load_screening_prompts()

    # Explicit format selection
    screening_prompts = load_screening_prompts(format='py')
    screening_prompts = load_screening_prompts(format='json')
    screening_prompts = load_screening_prompts(format='txt')
"""

import os
import json
import importlib.util
from pathlib import Path
from typing import Dict, Literal


# Get the prompts directory path
PROMPTS_DIR = Path(__file__).parent

# Supported prompt formats
PromptFormat = Literal['py', 'json', 'txt']


def load_prompt_from_py(file_path: Path) -> str:
    """Load prompt from Python file (.py)"""
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'PROMPT'):
        raise ValueError(f"Python prompt file {file_path} must define a PROMPT variable")

    return module.PROMPT


def load_prompt_from_json(file_path: Path) -> str:
    """Load prompt from JSON file (.json)"""
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        if isinstance(data, dict) and 'prompt' in data:
            return data['prompt']
        elif isinstance(data, str):
            return data
        else:
            raise ValueError(
                f"JSON prompt file {file_path} must be a string or "
                f"dict with 'prompt' key"
            )


def load_prompt_from_txt(file_path: Path) -> str:
    """Load prompt from text file (.txt)"""
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_prompt_file(base_path: Path, filename_without_ext: str, format: PromptFormat = 'txt') -> str:
    """
    Load prompt from file in specified format.

    Args:
        base_path: Directory containing the prompt file
        filename_without_ext: Filename without extension (e.g., "reviewer_system_prompt")
        format: File format to use ('py', 'json', or 'txt'). Default: 'py'

    Returns:
        Content of the prompt as a string

    Raises:
        FileNotFoundError: If the prompt file is not found
        ValueError: If format is invalid or file format is incorrect
    """
    file_path = base_path / f"{filename_without_ext}.{format}"

    if format == 'py':
        return load_prompt_from_py(file_path)
    elif format == 'json':
        return load_prompt_from_json(file_path)
    elif format == 'txt':
        return load_prompt_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'py', 'json', or 'txt'")


def load_screening_prompts(format: PromptFormat = 'txt') -> Dict[str, Dict[str, str]]:
    """
    Load all screening agent prompts.

    Args:
        format: File format to use ('py', 'json', or 'txt'). Default: 'py'

    Returns:
        Dictionary containing prompts for screening agents:
        {
            "reviewer": {
                "system_prompt": "...",
                "user_prompt_template": "..."
            },
            "adjudicator": {
                "system_prompt": "...",
                "user_prompt_template": "..."
            }
        }
    """
    screening_dir = PROMPTS_DIR / "screening_agents"

    return {
        "reviewer": {
            "system_prompt": load_prompt_file(screening_dir, "reviewer_system_prompt", format),
            "user_prompt_template": load_prompt_file(screening_dir, "reviewer_user_prompt", format)
        },
        "adjudicator": {
            "system_prompt": load_prompt_file(screening_dir, "adjudicator_system_prompt", format),
            "user_prompt_template": load_prompt_file(screening_dir, "adjudicator_user_prompt", format)
        }
    }


def load_extraction_prompts(format: PromptFormat = 'txt') -> Dict[str, str]:
    """
    Load all extraction agent prompts.

    Args:
        format: File format to use ('py', 'json', or 'txt'). Default: 'py'

    Returns:
        Dictionary containing prompts for extraction agents:
        {
            "quantitative_extraction": "...",
            "qualitative_extraction": "...",
            "figure_extraction": "...",
            "reconciliation_system": "...",
            "reconciliation_user_template": "..."
        }
    """
    extraction_dir = PROMPTS_DIR / "extraction_agents"

    return {
        "quantitative_extraction": load_prompt_file(extraction_dir, "quantitative_extraction_prompt", format),
        "qualitative_extraction": load_prompt_file(extraction_dir, "qualitative_extraction_prompt", format),
        "figure_extraction": load_prompt_file(extraction_dir, "figure_extraction_prompt", format),
        "reconciliation_system": load_prompt_file(extraction_dir, "reconciliation_system_prompt", format),
        "reconciliation_user_template": load_prompt_file(extraction_dir, "reconciliation_user_prompt", format)
    }


def get_all_prompts(format: PromptFormat = 'txt') -> Dict[str, Dict]:
    """
    Load all prompts (both screening and extraction).

    Args:
        format: File format to use ('py', 'json', or 'txt'). Default: 'py'

    Returns:
        Dictionary containing all prompts:
        {
            "screening": {...},
            "extraction": {...}
        }
    """
    return {
        "screening": load_screening_prompts(format),
        "extraction": load_extraction_prompts(format)
    }


if __name__ == "__main__":
    # Test the loader with all formats
    print("Testing prompt loader with all formats...")
    print("=" * 60)

    for fmt in ['txt', 'py', 'json']:
        print(f"\n Testing {fmt.upper()} format:")
        try:
            screening = load_screening_prompts(format=fmt)
            extraction = load_extraction_prompts(format=fmt)

            print(f"   Screening prompts loaded successfully")
            print(f"    - Reviewer system prompt: {len(screening['reviewer']['system_prompt'])} chars")
            print(f"    - Reviewer user template: {len(screening['reviewer']['user_prompt_template'])} chars")
            print(f"    - Adjudicator system prompt: {len(screening['adjudicator']['system_prompt'])} chars")
            print(f"    - Adjudicator user template: {len(screening['adjudicator']['user_prompt_template'])} chars")

            print(f"   Extraction prompts loaded successfully")
            print(f"    - Quantitative extraction: {len(extraction['quantitative_extraction'])} chars")
            print(f"    - Qualitative extraction: {len(extraction['qualitative_extraction'])} chars")
            print(f"    - Reconciliation system: {len(extraction['reconciliation_system'])} chars")
            print(f"    - Reconciliation user template: {len(extraction['reconciliation_user_template'])} chars")

        except FileNotFoundError as e:
            print(f"   Format not available: {e}")
        except Exception as e:
            print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print(" Prompt loader test complete!")
