"""
Prompt Configuration

Select which prompt format to use across the entire app.
"""

from typing import Literal

# Choose preferred prompt format: 'txt', 'py', or 'json'
# Default: 'py' (prompts as code)
#
# Options:
#   'py'   - Python modules (syntax highlighting, version control as code)
#   'txt'  - Plain text files (easiest to edit, works everywhere)
#   'json' - JSON format (structured, machine-readable)
#
# To change format:
# 1. You can update PROMPT_FORMAT below
# 2. Ensure corresponding files exist in screening_agents/ and extraction_agents/


PROMPT_FORMAT: Literal['txt', 'py', 'json'] = 'py'
