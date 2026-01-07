"""
Output Manager for Data Extraction Results

Handles saving extraction results in multiple formats (CSV, JSON)
with dynamic format selection.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Literal, Optional
from datetime import datetime

# Type definitions
OutputFormat = Literal["csv", "json"]


def _ensure_extension(path: str, format: OutputFormat) -> str:
    """
    Ensure the file path has an extension.
    If extension exists, use it. If not, use the format parameter.

    Args:
        path: File path (may or may not have extension)
        format: Desired format (csv or json) - only used if no extension in path

    Returns:
        Path with extension
    """
    path = path.strip()

    # If path already has .csv or .json extension, use it as-is
    if path.endswith('.csv') or path.endswith('.json'):
        return path

    # No extension, add the format extension
    return f"{path}.{format}"


class OutputConfig:
    """Configuration for output file management"""

    def __init__(
        self,
        output_dir: str = "./storage",
        qualitative_format: OutputFormat = "csv",
        quantitative_format: OutputFormat = "csv",
        screening_format: OutputFormat = "json",
        base_filename: Optional[str] = None
    ):
        """
        Initialize output configuration

        Args:
            output_dir: Base directory for output files
            qualitative_format: Format for qualitative data (csv/json)
            quantitative_format: Format for quantitative data (csv/json)
            screening_format: Format for screening data (csv/json)
            base_filename: Base filename (without extension). If None, auto-generates
        """
        self.output_dir = Path(output_dir)
        self.qualitative_format = qualitative_format
        self.quantitative_format = quantitative_format
        self.screening_format = screening_format

        # Generate base filename if not provided
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"extraction_{timestamp}"
        self.base_filename = base_filename

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_qualitative_path(self) -> str:
        """Get full path for qualitative output file"""
        ext = f".{self.qualitative_format}"
        return str(self.output_dir / f"{self.base_filename}_qualitative{ext}")

    def get_quantitative_path(self) -> str:
        """Get full path for quantitative output file"""
        ext = f".{self.quantitative_format}"
        return str(self.output_dir / f"{self.base_filename}_quantitative{ext}")

    def get_screening_path(self) -> str:
        """Get full path for screening output file"""
        ext = f".{self.screening_format}"
        return str(self.output_dir / f"{self.base_filename}_screening{ext}")


def save_qualitative_data(
    output_path: str,
    qual_data: Dict[str, Any],
    paper_metadata: Dict[str, Any],
    format: OutputFormat = "csv"
) -> None:
    """
    Save qualitative data in specified format

    Args:
        output_path: Full path to output file (extension will be auto-added if missing)
        qual_data: Qualitative data dictionary
        paper_metadata: Paper metadata (paper_id, authors, year, doi, title)
        format: Output format (csv or json)
    """
    # Auto-add extension if not present
    output_path = _ensure_extension(output_path, format)

    if format == "csv":
        _save_qualitative_csv(output_path, qual_data, paper_metadata)
    elif format == "json":
        _save_qualitative_json(output_path, qual_data, paper_metadata)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_quantitative_data(
    output_path: str,
    quant_data: Any,  # Can be dict or list
    paper_metadata: Dict[str, Any],
    format: OutputFormat = "csv"
) -> int:
    """
    Save quantitative data in specified format

    Args:
        output_path: Full path to output file (extension will be auto-added if missing)
        quant_data: Quantitative data (dict or list of dicts)
        paper_metadata: Paper metadata (paper_id, authors, year, doi, title)
        format: Output format (csv or json)

    Returns:
        Number of effect sizes saved
    """
    # Auto-add extension if not present
    output_path = _ensure_extension(output_path, format)

    # Convert dict to list if needed
    if isinstance(quant_data, dict):
        # If it's a single dict, wrap it in a list
        quant_data_array = [quant_data]
    elif isinstance(quant_data, list):
        quant_data_array = quant_data
    else:
        quant_data_array = []

    if format == "csv":
        return _save_quantitative_csv(output_path, quant_data_array, paper_metadata)
    elif format == "json":
        return _save_quantitative_json(output_path, quant_data_array, paper_metadata)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _save_qualitative_csv(
    csv_path: str,
    qual_data: Dict[str, Any],
    paper_metadata: Dict[str, Any]
) -> None:
    """Save qualitative data to CSV (1 row per paper)"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Define qualitative CSV headers
    qual_headers = [
        "paper_id", "authors", "year", "doi", "title",
        "study_location", "dominant_ecosystem", "focal_taxa", "disturbance_type",
        "main_response_variable", "study_type", "niche_metrics", "specialist_vs_generalist",
        "environmental_variability", "niche_change_direction", "trait_info", "primary_climatic_driver",
        "ecosystem_context", "vulnerability_impacts", "theoretical_framework",
        "conservation_implications", "overall_comment", "needs_human_review", "overall_confidence"
    ]

    # Create file if doesn't exist
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=qual_headers).to_csv(csv_path, index=False)

    # Prepare row data
    row_data = {
        "paper_id": paper_metadata.get("paper_id"),
        "authors": paper_metadata.get("authors"),
        "year": paper_metadata.get("year"),
        "doi": paper_metadata.get("doi"),
        "title": paper_metadata.get("title"),
        **{k: qual_data.get(k) for k in qual_headers if k not in ["paper_id", "authors", "year", "doi", "title"]}
    }

    # Append row
    df = pd.read_csv(csv_path)
    new_row_df = pd.DataFrame([row_data])
    df = pd.concat([df, new_row_df], ignore_index=True)
    df.to_csv(csv_path, index=False)


def _save_qualitative_json(
    json_path: str,
    qual_data: Dict[str, Any],
    paper_metadata: Dict[str, Any]
) -> None:
    """Save qualitative data to JSON (array of objects)"""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Combine metadata and qualitative data
    record = {
        **paper_metadata,
        **qual_data,
        "extraction_date": datetime.now().isoformat()
    }

    # Load existing data or create new array
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
    else:
        data = []

    # Append new record
    data.append(record)

    # Save back to file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def _save_quantitative_csv(
    csv_path: str,
    quant_data_array: List[Dict[str, Any]],
    paper_metadata: Dict[str, Any]
) -> int:
    """
    Save quantitative effect sizes to CSV (multiple rows per paper, one per effect size)

    Returns:
        Number of effect sizes saved
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Define quantitative CSV headers
    quant_headers = [
        "paper_id", "authors", "year", "doi", "title",
        "effect_size_id", "specialization_metric", "specialization_metric_type",
        "specialist_group", "generalist_group",
        "climate_variable", "climate_variable_type", "climate_change_magnitude", "study_period",
        "response_variable", "response_variable_type",
        "effect_size_type", "effect_size", "variance_type", "variance_value", "sample_size_es",
        "direction_coded", "stat_method",
        "latitude", "longitude",
        "moderator_study_design", "transformation_applied", "notes"
    ]

    # Create file if doesn't exist
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=quant_headers).to_csv(csv_path, index=False)

    # Handle empty array
    if not quant_data_array or not isinstance(quant_data_array, list):
        return 0

    # Prepare rows (one per effect size)
    rows = []
    for i, effect_size in enumerate(quant_data_array, start=1):
        if not isinstance(effect_size, dict):
            continue

        # Generate effect_size_id if not present
        if "effect_size_id" not in effect_size or not effect_size["effect_size_id"]:
            paper_id = paper_metadata.get("paper_id", "unknown")
            effect_size["effect_size_id"] = f"{paper_id}_ES{i}"

        row_data = {
            "paper_id": paper_metadata.get("paper_id"),
            "authors": paper_metadata.get("authors"),
            "year": paper_metadata.get("year"),
            "doi": paper_metadata.get("doi"),
            "title": paper_metadata.get("title"),
            **{k: effect_size.get(k) for k in quant_headers if k not in ["paper_id", "authors", "year", "doi", "title"]}
        }
        rows.append(row_data)

    if not rows:
        return 0

    # Append all rows
    df = pd.read_csv(csv_path)
    new_rows_df = pd.DataFrame(rows)
    df = pd.concat([df, new_rows_df], ignore_index=True)
    df.to_csv(csv_path, index=False)

    return len(rows)


def _save_quantitative_json(
    json_path: str,
    quant_data_array: List[Dict[str, Any]],
    paper_metadata: Dict[str, Any]
) -> int:
    """
    Save quantitative effect sizes to JSON

    Returns:
        Number of effect sizes saved
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    if not quant_data_array or not isinstance(quant_data_array, list):
        return 0

    # Load existing data or create new array
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
    else:
        data = []

    # Create records with metadata
    for i, effect_size in enumerate(quant_data_array, start=1):
        if not isinstance(effect_size, dict):
            continue

        # Generate effect_size_id if not present
        if "effect_size_id" not in effect_size or not effect_size["effect_size_id"]:
            paper_id = paper_metadata.get("paper_id", "unknown")
            effect_size["effect_size_id"] = f"{paper_id}_ES{i}"

        record = {
            **paper_metadata,
            **effect_size,
            "extraction_date": datetime.now().isoformat()
        }
        data.append(record)

    # Save back to file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    return len(quant_data_array)


def save_screening_data(
    output_path: str,
    screening_result: Any,  # ScreeningResult object
    format: OutputFormat = "json"
) -> None:
    """
    Save screening data in specified format

    Args:
        output_path: Full path to output file
        screening_result: ScreeningResult object
        format: Output format (csv or json)
    """
    if format == "csv":
        _save_screening_csv(output_path, screening_result)
    elif format == "json":
        _save_screening_json(output_path, screening_result)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _save_screening_csv(csv_path: str, screening_result: Any) -> None:
    """Save screening result to CSV"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Convert to dict
    if hasattr(screening_result, 'model_dump'):
        data = screening_result.model_dump()
    else:
        data = dict(screening_result)

    # Flatten for CSV
    row = {
        "paper_id": data.get("paper_id"),
        "paper_title": data.get("paper_title"),
        "paper_authors": data.get("paper_authors"),
        "paper_year": data.get("paper_year"),
        "final_decision": data.get("final_decision"),
        "final_exclusion_reason": data.get("final_exclusion_reason"),
        "needs_human_review": data.get("needs_human_review"),
        "adjudication_notes": data.get("adjudication_notes"),
        "screening_date": data.get("screening_date")
    }

    # Create or append to CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    new_row_df = pd.DataFrame([row])
    df = pd.concat([df, new_row_df], ignore_index=True)
    df.to_csv(csv_path, index=False)


def _save_screening_json(json_path: str, screening_result: Any) -> None:
    """Save screening result to JSON"""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Convert to dict
    if hasattr(screening_result, 'model_dump'):
        data = screening_result.model_dump(mode="json")
    else:
        data = dict(screening_result)

    # Load existing or create new
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            records = json.load(f)
            if not isinstance(records, list):
                records = [records]
    else:
        records = []

    records.append(data)

    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2)
