"""
Streamlit UI for Paper Screening and Data Extraction
"""

import streamlit as st
import os
import sys
from pathlib import Path
import pandas as pd
import json
import base64
from typing import Any, Dict, List, Optional, Tuple

# Add server directory to path
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from data_extraction import (
    PDFProcessor,
    DataExtractionPipeline,
    create_llm_provider,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    ExtractionResult,
)
from output_manager import (
    OutputConfig,
    save_qualitative_data,
    save_quantitative_data,
    save_screening_data
)

OLLAMA_LOCAL_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# CSV Headers for meta-analysis data
CSV_HEADERS = [
    "study_location", "study_area", "dominant_ecosystem", "focal_taxa", "disturbance_type", "main_response_variable", "methodology",
    "niche_metrics", "specialist_vs_generalist", "environmental_variability", "niche_change_direction", "trait_info", "primary_climatic_driver",
    "ecosystem_context", "vulnerability_impacts", "theoretical_framework", "conservation_implications", "overall_comment",
    "metric", "effect_size", "effect_size_unit", "sample_size", "variance_type", "variance_value", "method", "represents",
    "temperature_change", "precipitation_change", "other_climatic_drivers", "other_climatic_drivers_value", "latitude", "longitude",
    "notes", "Filename", "Article Title", "Publication Year", "paper_authors", "Justification", "needs_human_review", "num_conflicts", "overall_confidence"
]


def default_model_for(provider: str, *, vision: bool = False) -> str:
    if provider == "anthropic":
        return "claude-3-5-haiku-20241022"
    if provider == "openai":
        return "gpt-4o"
    if provider == "ollama":
        return DEFAULT_VISION_MODEL if vision else DEFAULT_TEXT_MODEL
    return DEFAULT_TEXT_MODEL


def build_pipeline(
    provider: str,
    model: str,
    *,
    use_cloud: bool,
    enable_vision: bool,
    use_custom_vision: bool,
    vision_provider: Optional[str],
    vision_model: Optional[str],
    vision_use_cloud: bool,
    save_vision_images: bool
) -> DataExtractionPipeline:
    pdf_processor = PDFProcessor()
    resolved_model = (model or "").strip() or default_model_for(provider)
    text_llm = create_llm_provider(
        provider,
        resolved_model,
        ollama_base_url=OLLAMA_LOCAL_BASE_URL,
        ollama_cloud=use_cloud if provider == "ollama" else False
    )

    vision_llm = None
    if enable_vision:
        resolved_vision_provider = None
        resolved_vision_model = None

        if use_custom_vision:
            resolved_vision_provider = vision_provider or provider
            resolved_vision_model = (vision_model or "").strip() or default_model_for(resolved_vision_provider, vision=True)
        elif not text_llm.supports_vision and provider == "ollama":
            resolved_vision_provider = "ollama"
            resolved_vision_model = DEFAULT_VISION_MODEL

        if resolved_vision_provider:
            if not resolved_vision_model:
                resolved_vision_model = default_model_for(resolved_vision_provider, vision=True)

            same_model = (
                resolved_vision_provider == provider
                and resolved_vision_model == resolved_model
            )
            if not same_model:
                vision_llm = create_llm_provider(
                    resolved_vision_provider,
                    resolved_vision_model,
                    ollama_base_url=OLLAMA_LOCAL_BASE_URL,
                    ollama_cloud=vision_use_cloud if resolved_vision_provider == "ollama" else False
                )

    return DataExtractionPipeline(
        text_llm_provider=text_llm,
        pdf_processor=pdf_processor,
        vision_llm_provider=vision_llm,
        use_vision=enable_vision,
        save_extracted_images=save_vision_images
    )


def extraction_payload(result: ExtractionResult, filename: str, screening_result=None) -> Dict[str, Any]:
    """
    Build extraction payload with combined data and metadata.

    Args:
        result: ExtractionResult object
        filename: Name of the PDF file
        screening_result: Optional ScreeningResult object to extract metadata from

    Returns:
        Dictionary with extraction data and metadata
    """
    payload = result.model_dump(mode="json")

    # Build combined_data with all metadata
    # Handle both list (new array format) and dict (legacy) for quantitative data
    if isinstance(result.final_quantitative_data, list):
        # New array format: use first effect size for legacy combined_data, or empty dict
        quant_data_for_combined = result.final_quantitative_data[0] if result.final_quantitative_data else {}
    else:
        # Legacy dict format
        quant_data_for_combined = result.final_quantitative_data

    combined_data = {
        **result.final_qualitative_data,
        **quant_data_for_combined,
        "Filename": filename,
        "needs_human_review": result.needs_human_review,
        "num_conflicts": len(result.conflicts),
        "overall_confidence": result.overall_confidence
    }

    # Add screening metadata if available
    if screening_result:
        combined_data["Article Title"] = screening_result.paper_title
        combined_data["Publication Year"] = screening_result.paper_year
        combined_data["paper_authors"] = screening_result.paper_authors

        # Build justification from screening decision
        justification_parts = []
        if screening_result.final_decision:
            justification_parts.append(f"Decision: {screening_result.final_decision.value}")
        if screening_result.final_exclusion_reason:
            justification_parts.append(f"Reason: {screening_result.final_exclusion_reason.value}")
        if screening_result.adjudication_notes:
            justification_parts.append(f"Notes: {screening_result.adjudication_notes}")
        combined_data["Justification"] = "; ".join(justification_parts)
    else:
        # Use extraction metadata if no screening data available
        combined_data["Article Title"] = result.paper_title or ""
        combined_data["Publication Year"] = result.paper_year or ""
        combined_data["paper_authors"] = result.paper_authors or ""
        combined_data["Justification"] = ""

    payload["combined_data"] = combined_data
    payload["num_conflicts"] = len(result.conflicts)
    return payload


def save_to_csv(csv_path: str, data_dict: Dict[str, Any]) -> None:
    """
    Append a row of data to CSV file. Creates file if it doesn't exist.
    LEGACY FUNCTION - Use save_qualitative_csv and save_quantitative_csv instead.

    Args:
        csv_path: Path to the CSV file
        data_dict: Dictionary containing the data to save (combined_data format)
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Create empty CSV with headers if it doesn't exist
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=CSV_HEADERS).to_csv(csv_path, index=False)

    # Read existing CSV
    df = pd.read_csv(csv_path)

    # Create a row dict with all CSV headers, filling missing fields with None
    row_data = {header: data_dict.get(header, None) for header in CSV_HEADERS}

    # Append new row
    new_row_df = pd.DataFrame([row_data])
    df = pd.concat([df, new_row_df], ignore_index=True)

    # Save back to CSV
    df.to_csv(csv_path, index=False)


def save_qualitative_csv(csv_path: str, qual_data: Dict[str, Any], paper_metadata: Dict[str, Any]) -> None:
    """
    Save qualitative study metadata to CSV (1 row per paper).

    Args:
        csv_path: Path to qualitative CSV file
        qual_data: Qualitative data dictionary from extraction
        paper_metadata: Paper metadata (paper_id, authors, year, doi, title)
    """
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


def save_quantitative_csv(csv_path: str, quant_data_array: List[Dict[str, Any]], paper_metadata: Dict[str, Any]) -> int:
    """
    Save quantitative effect sizes to CSV (multiple rows per paper, one per effect size).

    Args:
        csv_path: Path to quantitative CSV file
        quant_data_array: List of effect size dictionaries (array from extraction)
        paper_metadata: Paper metadata (paper_id, authors, year, doi, title)

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
    for effect_size in quant_data_array:
        if not isinstance(effect_size, dict):
            continue

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


# Page config
st.set_page_config(
    page_title="Meta-Analysis Data Extractor",
    page_icon="📄",
    layout="wide"
)

st.title("Meta-Analysis Paper Screening & Data Extraction")
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header(" Configuration")

    provider = st.selectbox(
        "Select LLM Provider",
        ["ollama", "anthropic", "openai"],
        help="Choose which LLM provider to use"
    )

    api_key = ""
    model = ""
    use_cloud = False

    if provider == "ollama":
        use_local = st.checkbox("Use Local Ollama (requires local installation)", value=False)
        if use_local:
            st.warning("⚠️ Local Ollama only works when running this app locally. It will not work on Streamlit Cloud.")
            st.info("Using local Ollama installation")
            api_key = "local"
            use_cloud = False
        else:
            api_key = st.text_input(
                "Ollama Cloud API Key",
                type="password",
                help="Enter your Ollama Cloud API key"
            )
            if api_key:
                os.environ["OLLAMA_API_KEY"] = api_key
                os.environ["OLLAMA_CLOUD"] = "true"
            use_cloud = True
        model = st.text_input("Model", value=DEFAULT_TEXT_MODEL)
    elif provider == "anthropic":
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Enter your Anthropic API key (sk-ant-...)"
        )
        model = st.text_input("Model", value="claude-3-5-haiku-20241022")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
    elif provider == "openai":
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key (sk-...)"
        )
        model = st.text_input("Model", value="gpt-4o")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")
    st.subheader(" Vision / Figures")

    enable_vision = st.checkbox("Enable figure extraction (vision LLM)", value=True)
    save_vision_images = st.checkbox(
        "Save extracted figure images",
        value=False,
        disabled=not enable_vision
    )

    default_separate = provider == "ollama"
    use_custom_vision = st.checkbox(
        "Use separate vision model",
        value=default_separate,
        disabled=not enable_vision
    )

    vision_provider = None
    vision_model = None
    vision_use_cloud = False

    if enable_vision and use_custom_vision:
        provider_options = ["ollama", "anthropic", "openai"]
        default_index = provider_options.index("ollama") if provider != "anthropic" else provider_options.index("anthropic")
        vision_provider = st.selectbox(
            "Vision provider",
            provider_options,
            index=default_index,
            help="Choose which LLM handles figure analysis",
            key="vision_provider"
        )
        suggested_model = default_model_for(vision_provider, vision=True)
        vision_model = st.text_input(
            "Vision model",
            value=suggested_model,
            key="vision_model"
        )

        if vision_provider == "ollama":
            vision_api_key = st.text_input(
                "Ollama Cloud API Key (for vision)",
                type="password",
                help="Enter your Ollama Cloud API key for vision",
                key="vision_ollama_api"
            )
            vision_use_local = st.checkbox(
                "Use Local Ollama instead (ignores API key above)",
                value=False,
                key="vision_use_local"
            )
            if vision_use_local:
                st.warning("⚠️ Local Ollama only works when running this app locally. It will not work on Streamlit Cloud.")
                vision_use_cloud = False
            else:
                if vision_api_key:
                    os.environ["OLLAMA_API_KEY"] = vision_api_key
                    os.environ["OLLAMA_CLOUD"] = "true"
                vision_use_cloud = True
        elif vision_provider == "anthropic" and provider != "anthropic":
            vision_api_key = st.text_input(
                "Vision Anthropic API Key",
                type="password",
                help="Needed only if different from text provider",
                key="vision_anthropic_api"
            )
            if vision_api_key:
                os.environ["ANTHROPIC_API_KEY"] = vision_api_key
        elif vision_provider == "openai" and provider != "openai":
            vision_api_key = st.text_input(
                "Vision OpenAI API Key",
                type="password",
                help="Needed only if different from text provider",
                key="vision_openai_api"
            )
            if vision_api_key:
                os.environ["OPENAI_API_KEY"] = vision_api_key
    else:
        save_vision_images = False

text_use_cloud = use_cloud if provider == "ollama" else False
pipeline_kwargs = dict(
    provider=provider,
    model=model,
    use_cloud=text_use_cloud,
    enable_vision=enable_vision,
    use_custom_vision=use_custom_vision,
    vision_provider=vision_provider if use_custom_vision else None,
    vision_model=vision_model if use_custom_vision else None,
    vision_use_cloud=vision_use_cloud if (use_custom_vision and vision_provider == "ollama") else False,
    save_vision_images=save_vision_images
)

# Main content
tab1, tab2, tab3 = st.tabs([" Paper Screening", " Data Extraction", " Batch Processing"])

# Tab 1: Paper Screening
with tab1:
    st.header("Screen Individual Paper")

    col1, col2 = st.columns([2, 1])

    with col1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload PDF Paper",
            type=['pdf'],
            help="Upload a PDF file to screen"
        )

    with col2:
        st.markdown("### Screening Criteria")
        st.info("""
        Papers are screened for:
        - Ecological niche relevance
        - Climate change context
        - Quantitative data availability
        """)

        st.markdown("### Save Options")
        auto_save_screening = st.checkbox(
            "Auto-save screening results",
            value=True,
            help="Automatically save screening results to JSON file",
            key="auto_save_screening"
        )
        screening_output_path = st.text_input(
            "Screening Output Path (JSON)",
            value="./storage/screening_results.json",
            help="Path where screening results will be saved",
            key="screening_output_path"
        )
        st.info("ℹ️ Note: CSV export is only available for extraction data. Screening results are saved to JSON only.")

    if st.button(" Screen Paper", type="primary"):
        if not api_key or api_key == "":
            st.error("Please enter API key in the sidebar")
        elif not uploaded_file:
            st.error("Please upload a PDF file")
        else:
            # Save uploaded file to temp location
            pdf_path = f"/tmp/{uploaded_file.name}"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())

            with st.spinner("Screening paper... This may take a minute."):
                try:
                    screening_pipeline = build_pipeline(
                        provider=provider,
                        model=model,
                        use_cloud=text_use_cloud,
                        enable_vision=False,
                        use_custom_vision=False,
                        vision_provider=None,
                        vision_model=None,
                        vision_use_cloud=False,
                        save_vision_images=False
                    )
                    screening_result = screening_pipeline.screen_paper(pdf_path)

                    metadata = {
                        "title": screening_result.paper_title,
                        "authors": screening_result.paper_authors,
                        "year": screening_result.paper_year,
                        "pdf_path": screening_result.pdf_path
                    }

                    reviewer1 = screening_result.reviewer1
                    reviewer2 = screening_result.reviewer2

                    if auto_save_screening:
                        try:
                            payload = screening_result.model_dump(mode="json")
                            payload["paper"] = Path(pdf_path).name
                            payload["screened_at"] = pd.Timestamp.now().isoformat()

                            if os.path.exists(screening_output_path):
                                with open(screening_output_path, 'r', encoding='utf-8') as f:
                                    all_screenings = json.load(f)
                                    if not isinstance(all_screenings, list):
                                        all_screenings = [all_screenings]
                            else:
                                all_screenings = []
                                os.makedirs(os.path.dirname(screening_output_path), exist_ok=True)

                            all_screenings.append(payload)

                            with open(screening_output_path, 'w', encoding='utf-8') as f:
                                json.dump(all_screenings, f, indent=2, ensure_ascii=False)

                            st.success(f" Screening Complete! Auto-saved to {screening_output_path}")
                        except Exception as save_err:
                            st.warning(f" Screening complete but auto-save failed: {save_err}")
                            st.success(" Screening Complete!")
                    else:
                        st.success(" Screening Complete!")

                    # Final decision with reason
                    decision = screening_result.final_decision.value
                    final_reason = screening_result.final_exclusion_reason.value if screening_result.final_exclusion_reason else None

                    if decision == "INCLUDE":
                        st.success(f"### 🟢 Decision: {decision}")
                        st.info(f"**Reason:** Paper meets all inclusion criteria")
                    elif decision == "EXCLUDE":
                        reason_text = f" - Reason: {final_reason}" if final_reason else ""
                        st.error(f"### 🔴 Decision: {decision}{reason_text}")
                        if final_reason:
                            # Map exclusion codes to descriptions
                            exclusion_codes = {
                                "E1": "No quantitative specialization metric",
                                "E2": "No climate quantification (either not climate-related OR no quantified climate data)",
                                "E3": "No specialist vs generalist comparison",
                                "E4": "Insufficient data for effect size extraction",
                                "E5": "Reserved (tracked as moderator)",
                                "E6": "Reserved (can be flagged for human review)",
                                "E7": "Reserved",
                                "E8": "Duplicate study (same paper already screened)",
                                "E9": "Inaccessible (PDF corrupted, unreadable, or missing text)",
                                "E10": "Other (catch-all for miscellaneous reasons)"
                            }
                            st.warning(f"**{final_reason}:** {exclusion_codes.get(final_reason, 'Unknown reason')}")
                    else:
                        st.warning(f"### 🟡 Decision: {decision}")
                        st.info(f"**Reason:** One or more criteria unclear - needs discussion")

                    # Metadata
                    with st.expander(" Paper Metadata", expanded=True):
                        st.json(metadata)

                    # Reviewer assessments
                    col1, col2 = st.columns(2)

                    with col1:
                        with st.expander("👤 Reviewer 1 Assessment"):
                            st.write(f"**Decision:** {reviewer1.decision.value}")
                            st.write(f"**Confidence:** {reviewer1.confidence}")
                            if reviewer1.exclusion_reason:
                                st.write(f"**Exclusion Reason:** {reviewer1.exclusion_reason.value}")
                            st.write(f"**Notes:** {reviewer1.notes}")

                    with col2:
                        with st.expander("👤 Reviewer 2 Assessment"):
                            st.write(f"**Decision:** {reviewer2.decision.value}")
                            st.write(f"**Confidence:** {reviewer2.confidence}")
                            if reviewer2.exclusion_reason:
                                st.write(f"**Exclusion Reason:** {reviewer2.exclusion_reason.value}")
                            st.write(f"**Notes:** {reviewer2.notes}")

                    with st.expander("⚖️ Adjudication Details"):
                        st.write(f"**Notes:** {screening_result.adjudication_notes}")
                        st.write(f"**Needs Human Review:** {screening_result.needs_human_review}")
                        if screening_result.final_exclusion_reason:
                            st.write(f"**Exclusion Reason:** {screening_result.final_exclusion_reason.value}")

                except Exception as e:
                    st.error(f"Error during screening: {str(e)}")
                    st.exception(e)

# Tab 2: Data Extraction
with tab2:
    st.header("Extract Data from Paper")

    col1, col2 = st.columns([2, 1])

    with col1:
        # File uploader for extraction
        uploaded_file_extract = st.file_uploader(
            "Upload PDF Paper for Extraction",
            type=['pdf'],
            key="extract_upload",
            help="Upload a PDF file to extract data"
        )

    with col2:
        st.markdown("### Extraction Types")
        extract_quantitative = st.checkbox("Quantitative Data", value=True)
        extract_qualitative = st.checkbox("Qualitative Data", value=True)

        st.markdown("### Save Options")
        auto_save_json = st.checkbox(
            "Auto-save to file",
            value=True,
            help="Automatically save results to JSON and CSV formats",
            key="auto_save_json"
        )

        # JSON output (always JSON)
        json_output_path = st.text_input(
            "📄 JSON Output Path (Full Details)",
            value="./storage/extractions.json",
            help="Complete extraction results with dual extractor data and reconciliation details",
            key="json_output_path"
        )

        # Qualitative output with format selection
        st.markdown("**Qualitative Data Output**")
        col_q1, col_q2 = st.columns([3, 1])
        with col_q1:
            qualitative_path = st.text_input(
                "File Path",
                value="./storage/qualitative_data.csv",
                help="Study metadata (1 row per paper)",
                key="qualitative_path_extract",
                label_visibility="collapsed"
            )
        with col_q2:
            qual_format_extract = st.radio(
                "Format",
                ["csv", "json"],
                index=0,
                key="qual_format_extract",
                horizontal=True,
                label_visibility="collapsed"
            )

        # Quantitative output with format selection
        st.markdown("**Quantitative Data Output**")
        col_qt1, col_qt2 = st.columns([3, 1])
        with col_qt1:
            quantitative_path = st.text_input(
                "File Path",
                value="./storage/quantitative_effect_sizes.csv",
                help="Effect size data (multiple rows per paper)",
                key="quantitative_path_extract",
                label_visibility="collapsed"
            )
        with col_qt2:
            quant_format_extract = st.radio(
                "Format",
                ["csv", "json"],
                index=0,
                key="quant_format_extract",
                horizontal=True,
                label_visibility="collapsed"
            )

    # Initialize session state for extraction results
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = None
    if 'extraction_full_result' not in st.session_state:
        st.session_state.extraction_full_result = None
    if 'extraction_pdf_name' not in st.session_state:
        st.session_state.extraction_pdf_name = None

    if st.button(" Extract Data", type="primary"):
        if not api_key or api_key == "":
            st.error("Please enter API key in the sidebar")
        elif not uploaded_file_extract:
            st.error("Please upload a PDF file")
        else:
            # Save uploaded file to temp location
            pdf_path = f"/tmp/{uploaded_file_extract.name}"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file_extract.read())

            with st.spinner("Extracting data... This may take a few minutes."):
                try:
                    pipeline = build_pipeline(**pipeline_kwargs)
                    if pipeline_kwargs["enable_vision"] and pipeline.figure_extractor is None:
                        st.info("Vision extraction disabled for this run (selected models do not support vision).")
                    extraction_result = pipeline.extract_data(
                        pdf_path,
                        extract_quantitative=extract_quantitative,
                        extract_qualitative=extract_qualitative
                    )

                    quant_data1 = extraction_result.extractor1.quantitative_data or {}
                    quant_data2 = extraction_result.extractor2.quantitative_data or {}
                    qual_data1 = extraction_result.extractor1.qualitative_data or {}
                    qual_data2 = extraction_result.extractor2.qualitative_data or {}

                    reconciliation = {
                        "final_quantitative_data": extraction_result.final_quantitative_data,
                        "final_qualitative_data": extraction_result.final_qualitative_data,
                        "auto_resolved": extraction_result.auto_resolved,
                        "conflicts": extraction_result.conflicts,
                        "overall_confidence": extraction_result.overall_confidence,
                        "needs_human_review": extraction_result.needs_human_review,
                        "reconciliation_notes": extraction_result.reconciliation_notes
                    }

                    # Store results in session state
                    # Handle both list (new array format) and dict (legacy) for quantitative data
                    quant_data = reconciliation.get("final_quantitative_data", {})
                    if isinstance(quant_data, list):
                        # New array format: use first effect size for combined_data, or empty dict
                        quant_data_for_combined = quant_data[0] if quant_data else {}
                    else:
                        # Legacy dict format
                        quant_data_for_combined = quant_data

                    combined_data = {
                        **reconciliation.get("final_qualitative_data", {}),
                        **quant_data_for_combined
                    }
                    combined_data["Filename"] = Path(pdf_path).name
                    combined_data["extracted_at"] = pd.Timestamp.now().isoformat()
                    combined_data["needs_human_review"] = extraction_result.needs_human_review
                    combined_data["num_conflicts"] = len(extraction_result.conflicts)

                    st.session_state.extraction_results = combined_data
                    st.session_state.extraction_full_result = extraction_result  # Store full result for download
                    st.session_state.extraction_pdf_name = Path(pdf_path).stem

                    # Auto-save to JSON and CSV if enabled
                    if auto_save_json:
                        save_messages = []
                        try:
                            # Load existing data if file exists
                            if os.path.exists(json_output_path):
                                with open(json_output_path, 'r', encoding='utf-8') as f:
                                    all_extractions = json.load(f)
                                    if not isinstance(all_extractions, list):
                                        all_extractions = [all_extractions]
                            else:
                                all_extractions = []
                                os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

                            # Append new extraction (save FULL extraction result with dual extractor data)
                            all_extractions.append(extraction_result.model_dump(mode="json"))

                            # Save back to file
                            with open(json_output_path, 'w', encoding='utf-8') as f:
                                json.dump(all_extractions, f, indent=2, ensure_ascii=False)

                            save_messages.append(f"JSON: {json_output_path}")
                        except Exception as e:
                            st.warning(f"⚠️ JSON auto-save failed: {str(e)}")

                        # Save qualitative and quantitative data using output manager
                        try:
                            # Prepare paper metadata
                            paper_metadata = {
                                "paper_id": extraction_result.paper_id,
                                "authors": extraction_result.paper_authors or "",
                                "year": extraction_result.paper_year or "",
                                "doi": extraction_result.paper_doi or "",
                                "title": extraction_result.paper_title or ""
                            }

                            # Save qualitative data
                            qual_data_with_meta = {
                                **extraction_result.final_qualitative_data,
                                "needs_human_review": extraction_result.needs_human_review,
                                "overall_confidence": extraction_result.overall_confidence
                            }
                            save_qualitative_data(
                                qualitative_path,
                                qual_data_with_meta,
                                paper_metadata,
                                format=qual_format_extract
                            )
                            save_messages.append(f"Qualitative {qual_format_extract.upper()}: {qualitative_path}")

                            # Save quantitative data
                            quant_data = extraction_result.final_quantitative_data
                            num_effect_sizes = save_quantitative_data(
                                quantitative_path,
                                quant_data,
                                paper_metadata,
                                format=quant_format_extract
                            )
                            save_messages.append(f"Quantitative {quant_format_extract.upper()}: {quantitative_path} ({num_effect_sizes} effect size(s))")

                        except Exception as e:
                            st.warning(f"⚠️ Data auto-save failed: {str(e)}")

                        if save_messages:
                            st.success(f"✅ Extraction Complete! Auto-saved to:\n- " + "\n- ".join(save_messages))
                        else:
                            st.warning("⚠️ Extraction complete but auto-save failed")
                    else:
                        st.success("✅ Extraction Complete!")

                    # Show overall confidence and status
                    overall_conf = extraction_result.overall_confidence
                    conf_color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(overall_conf, "⚪")
                    st.info(f"{conf_color} **Overall Confidence**: {overall_conf}")

                    # Show auto-resolved summary
                    num_auto = len(extraction_result.auto_resolved)
                    if num_auto > 0:
                        st.success(f" **Auto-resolved**: {num_auto} semantic equivalence(s) detected and merged")

                    # Show conflict warning if needed
                    if extraction_result.needs_human_review:
                        st.warning(f"⚠️ **Needs Human Review**: {combined_data['num_conflicts']} conflict(s) detected between extractors")

                    # Display final reconciled data
                    with st.expander(" Final Reconciled Data", expanded=True):
                        st.json(combined_data)

                    # Display auto-resolved fields
                    if extraction_result.auto_resolved:
                        with st.expander(f" Auto-Resolved Fields ({len(extraction_result.auto_resolved)})", expanded=False):
                            for resolved in extraction_result.auto_resolved:
                                st.success(f"**Field:** {resolved.get('field', 'unknown')}")
                                st.write(f"- **Extractor 1:** {resolved.get('extractor1_value', 'N/A')}")
                                st.write(f"- **Extractor 2:** {resolved.get('extractor2_value', 'N/A')}")
                                st.write(f"- **Resolved to:** {resolved.get('resolved_value', 'N/A')}")
                                st.write(f"- **Confidence:** {resolved.get('confidence', 'N/A')}")
                                st.write(f"- **Reason:** {resolved.get('reason', 'N/A')}")
                                st.markdown("---")

                    # Display conflicts if any
                    if extraction_result.conflicts:
                        with st.expander(f"⚠️ Conflicts ({len(extraction_result.conflicts)})", expanded=False):
                            for conflict in extraction_result.conflicts:
                                st.warning(f"**Field:** {conflict.get('field', 'unknown')}")
                                st.write(f"- **Extractor 1:** {conflict.get('extractor1_value', 'N/A')}")
                                st.write(f"- **Extractor 2:** {conflict.get('extractor2_value', 'N/A')}")
                                st.write(f"- **Type:** {conflict.get('conflict_type', 'N/A')}")
                                st.write(f"- **Confidence:** {conflict.get('confidence', 'N/A')}")
                                st.write(f"- **Recommendation:** {conflict.get('recommendation', 'N/A')}")
                                st.markdown("---")

                    # Display individual extractor results for comparison
                    col1, col2 = st.columns(2)

                    with col1:
                        with st.expander(" Extractor 1 Results", expanded=False):
                            st.write("**Quantitative:**")
                            st.json(quant_data1)
                            st.write("**Qualitative:**")
                            st.json(qual_data1)

                    with col2:
                        with st.expander(" Extractor 2 Results", expanded=False):
                            st.write("**Quantitative:**")
                            st.json(quant_data2)
                            st.write("**Qualitative:**")
                            st.json(qual_data2)

                except Exception as e:
                    st.error(f"Error during extraction: {str(e)}")
                    st.exception(e)

    # Download button (outside the extraction button block, persists after extraction)
    if st.session_state.extraction_results is not None:
        download_source = getattr(st.session_state, 'extraction_full_result', None)
        download_data = download_source.model_dump(mode="json") if download_source else st.session_state.extraction_results

        st.download_button(
            label=" Download Full Extraction (with dual extractor data)",
            data=json.dumps(download_data, indent=2, ensure_ascii=False),
            file_name=f"extracted_data_full_{st.session_state.extraction_pdf_name}.json",
            mime="application/json",
            key="download_extraction_json"
        )

# Tab 3: Batch Processing
with tab3:
    st.header("Batch Process Multiple Papers")

    # Multi-file uploader for batch processing
    uploaded_batch_files = st.file_uploader(
        "Upload PDF Papers for Batch Processing",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload multiple PDF files to process in batch",
        key="batch_upload"
    )

    # Store uploaded files in session state to persist them
    if uploaded_batch_files:
        # Save uploaded files to temp directory and track them
        batch_pdf_paths = {}
        for uploaded_file in uploaded_batch_files:
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            batch_pdf_paths[uploaded_file.name] = temp_path
        st.session_state.batch_pdf_paths = batch_pdf_paths
        st.success(f"Uploaded {len(uploaded_batch_files)} PDF files")

    # Get list of available PDFs from session state
    all_pdfs = list(st.session_state.get('batch_pdf_paths', {}).keys())

    if all_pdfs:
        col1, col2 = st.columns([2, 1])

        with col1:
            batch_mode = st.radio(
                "Processing Mode",
                ["Screen Only", "Extract Only", "Full Pipeline (Screen + Extract)"]
            )

            # Paper selection method
            st.markdown("### Paper Selection")
            selection_method = st.radio(
                "How to select papers?",
                ["First N papers", "Specific papers (multiselect)", "All papers"],
                help="Choose how to select papers for batch processing"
            )

            papers_to_process = []

            if selection_method == "First N papers":
                limit = st.number_input(
                    "Process first N papers",
                    min_value=1,
                    max_value=len(all_pdfs),
                    value=min(5, len(all_pdfs)),
                    help="Select the first N papers from uploaded files"
                )
                papers_to_process = all_pdfs[:limit]
                st.info(f"Will process first {len(papers_to_process)} paper(s)")

            elif selection_method == "Specific papers (multiselect)":
                selected_papers = st.multiselect(
                    "Select specific papers to process",
                    options=all_pdfs,
                    default=[all_pdfs[0]] if all_pdfs else [],
                    help="Choose one or more specific papers"
                )
                papers_to_process = selected_papers
                if papers_to_process:
                    st.info(f"Will process {len(papers_to_process)} selected paper(s)")

            else:  # All papers
                papers_to_process = all_pdfs
                st.info(f"Will process all {len(all_pdfs)} papers")

            # Show selected papers
            if papers_to_process:
                with st.expander(f" Selected Papers ({len(papers_to_process)})", expanded=False):
                    for i, paper in enumerate(papers_to_process, 1):
                        st.write(f"{i}. {paper}")

        with col2:
            st.markdown("### Save Options")
            auto_save_batch = st.checkbox(
                "Auto-save batch results",
                value=True,
                help="Automatically save batch processing results",
                key="auto_save_batch"
            )

            # JSON output (always JSON)
            batch_output_path = st.text_input(
                "📄 JSON Output Path (Full Details)",
                value="./storage/batch_results.json",
                help="Complete batch metadata and extraction results",
                key="batch_output_path"
            )

            # Qualitative output with format selection
            st.markdown("**Qualitative Data Output**")
            col_bq1, col_bq2 = st.columns([3, 1])
            with col_bq1:
                batch_qualitative_path = st.text_input(
                    "File Path",
                    value="./storage/batch_qualitative_data.csv",
                    help="Study metadata (1 row per paper)",
                    key="batch_qualitative_path",
                    label_visibility="collapsed"
                )
            with col_bq2:
                qualitative_format = st.radio(
                    "Format",
                    ["csv", "json"],
                    index=0,
                    key="qual_format_batch",
                    horizontal=True,
                    label_visibility="collapsed"
                )

            # Quantitative output with format selection
            st.markdown("**Quantitative Data Output**")
            col_bqt1, col_bqt2 = st.columns([3, 1])
            with col_bqt1:
                batch_quantitative_path = st.text_input(
                    "File Path",
                    value="./storage/batch_quantitative_effect_sizes.csv",
                    help="Effect size data (multiple rows per paper)",
                    key="batch_quantitative_path",
                    label_visibility="collapsed"
                )
            with col_bqt2:
                quantitative_format = st.radio(
                    "Format",
                    ["csv", "json"],
                    index=0,
                    key="quant_format_batch",
                    horizontal=True,
                    label_visibility="collapsed"
                )

        if st.button(" Start Batch Processing", type="primary"):
            if not api_key or api_key == "":
                st.error("Please enter API key in the sidebar")
            elif not papers_to_process:
                st.error("Please select at least one paper to process")
            else:
                st.write(f"Processing {len(papers_to_process)} papers...")

                progress_bar = st.progress(0)
                status_text = st.empty()

                # Results tracking
                results_summary = {
                    "processed": 0,
                    "included": 0,
                    "excluded": 0,
                    "discuss": 0,
                    "errors": 0,
                    "details": []
                }

                try:
                    pipeline = build_pipeline(**pipeline_kwargs)
                except Exception as e:
                    st.error(f"Failed to initialize pipeline: {e}")
                    st.stop()

                def run_dual_extraction(target_pdf: str, screening=None) -> Dict[str, Any]:
                    """Run extraction and build payload with optional screening metadata."""
                    extraction = pipeline.extract_data(target_pdf)
                    payload = extraction_payload(extraction, Path(target_pdf).name, screening_result=screening)
                    payload["needs_human_review"] = extraction.needs_human_review
                    return payload

                # Results display area
                results_container = st.container()

                # Process each paper
                for idx, paper in enumerate(papers_to_process):
                    status_text.text(f"Processing {idx + 1}/{len(papers_to_process)}: {paper}")

                    # Get path from session state (uploaded files)
                    pdf_path = st.session_state.batch_pdf_paths.get(paper)
                    paper_result = {"paper": paper, "status": "processing"}

                    try:
                        # Screen Only Mode
                        if batch_mode == "Screen Only":
                            screening = pipeline.screen_paper(pdf_path)
                            paper_result["screening"] = screening.model_dump(mode="json")

                            decision = screening.final_decision.value
                            paper_result["decision"] = decision
                            paper_result["exclusion_reason"] = screening.final_exclusion_reason.value if screening.final_exclusion_reason else None
                            paper_result["adjudication_notes"] = screening.adjudication_notes
                            paper_result["needs_human_review"] = screening.needs_human_review
                            paper_result["reviewer1"] = screening.reviewer1.model_dump(mode="json")
                            paper_result["reviewer2"] = screening.reviewer2.model_dump(mode="json")
                            paper_result["status"] = "success"

                            # Update counters
                            if decision == "INCLUDE":
                                results_summary["included"] += 1
                            elif decision == "EXCLUDE":
                                results_summary["excluded"] += 1
                            else:
                                results_summary["discuss"] += 1

                        # Extract Only Mode
                        elif batch_mode == "Extract Only":
                            extraction_data = run_dual_extraction(pdf_path, screening=None)
                            paper_result["extraction"] = extraction_data
                            paper_result["combined_data"] = extraction_data["combined_data"]
                            paper_result["status"] = "success"

                        # Full Pipeline Mode
                        elif batch_mode == "Full Pipeline (Screen + Extract)":
                            screening = pipeline.screen_paper(pdf_path)
                            paper_result["screening"] = screening.model_dump(mode="json")

                            decision = screening.final_decision.value
                            paper_result["decision"] = decision
                            paper_result["exclusion_reason"] = screening.final_exclusion_reason.value if screening.final_exclusion_reason else None
                            paper_result["adjudication_notes"] = screening.adjudication_notes
                            paper_result["needs_human_review"] = screening.needs_human_review
                            paper_result["reviewer1"] = screening.reviewer1.model_dump(mode="json")
                            paper_result["reviewer2"] = screening.reviewer2.model_dump(mode="json")

                            if decision == "INCLUDE":
                                results_summary["included"] += 1
                                # Pass screening results to extraction for metadata
                                extraction_data = run_dual_extraction(pdf_path, screening=screening)
                                paper_result["extraction"] = extraction_data
                                paper_result["combined_data"] = extraction_data["combined_data"]
                            elif decision == "EXCLUDE":
                                results_summary["excluded"] += 1
                                paper_result["extraction"] = None
                            else:
                                results_summary["discuss"] += 1
                                paper_result["extraction"] = None

                            paper_result["status"] = "success"

                        results_summary["processed"] += 1

                    except Exception as e:
                        paper_result["status"] = "error"
                        paper_result["error"] = str(e)
                        results_summary["errors"] += 1

                    results_summary["details"].append(paper_result)

                    # Update progress
                    progress_bar.progress((idx + 1) / len(papers_to_process))

                    # Display result
                    with results_container:
                        if paper_result["status"] == "success":
                            if "decision" in paper_result:
                                decision_emoji = {"INCLUDE": "🟢", "EXCLUDE": "🔴", "DISCUSS": "🟡"}.get(paper_result["decision"], "⚪")
                                decision_text = paper_result.get('decision', 'Extracted')
                                reason = paper_result.get('exclusion_reason')
                                reason_text = f" ({reason})" if reason else ""
                                st.write(f"{decision_emoji} {paper} - {decision_text}{reason_text}")
                            else:
                                st.write(f" {paper} - Data extracted")
                        else:
                            st.write(f" {paper} - Error: {paper_result.get('error', 'Unknown')}")

                # Auto-save batch results if enabled
                if auto_save_batch:
                    save_messages = []

                    # Save JSON
                    try:
                        results_summary["batch_mode"] = batch_mode
                        results_summary["completed_at"] = pd.Timestamp.now().isoformat()

                        # Load existing data if file exists
                        if os.path.exists(batch_output_path):
                            with open(batch_output_path, 'r', encoding='utf-8') as f:
                                all_batch_results = json.load(f)
                                if not isinstance(all_batch_results, list):
                                    all_batch_results = [all_batch_results]
                        else:
                            all_batch_results = []
                            os.makedirs(os.path.dirname(batch_output_path), exist_ok=True)

                        # Append new batch results
                        all_batch_results.append(results_summary)

                        # Save back to file
                        with open(batch_output_path, 'w', encoding='utf-8') as f:
                            json.dump(all_batch_results, f, indent=2, ensure_ascii=False)

                        save_messages.append(f"JSON: {batch_output_path}")
                    except Exception as e:
                        st.warning(f"⚠️ JSON auto-save failed: {str(e)}")

                    # Save extraction data (qualitative and quantitative, only for papers with extraction data)
                    try:
                        qual_rows_added = 0
                        total_effect_sizes = 0

                        for detail in results_summary["details"]:
                            # Only save if extraction data exists
                            if detail.get("status") == "success" and "extraction" in detail and detail["extraction"] is not None:
                                extraction_dict = detail["extraction"]

                                # Prepare paper metadata (extraction is a dict from extraction_payload)
                                paper_metadata = {
                                    "paper_id": extraction_dict.get("paper_id", "Unknown"),
                                    "authors": extraction_dict.get("paper_authors") or "",
                                    "year": extraction_dict.get("paper_year") or "",
                                    "doi": extraction_dict.get("paper_doi") or "",
                                    "title": extraction_dict.get("paper_title") or ""
                                }

                                # Save qualitative data (1 row per paper) using new output manager
                                qual_data_with_meta = {
                                    **extraction_dict.get("final_qualitative_data", {}),
                                    "needs_human_review": extraction_dict.get("needs_human_review", False),
                                    "overall_confidence": extraction_dict.get("overall_confidence", "MEDIUM")
                                }
                                save_qualitative_data(
                                    batch_qualitative_path,
                                    qual_data_with_meta,
                                    paper_metadata,
                                    format=qualitative_format
                                )
                                qual_rows_added += 1

                                # Save quantitative data (multiple rows per paper) using new output manager
                                quant_data = extraction_dict.get("final_quantitative_data", {})
                                # The output manager handles both dict and list formats automatically
                                num_effect_sizes = save_quantitative_data(
                                    batch_quantitative_path,
                                    quant_data,
                                    paper_metadata,
                                    format=quantitative_format
                                )
                                total_effect_sizes += num_effect_sizes

                        if qual_rows_added > 0:
                            save_messages.append(f"Qualitative {qualitative_format.upper()}: {batch_qualitative_path} ({qual_rows_added} papers)")
                        if total_effect_sizes > 0:
                            save_messages.append(f"Quantitative {quantitative_format.upper()}: {batch_quantitative_path} ({total_effect_sizes} effect sizes)")

                    except Exception as e:
                        st.warning(f"⚠️ Data auto-save failed: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())

                    if save_messages:
                        st.success(f"✅ Batch processing complete! Auto-saved to:\n- " + "\n- ".join(save_messages))
                    else:
                        st.warning("⚠️ Batch processing complete but auto-save failed")
                else:
                    st.success("✅ Batch processing complete!")

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Processed", results_summary["processed"])
                col2.metric("Included", results_summary["included"])
                col3.metric("Excluded", results_summary["excluded"])
                col4.metric("Discuss", results_summary["discuss"])
                col5.metric("Errors", results_summary["errors"])

                # Export results - fixed: removed nested button
                st.download_button(
                    label=" Download Results as JSON",
                    data=json.dumps(results_summary, indent=2, ensure_ascii=False),
                    file_name=f"batch_results_{batch_mode.replace(' ', '_').lower()}.json",
                    mime="application/json",
                    key="download_batch_json"
                )
    else:
        st.info("Upload PDF files above to start batch processing.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Meta-Analysis Data Extractor | Built with Streamlit by Hammed Akande</p>
</div>
""", unsafe_allow_html=True)
