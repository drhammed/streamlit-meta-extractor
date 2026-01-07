import os
import sys
import json
import logging
import argparse
import re
import socket
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

# Text processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# PDF extraction
from unstructured.documents.elements import Title
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Grobid for metadata extraction
try:
    from grobid_client.grobid_client import GrobidClient
    GROBID_AVAILABLE = True
except ImportError:
    GROBID_AVAILABLE = False
    logger.warning("grobid-client-python not installed. Falling back to heuristic metadata extraction.")

# Prompt loading
from prompts.prompt_loader import load_screening_prompts, load_extraction_prompts
from prompts.config import PROMPT_FORMAT


# ============================================================================
# CONFIGURATION
# ============================================================================

CHUNK_SIZE = 8000
CHUNK_OVERLAP = 500
TOP_K = 5


# ============================================================================
# JSON REPAIR UTILITIES
# ============================================================================

def repair_json(json_str: str) -> str:
    """
    Attempt to repair common JSON formatting issues from LLM responses.

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Repaired JSON string
    """
    if not json_str:
        return json_str

    # 1. Remove JavaScript-style comments first
    json_str = re.sub(r'//.*?\n', '\n', json_str)  # Single-line comments
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)  # Multi-line comments

    # 2. Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

    # 3. Fix unquoted property names (e.g., {key: "value"} -> {"key": "value"})
    # Only match unquoted keys (no quotes before the word)
    json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)

    # 4. Replace single quotes with double quotes for string values
    # Match single-quoted strings and replace with double quotes
    json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)  # For values
    json_str = re.sub(r"\[\s*'([^']*)'", r'["\1"', json_str)  # For array items

    # 5. Remove control characters that might cause issues (except newlines and tabs)
    json_str = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', json_str)

    return json_str

# Preferred local model defaults
DEFAULT_TEXT_MODEL = "gemma3:4b"
DEFAULT_VISION_MODEL = "gemma3:27b"

# Default paths and settings - change these as needed
DEFAULT_PDF_DIR = "./filtered_pdfs_168"
DEFAULT_OUTPUT_DIR = "./results"
DEFAULT_PROVIDER = "ollama"
DEFAULT_USE_VISION = True
DEFAULT_SAVE_IMAGES = False
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_CLOUD = False

# Load prompts from external files (cached for performance)
# Format is configured in prompts/config.py
_PROMPTS_CACHE = None

def _get_prompts():
    """Get or load prompts (cached)"""
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is None:
        _PROMPTS_CACHE = {
            "screening": load_screening_prompts(format=PROMPT_FORMAT),
            "extraction": load_extraction_prompts(format=PROMPT_FORMAT)
        }
    return _PROMPTS_CACHE


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        logger.info(f"Loading API keys from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value.strip()
        logger.info("API keys loaded from .env file")

# Load .env on import
load_env_file()


def ensure_ollama_running(host="localhost", port=11434):
    """Ensure local Ollama server is running"""
    sock = socket.socket()
    try:
        sock.settimeout(1)
        sock.connect((host, port))
    except Exception:
        logger.info("Starting local Ollama server...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        sock.close()


# ============================================================================
# DATA MODELS FOR SCREENING
# ============================================================================

class Decision(str, Enum):
    """Final screening decision"""
    INCLUDE = "INCLUDE"
    EXCLUDE = "EXCLUDE"
    DISCUSS = "DISCUSS"


class ExclusionCode(str, Enum):
    """Exclusion reason codes"""
    E1 = "E1"  # No quantitative specialization metric
    E2 = "E2"  # No climate quantification (either not climate-related OR no quantified climate data)
    E3 = "E3"  # No specialist vs generalist comparison
    E4 = "E4"  # Insufficient data for effect size extraction
    E5 = "E5"  # Reserved (tracked as moderator)
    E6 = "E6"  # Reserved (can be flagged for human review)
    E7 = "E7"  # Reserved
    E8 = "E8"  # Duplicate study (same paper already screened)
    E9 = "E9"  # Inaccessible (PDF corrupted, unreadable, or missing text)
    E10 = "E10"  # Other (catch-all for miscellaneous reasons)
    NONE = None


class CriteriaResponse(BaseModel):
    """Response for a single criterion"""
    answer: Literal["Yes", "No", "Unclear"]
    details: str = Field(description="Explanation or evidence for the answer")


class ScreeningCriteria(BaseModel):
    """All 6 screening criteria responses"""
    # Criterion 1: Specialization
    has_specialization_metric: CriteriaResponse = Field(default=CriteriaResponse(answer="Unclear", details="Not evaluated"))
    quantitative_metric: CriteriaResponse = Field(default=CriteriaResponse(answer="Unclear", details="Not evaluated"))
    specialization_metric_type: Optional[str] = None

    # Criterion 2: Climate Change
    climate_related: CriteriaResponse = Field(default=CriteriaResponse(answer="Unclear", details="Not evaluated"))
    climate_quantified: CriteriaResponse = Field(default=CriteriaResponse(answer="Unclear", details="Not evaluated"))
    climate_data_provided: CriteriaResponse = Field(default=CriteriaResponse(answer="Unclear", details="Not evaluated"))
    study_design_appropriate: CriteriaResponse = Field(default=CriteriaResponse(answer="Unclear", details="Not evaluated"))
    climate_variable: Optional[str] = None
    climate_change_magnitude: Optional[Union[str, float, int]] = None

    # Criterion 3: Relationship
    tests_relationship: CriteriaResponse = Field(default=CriteriaResponse(answer="Unclear", details="Not evaluated"))
    effect_direction_clear: CriteriaResponse = Field(default=CriteriaResponse(answer="Unclear", details="Not evaluated"))
    effect_direction: Optional[str] = None

    # Criterion 4: Quantitative Data
    statistics_reported: CriteriaResponse = Field(default=CriteriaResponse(answer="Unclear", details="Not evaluated"))
    effect_size_extractable: CriteriaResponse = Field(default=CriteriaResponse(answer="Unclear", details="Not evaluated"))
    effect_size_type: Optional[str] = None
    effect_size_value: Optional[Union[str, float, int]] = None

    # Criterion 5: Multi-species
    multiple_species: CriteriaResponse = Field(default=CriteriaResponse(answer="Unclear", details="Not evaluated"))
    community_metric: CriteriaResponse = Field(default=CriteriaResponse(answer="Unclear", details="Not evaluated"))
    number_of_species: Optional[int] = None

    # Criterion 6: Quality (peer-review check removed - assumed all papers are peer-reviewed)
    design_quality: CriteriaResponse = Field(default=CriteriaResponse(answer="Unclear", details="Not evaluated"))


class ReviewerAssessment(BaseModel):
    """A single reviewer's assessment"""
    reviewer_id: str
    criteria: ScreeningCriteria
    decision: Decision
    exclusion_reason: Optional[ExclusionCode] = None
    confidence: Literal["High", "Medium", "Low"]
    notes: str = ""


class ScreeningResult(BaseModel):
    """Final screening result after adjudication"""
    paper_id: str
    paper_title: str
    paper_authors: str
    paper_year: Optional[int] = None

    reviewer1: ReviewerAssessment
    reviewer2: ReviewerAssessment

    # Adjudication results
    final_decision: Decision
    final_exclusion_reason: Optional[ExclusionCode] = None
    adjudication_notes: str = ""
    needs_human_review: bool = False

    # Metadata
    screening_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    pdf_path: str = ""


# ============================================================================
# PDF PROCESSING MODULE (Shared)
# ============================================================================

class PDFProcessor:
    """Shared PDF processing with section extraction, chunking, and FAISS retrieval"""

    SECTION_KEYWORDS = {
        "methodology": [
            "methodology", "methods", "materials and methods", "materials & methods",
            "experimental procedure", "study design"
        ],
        "results": [
            "results", "findings", "results and discussion", "results & discussion",
            "results/discussion"
        ],
        "discussion": [
            "discussion", "interpretation", "discussion and conclusion",
            "discussion & conclusion"
        ],
        "conclusion": [
            "conclusion", "conclusions"
        ],
        "recommendations": [
            "recommendation", "recommendations", "implications",
            "practical implications", "future directions"
        ]
    }

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP,
                 grobid_server: str = "http://localhost:8070", use_grobid: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Initialize Grobid client if available and requested
        self.use_grobid = use_grobid and GROBID_AVAILABLE
        self.grobid_server = grobid_server
        self.grobid_client = None
        if self.use_grobid:
            try:
                self.grobid_client = GrobidClient(grobid_server=grobid_server)
                # Test connection
                import requests
                response = requests.get(f"{grobid_server}/api/isalive", timeout=2)
                if response.status_code != 200:
                    logger.warning(f"Grobid server at {grobid_server} is not responding. Falling back to heuristic extraction.")
                    self.use_grobid = False
                else:
                    logger.info(f"Connected to Grobid server at {grobid_server}")
            except Exception as e:
                logger.warning(f"Failed to connect to Grobid server: {e}. Falling back to heuristic extraction.")
                self.use_grobid = False

    def extract_text_from_pdf(self, path: str, use_ocr: bool = False) -> str:
        """Extract all text from PDF with optional OCR for scanned documents"""
        try:
            if use_ocr:
                # Use OCR-enabled extraction for scanned PDFs
                elems = partition_pdf(
                    path,
                    strategy=PartitionStrategy.HI_RES,
                    extract_images_in_pdf=True,
                    infer_table_structure=True,
                    ocr_languages="eng"
                )
            else:
                elems = partition_pdf(path, strategy=PartitionStrategy.HI_RES)

            texts = []
            for el in elems:
                if hasattr(el, "get_text"):
                    texts.append(el.get_text())
                elif hasattr(el, "text"):
                    texts.append(el.text)
                else:
                    texts.append(str(el))

            full_text = "\n".join(texts)

            # Check if extraction yielded very little text (might be scanned)
            if not use_ocr and len(full_text.strip()) < 500:
                logger.warning(f"PDF appears to have minimal text. Retrying with OCR enabled.")
                return self.extract_text_from_pdf(path, use_ocr=True)

            return full_text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            if not use_ocr:
                logger.info("Retrying with OCR enabled...")
                return self.extract_text_from_pdf(path, use_ocr=True)
            raise

    def extract_images_from_pdf(self, path: str, save_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract images from PDF pages as base64-encoded data

        Args:
            path: Path to PDF file
            save_dir: Optional directory to save images to disk (for debugging)

        Returns:
            List of dicts with 'page', 'base64', 'media_type' keys
        """
        import fitz  # PyMuPDF
        import base64
        from PIL import Image
        import io

        doc = fitz.open(path)
        images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()

            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Convert to PNG for consistency (most models prefer PNG)
                    image = Image.open(io.BytesIO(image_bytes))

                    # Convert CMYK to RGB (PNG doesn't support CMYK)
                    if image.mode == 'CMYK':
                        image = image.convert('RGB')
                    elif image.mode not in ('RGB', 'RGBA', 'L'):
                        # Convert any other mode to RGB
                        image = image.convert('RGB')

                    # Resize if too large (to save tokens/bandwidth)
                    max_size = 1024
                    if image.width > max_size or image.height > max_size:
                        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                    # Convert to base64
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()

                    image_data = {
                        "page": page_num + 1,
                        "index": img_index,
                        "base64": img_base64,
                        "media_type": "image/png"
                    }

                    # Optionally save to disk for debugging
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        img_path = os.path.join(save_dir, f"page{page_num+1}_img{img_index}.png")
                        image.save(img_path)
                        image_data["path"] = img_path
                        logger.info(f"Saved image to {img_path}")

                    images.append(image_data)

                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                    continue

        logger.info(f"Extracted {len(images)} images from {Path(path).name}")
        return images

    def extract_first_page(self, pdf_path: str) -> str:
        """Extract only the first page"""
        elements = partition_pdf(pdf_path, strategy=PartitionStrategy.AUTO)
        page1 = []
        for el in elements:
            meta = getattr(el, 'metadata', {})
            if isinstance(meta, dict) and meta.get('page_number') == 1:
                if hasattr(el, "get_text"):
                    page1.append(el.get_text())
                elif hasattr(el, "text"):
                    page1.append(el.text)
                else:
                    page1.append(str(el))
        return "\n".join(page1)

    def remove_unneeded_sections(self, text: str, cut_markers: Optional[List[str]] = None) -> str:
        """Remove references, acknowledgments, etc."""
        if cut_markers is None:
            cut_markers = ["references", "bibliography", "acknowledgments", "funding", "supplementary"]
        low = text.lower()
        indices = [low.find(m) for m in cut_markers if low.find(m) != -1]
        if indices:
            cut_at = min(indices)
            return text[:cut_at]
        return text

    def extract_sections(self, pdf_path: str) -> Dict[str, List[str]]:
        """Extract specific sections (Methods, Results, Discussion, etc.)"""
        # Try Unstructured Titles
        elems = partition_pdf(pdf_path, strategy=PartitionStrategy.HI_RES)
        sections = {sec: [] for sec in self.SECTION_KEYWORDS}
        current = None

        for el in elems:
            if isinstance(el, Title):
                heading = re.sub(r'^\d+(\.\d+)*[\.\)]?\s*', '', el.text).strip().lower()
                if any(m in heading for m in ("reference", "bibliography", "acknowledgments")):
                    break
                for sec, kw_list in self.SECTION_KEYWORDS.items():
                    if any(kw in heading for kw in kw_list):
                        current = sec
                        break
                continue

            if current:
                txt = el.get_text() if hasattr(el, "get_text") else getattr(el, "text", str(el))
                sections[current].append(txt)

        # If nothing, fallback to regex on raw text
        if not any(sections.values()):
            raw = self.extract_text_from_pdf(pdf_path)
            raw = self.remove_unneeded_sections(raw)
            sections = {sec: [] for sec in self.SECTION_KEYWORDS}
            current = None
            for line in raw.splitlines():
                ll = re.sub(r'^\d+(\.\d+)*[\.\)]?\s*', '', line.strip().lower())
                if any(m in ll for m in ("references", "bibliography", "acknowledgments")):
                    break
                for sec, kw_list in self.SECTION_KEYWORDS.items():
                    if any(ll.startswith(kw) for kw in kw_list):
                        current = sec
                        break
                if current:
                    sections[current].append(line)
        return sections

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        return self.text_splitter.split_text(text)

    def get_relevant_chunks(self, text: str, top_k: int = TOP_K) -> List[str]:
        """Get top-K most relevant chunks using FAISS"""
        chunks = self.chunk_text(text)
        if not chunks:
            return []

        # Create FAISS vector store
        vector_store = FAISS.from_texts(chunks, self.embeddings)

        # Get top-K similar chunks
        docs = vector_store.similarity_search(text, k=min(top_k, len(chunks)))

        # Remove duplicates while preserving order
        unique_chunks = []
        seen = set()
        for d in docs:
            content = d.page_content
            if content not in seen:
                unique_chunks.append(content)
                seen.add(content)

        return unique_chunks

    def extract_metadata_with_grobid(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata using Grobid service.
        Returns None if extraction fails or Grobid is not available.
        """
        if not self.use_grobid:
            return None

        try:
            import requests

            # Use Grobid REST API directly for header extraction
            grobid_url = self.grobid_server
            api_endpoint = f"{grobid_url}/api/processHeaderDocument"

            with open(pdf_path, 'rb') as pdf_file:
                files = {'input': (Path(pdf_path).name, pdf_file, 'application/pdf')}
                data = {'consolidateHeader': '1'}

                response = requests.post(
                    api_endpoint,
                    files=files,
                    data=data,
                    timeout=60
                )

                if response.status_code != 200:
                    logger.warning(f"Grobid returned status {response.status_code} for {pdf_path}")
                    return None

                # Parse BibTeX response
                bibtex = response.text.strip()

                # Extract title
                title = "Unknown"
                title_match = re.search(r'title\s*=\s*\{([^}]+)\}', bibtex, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()

                # Extract authors
                authors = "Unknown"
                author_match = re.search(r'author\s*=\s*\{([^}]+)\}', bibtex, re.IGNORECASE)
                if author_match:
                    authors = author_match.group(1).strip()

                # Extract year from BibTeX
                year = None
                year_match = re.search(r'year\s*=\s*\{([^}]+)\}', bibtex, re.IGNORECASE)
                if year_match:
                    try:
                        year = int(year_match.group(1).strip())
                    except (ValueError, TypeError):
                        pass

                # If no year found, try extracting from abstract or other BibTeX fields
                if year is None:
                    year_pattern = re.search(r'\b(19|20)\d{2}\b', bibtex)
                    if year_pattern:
                        try:
                            year = int(year_pattern.group(0))
                        except (ValueError, TypeError):
                            pass

                # Extract DOI
                doi = None
                doi_match = re.search(r'doi\s*=\s*\{([^}]+)\}', bibtex, re.IGNORECASE)
                if doi_match:
                    doi = doi_match.group(1).strip()

                # Create paper_id as FirstAuthor_Year
                first_author = authors.split(',')[0].strip().split()[-1] if authors != "Unknown" else "Unknown"
                paper_id = f"{first_author}_{year}" if year else Path(pdf_path).stem

                metadata = {
                    "paper_id": paper_id,
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "doi": doi
                }

                logger.info(f"Grobid extracted metadata for {Path(pdf_path).name}: {title[:50]}...")
                return metadata

        except Exception as e:
            logger.warning(f"Grobid metadata extraction failed for {pdf_path}: {e}")
            return None

    def _extract_year_from_text(self, paper_text: str) -> Optional[int]:
        """
        Helper method to extract publication year from paper text using heuristics.
        Searches the first ~3000 characters for a 4-digit year (19XX or 20XX).
        Returns None if no year found.
        """
        year_match = re.search(r'\b(19|20)\d{2}\b', paper_text[:3000])
        if year_match:
            try:
                return int(year_match.group(0))
            except (ValueError, TypeError):
                return None
        return None

    def extract_metadata(self, pdf_path: str, paper_text: str) -> Dict[str, Any]:
        """
        Extract basic metadata from paper.
        Tries Grobid first if available, falls back to heuristic extraction.
        Uses hybrid approach: Grobid for title/authors, heuristic fallback for missing year.
        """
        # Try Grobid first if enabled
        grobid_metadata = None
        if self.use_grobid:
            grobid_metadata = self.extract_metadata_with_grobid(pdf_path)
            if grobid_metadata:
                # If Grobid succeeded but year is missing, try heuristic year extraction
                if grobid_metadata.get("year") is None:
                    logger.info(f"Grobid extracted title/authors but not year for {Path(pdf_path).name}, trying heuristic year extraction")
                    year = self._extract_year_from_text(paper_text)
                    if year:
                        grobid_metadata["year"] = year
                        logger.info(f"Successfully extracted year {year} using heuristic fallback")
                return grobid_metadata
            else:
                logger.info(f"Grobid extraction failed, falling back to heuristic extraction for {Path(pdf_path).name}")

        # Fallback: heuristic extraction
        metadata = {
            "paper_id": Path(pdf_path).stem,
            "title": "Unknown",
            "authors": "Unknown",
            "year": None,
            "doi": None
        }

        # Extract title: Look for title in first few lines (skip URLs, DOIs, journal names)
        lines = [line.strip() for line in paper_text[:2000].split('\n') if line.strip()]

        # Try to find the actual title (typically the longest substantive line in first section)
        # Skip lines that are likely not the title
        potential_titles = []
        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            # Skip if line looks like URL, DOI, journal name, or is too short
            if any([
                line.lower().startswith('http'),
                'doi.org' in line.lower(),
                line.startswith('10.'),
                len(line) < 20,
                line.isupper() and len(line) < 50,  # ALL CAPS short lines (likely headers)
                'oecologia' in line.lower(),
                'journal' in line.lower(),
                line.count('@') > 0,  # Email addresses
            ]):
                continue

            # Likely a title if it's a longer line
            if 30 < len(line) < 300:
                potential_titles.append(line)

        if potential_titles:
            # Use the first substantial line as title
            metadata["title"] = potential_titles[0]
        elif lines:
            # Fallback: use first non-empty line
            metadata["title"] = lines[0]

        # Try to extract authors (usually comes after title, before abstract)
        # Look for lines with names (typically Title Case with commas or "and")
        for i, line in enumerate(lines[:20]):
            if any(indicator in line.lower() for indicator in ['abstract', 'introduction', 'keywords']):
                break
            # Check if line looks like author names
            if (',' in line or ' and ' in line) and not any(skip in line.lower() for skip in ['doi', 'http', 'journal', 'volume', 'pages']):
                # Count capitalized words (authors usually have multiple caps)
                cap_words = sum(1 for word in line.split() if word and word[0].isupper())
                if cap_words >= 2 and len(line) < 200:
                    metadata["authors"] = line
                    break

        # Try to extract year using regex
        year = self._extract_year_from_text(paper_text)
        if year:
            metadata["year"] = year

        # Try to extract DOI
        doi_pattern = r'(?:doi:|DOI:)?\s*(10\.\d{4,}/[^\s]+)'
        doi_match = re.search(doi_pattern, paper_text[:3000], re.IGNORECASE)
        if doi_match:
            metadata["doi"] = doi_match.group(1).rstrip('.,;')

        # Create paper_id as FirstAuthor_Year
        if metadata["authors"] != "Unknown":
            first_author = metadata["authors"].split(',')[0].strip().split()[-1]
        else:
            first_author = "Unknown"

        if metadata["year"]:
            metadata["paper_id"] = f"{first_author}_{metadata['year']}"
        else:
            metadata["paper_id"] = Path(pdf_path).stem

        return metadata

    def prepare_text_for_screening(self, pdf_path: str) -> Tuple[str, Dict[str, Any], List[str]]:
        """
        Prepare PDF text for screening phase
        Focuses on Abstract + Methods + Results (skips Introduction/Background)
        Returns: (full_text, metadata, relevant_chunks)
        """
        logger.info(f"Processing PDF for screening: {pdf_path}")

        # Extract full text
        try:
            full_text = self.extract_text_from_pdf(pdf_path)
        except Exception as e:
            logger.warning(f"Failed to extract full text, using first page: {e}")
            full_text = self.extract_first_page(pdf_path)

        # Clean text
        cleaned_text = self.remove_unneeded_sections(full_text)
        if not cleaned_text:
            cleaned_text = self.extract_first_page(pdf_path)

        # Extract metadata
        metadata = self.extract_metadata(pdf_path, cleaned_text)

        # Extract sections for targeted screening (skip Introduction/Background)
        sections_dict = self.extract_sections(pdf_path)

        # Combine ONLY Abstract + Methods + Results for screening
        screening_text_parts = []

        # Get Abstract (first ~2000 chars typically contain abstract)
        abstract_text = cleaned_text[:2000]
        screening_text_parts.append(abstract_text)

        # Get Methods section
        if sections_dict.get("methodology"):
            methods_text = "\n".join(sections_dict["methodology"])
            screening_text_parts.append(methods_text)

        # Get Results section
        if sections_dict.get("results"):
            results_text = "\n".join(sections_dict["results"])
            screening_text_parts.append(results_text)

        # Combine screening-relevant sections
        screening_text = "\n\n".join(screening_text_parts)

        # If section extraction failed, fallback to first 8000 chars (but log warning)
        if not screening_text or len(screening_text) < 500:
            logger.warning("Section extraction yielded insufficient text, using first 8000 chars as fallback")
            screening_text = cleaned_text[:8000]

        # Get relevant chunks for screening from the targeted sections
        relevant_chunks = self.get_relevant_chunks(screening_text, top_k=3)

        return cleaned_text, metadata, relevant_chunks

    def prepare_text_for_extraction(self, pdf_path: str) -> Tuple[str, Dict[str, str], List[str], List[str]]:
        """
        Prepare PDF text for data extraction phase
        Returns: (full_text, sections_dict, quant_chunks, qual_chunks)
        """
        logger.info(f"Processing PDF for data extraction: {pdf_path}")

        # Extract full text
        try:
            full_text = self.extract_text_from_pdf(pdf_path)
        except Exception as e:
            logger.warning(f"Failed to extract full text: {e}")
            full_text = self.extract_first_page(pdf_path)

        # Extract sections
        sections = self.extract_sections(pdf_path)

        # Prepare text for qualitative extraction (all text, trimmed)
        trimmed_full = self.remove_unneeded_sections(full_text)
        if not trimmed_full:
            trimmed_full = self.extract_first_page(pdf_path)

        # Prepare text for quantitative extraction (specific sections only)
        section_order = ["methodology", "results", "discussion", "conclusion", "recommendations"]
        combined_lines = []
        for sec in section_order:
            combined_lines.extend(sections.get(sec, []))
        combined_sections = "\n".join(combined_lines).strip()

        # Get relevant chunks for quantitative extraction
        quant_chunks = []
        if combined_sections:
            quant_chunks = self.get_relevant_chunks(combined_sections, top_k=TOP_K)

        # Get relevant chunks for qualitative extraction
        qual_chunks = []
        if trimmed_full:
            qual_chunks = self.get_relevant_chunks(trimmed_full, top_k=TOP_K)

        sections_dict = {
            "full_text": full_text,
            "trimmed_text": trimmed_full,
            "combined_sections": combined_sections
        }

        return full_text, sections_dict, quant_chunks, qual_chunks


# ============================================================================
# LLM PROVIDER INTERFACE
# ============================================================================

class LLMProvider:
    """Abstract base class for LLM providers"""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.supports_vision = False  # Override in vision-capable providers

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from prompt"""
        raise NotImplementedError

    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        system_prompt: Optional[str] = None
    ) -> BaseModel:
        """Generate structured output conforming to a Pydantic model"""
        raise NotImplementedError

    def generate_with_vision(
        self,
        prompt: str,
        images: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text from prompt with images (base64 encoded)

        Args:
            prompt: Text prompt
            images: List of dicts with 'base64' and 'media_type' keys
            system_prompt: Optional system prompt

        Returns:
            Generated text response
        """
        raise NotImplementedError("This provider does not support vision")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider via SDK"""

    def __init__(self, model_name: str = "claude-3-5-haiku-20241022", **kwargs):
        super().__init__(model_name, **kwargs)
        self.supports_vision = True  # Claude models support vision
        try:
            import anthropic

            # Try to get API key from environment
            api_key = os.environ.get("ANTHROPIC_API_KEY")

            # If not found, prompt user
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not found in environment variables")
                try:
                    import getpass
                    api_key = getpass.getpass("Enter Anthropic API key: ").strip()
                    if not api_key:
                        raise ValueError("API key is required for Anthropic provider")
                    os.environ["ANTHROPIC_API_KEY"] = api_key
                    logger.info("API key saved to environment for this session")
                except Exception as e:
                    raise ValueError(f"ANTHROPIC_API_KEY required: {e}")

            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model_name,
            "max_tokens": 4096,
            "messages": messages,
        }

        if system_prompt:
            # Enable prompt caching for system prompts
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        system_prompt: Optional[str] = None
    ) -> BaseModel:
        """Generate structured output using Anthropic's native structured output"""
        response_text = self.generate(prompt, system_prompt)

        # Parse JSON from response
        try:
            # Try to extract JSON if wrapped in markdown
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text

            # Repair and parse JSON
            json_str = repair_json(json_str)
            data = json.loads(json_str)
            return response_model.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to parse structured response: {e}")
            raise

    def generate_with_vision(
        self,
        prompt: str,
        images: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response with both text and images"""

        # Build content with images and text
        content = []

        # Add images first
        for img in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.get("media_type", "image/png"),
                    "data": img["base64"]
                }
            })

        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })

        messages = [{"role": "user", "content": content}]

        kwargs = {
            "model": self.model_name,
            "max_tokens": 4096,
            "messages": messages,
        }

        if system_prompt:
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class OpenAIProvider(LLMProvider):
    """OpenAI provider"""

    def __init__(self, model_name: str = "gpt-4o", **kwargs):
        super().__init__(model_name, **kwargs)
        # GPT-4o, GPT-4-turbo, and GPT-4V support vision
        self.supports_vision = "gpt-4" in model_name.lower() or "vision" in model_name.lower()
        try:
            from openai import OpenAI

            # Try to get API key from environment
            api_key = os.environ.get("OPENAI_API_KEY")

            # If not found, prompt user
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment variables")
                try:
                    import getpass
                    api_key = getpass.getpass("Enter OpenAI API key: ").strip()
                    if not api_key:
                        raise ValueError("API key is required for OpenAI provider")
                    os.environ["OPENAI_API_KEY"] = api_key
                    logger.info("API key saved to environment for this session")
                except Exception as e:
                    raise ValueError(f"OPENAI_API_KEY required: {e}")

            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content

    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        system_prompt: Optional[str] = None
    ) -> BaseModel:
        """Generate structured output using OpenAI's structured outputs"""
        response_text = self.generate(prompt, system_prompt)

        # Parse JSON from response
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text

            # Repair and parse JSON
            json_str = repair_json(json_str)
            data = json.loads(json_str)
            return response_model.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to parse structured response: {e}")
            raise

    def generate_with_vision(
        self,
        prompt: str,
        images: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response with both text and images"""

        if not self.supports_vision:
            raise NotImplementedError(f"Model {self.model_name} does not support vision")

        # Build content with images and text
        content = []

        # Add images first
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img.get('media_type', 'image/png')};base64,{img['base64']}"
                }
            })

        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=4096
        )
        return response.choices[0].message.content


class OllamaProvider(LLMProvider):
    """Ollama provider for local and cloud models"""

    def __init__(
        self,
        model_name: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url
        self.api_key = api_key

        # Detect vision support based on model name
        # Gemma 3 (4b, 27b), llama3.2-vision, Qwen3-VL, and other vision models
        vision_models = ["gemma3", "llama3.2-vision", "llava", "bakllava", "moondream", "qwen"]
        self.supports_vision = any(vm in model_name.lower() for vm in vision_models)

        # Detect if using Ollama Cloud
        self.is_cloud = "ollama.com" in base_url or api_key is not None

        try:
            from langchain_ollama import OllamaLLM

            # Configure for cloud or local
            if self.is_cloud:
                logger.info(f"Using Ollama Cloud with model {model_name}")
                if not api_key:
                    api_key = os.environ.get("OLLAMA_API_KEY", "")
                    if not api_key:
                        logger.warning("OLLAMA_API_KEY not found in environment variables")
                        try:
                            import getpass
                            api_key = getpass.getpass("Enter Ollama Cloud API key: ").strip()
                            if not api_key:
                                raise ValueError("API key is required for Ollama Cloud")
                            os.environ["OLLAMA_API_KEY"] = api_key
                            logger.info("API key saved to environment for this session")
                        except Exception as e:
                            raise ValueError(f"OLLAMA_API_KEY required for cloud models: {e}")

                cloud_base_url = "https://ollama.com" if base_url == "http://localhost:11434" else base_url

                self.client = OllamaLLM(
                    model=model_name,
                    base_url=cloud_base_url,
                    temperature=0,
                    headers={"Authorization": f"Bearer {api_key}"}
                )
            else:
                logger.info(f"Using local Ollama with model {model_name}")
                ensure_ollama_running()
                self.client = OllamaLLM(
                    model=model_name,
                    base_url=base_url,
                    temperature=0
                )
        except ImportError:
            raise ImportError("langchain-ollama not installed. Run: pip install langchain-ollama")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        return self.client.invoke(full_prompt)

    def generate_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        system_prompt: Optional[str] = None
    ) -> BaseModel:
        response_text = self.generate(prompt, system_prompt)

        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text

            # Repair and parse JSON
            json_str = repair_json(json_str)
            data = json.loads(json_str)
            return response_model.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to parse structured response: {e}")
            raise

    def generate_with_vision(
        self,
        prompt: str,
        images: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response with both text and images using Ollama API"""

        if not self.supports_vision:
            raise NotImplementedError(f"Model {self.model_name} does not support vision")

        # Use direct Ollama API for vision (langchain doesn't support it yet)
        import requests

        # Prepare the message with images
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Ollama expects images as base64 strings in the images array
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "images": [img["base64"] for img in images],
            "stream": False,
            "options": {
                "temperature": 0
            }
        }

        # Make API call
        url = f"{self.base_url}/api/generate"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=300)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.error(f"Error calling Ollama vision API: {e}")
            raise


# ============================================================================
# PHASE 1: SCREENING AGENTS
# ============================================================================

class ScreeningReviewerAgent:
    """A reviewer agent that screens papers based on inclusion criteria"""

    def __init__(self, agent_id: str, llm_provider: LLMProvider):
        self.agent_id = agent_id
        self.llm = llm_provider

        # Load prompts from external files
        prompts = _get_prompts()
        self.system_prompt = prompts["screening"]["reviewer"]["system_prompt"]
        self.user_prompt_template = prompts["screening"]["reviewer"]["user_prompt_template"]

    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """
        Extract JSON from LLM response, handling various formats:
        - Markdown code blocks (```json ... ```)
        - Plain JSON with text before/after
        - Direct JSON response
        Also attempts to repair common JSON formatting issues.
        """
        if not response_text:
            return None

        # Try markdown code block first
        if "```json" in response_text:
            try:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
                # Attempt to repair and validate
                json_str = repair_json(json_str)
                json.loads(json_str)  # Validate
                return json_str
            except (IndexError, json.JSONDecodeError):
                pass

        # Try to find JSON object by looking for the first { and last }
        first_brace = response_text.find('{')
        last_brace = response_text.rfind('}')

        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            json_str = response_text[first_brace:last_brace + 1].strip()
            # Attempt to repair and validate
            try:
                json_str = repair_json(json_str)
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError as e:
                logger.warning(f"{self.agent_id}: Failed to parse extracted JSON after repair: {e}")
                logger.debug(f"Extracted JSON: {json_str[:500]}...")
                pass

        # Fallback: try to repair the entire response
        try:
            repaired = repair_json(response_text.strip())
            json.loads(repaired)  # Validate
            return repaired
        except json.JSONDecodeError:
            pass

        # Last resort: return stripped response
        return response_text.strip()

    def review_paper(self, paper_chunks: List[str], paper_metadata: Dict[str, str]) -> ReviewerAssessment:
        """Review a paper and return assessment"""

        # Combine chunks for context
        combined_text = "\n\n---\n\n".join(paper_chunks[:3])  # Use top 3 chunks

        # Format user prompt with actual data
        prompt = self.user_prompt_template.format(
            title=paper_metadata.get('title', 'Unknown'),
            authors=paper_metadata.get('authors', 'Unknown'),
            year=paper_metadata.get('year', 'Unknown'),
            combined_text=combined_text
        )

        try:
            # Get full structured response
            full_response_text = self.llm.generate(prompt, self.system_prompt)

            # Check if response is empty
            if not full_response_text or not full_response_text.strip():
                logger.error(f"Empty response from LLM for {self.agent_id}")
                raise ValueError(f"{self.agent_id}: Received empty response from LLM")

            # Log the response for debugging
            logger.debug(f"{self.agent_id} raw response: {full_response_text[:200]}...")

            # Parse JSON - improved extraction to handle text before JSON
            json_str = self._extract_json_from_response(full_response_text)

            if not json_str:
                logger.error(f"No JSON content found in response for {self.agent_id}")
                logger.error(f"Full response: {full_response_text}")
                raise ValueError(f"{self.agent_id}: No valid JSON found in LLM response")

            try:
                full_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"{self.agent_id}: JSON decode error after repair: {e}")
                logger.error(f"Attempted to parse: {json_str[:500]}...")
                raise

            # Parse criteria
            criteria = ScreeningCriteria.model_validate(full_data["criteria"])

            assessment = ReviewerAssessment(
                reviewer_id=self.agent_id,
                criteria=criteria,
                decision=Decision(full_data["decision"]),
                exclusion_reason=ExclusionCode(full_data["exclusion_reason"]) if full_data.get("exclusion_reason") else None,
                confidence=full_data["confidence"],
                notes=full_data.get("notes", "")
            )

            return assessment

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in {self.agent_id}: {e}")
            logger.error(f"Response text: {full_response_text if 'full_response_text' in locals() else 'No response received'}")
            raise ValueError(f"{self.agent_id}: Failed to parse JSON response - {e}")
        except Exception as e:
            logger.error(f"Error in reviewer {self.agent_id}: {e}")
            if 'full_response_text' in locals():
                logger.error(f"Response text: {full_response_text[:500]}...")
            raise


class ScreeningAdjudicatorAgent:
    """Adjudicator agent that resolves conflicts between screening reviewers"""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

        # Load prompts from external files
        prompts = _get_prompts()
        self.system_prompt = prompts["screening"]["adjudicator"]["system_prompt"]
        self.user_prompt_template = prompts["screening"]["adjudicator"]["user_prompt_template"]

    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """
        Extract JSON from LLM response, handling various formats:
        - Markdown code blocks (```json ... ```)
        - Plain JSON with text before/after
        - Direct JSON response
        Also attempts to repair common JSON formatting issues.
        """
        if not response_text:
            return None

        # Try markdown code block first
        if "```json" in response_text:
            try:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
                # Attempt to repair and validate
                json_str = repair_json(json_str)
                json.loads(json_str)  # Validate
                return json_str
            except (IndexError, json.JSONDecodeError):
                pass

        # Try to find JSON object by looking for the first { and last }
        first_brace = response_text.find('{')
        last_brace = response_text.rfind('}')

        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            json_str = response_text[first_brace:last_brace + 1].strip()
            # Attempt to repair and validate
            try:
                json_str = repair_json(json_str)
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError as e:
                logger.warning(f"Adjudicator: Failed to parse extracted JSON after repair: {e}")
                logger.debug(f"Extracted JSON: {json_str[:500]}...")
                pass

        # Fallback: try to repair the entire response
        try:
            repaired = repair_json(response_text.strip())
            json.loads(repaired)  # Validate
            return repaired
        except json.JSONDecodeError:
            pass

        # Last resort: return stripped response
        return response_text.strip()

    def adjudicate(
        self,
        paper_metadata: Dict[str, str],
        review1: ReviewerAssessment,
        review2: ReviewerAssessment
    ) -> Dict[str, Any]:
        """Adjudicate between two reviewer assessments"""

        # Format user prompt with actual data
        agreement_status = "REVIEWERS AGREE" if review1.decision == review2.decision else "REVIEWERS DISAGREE"

        prompt = self.user_prompt_template.format(
            title=paper_metadata.get('title', 'Unknown'),
            authors=paper_metadata.get('authors', 'Unknown'),
            reviewer1_id=review1.reviewer_id,
            reviewer1_decision=review1.decision,
            reviewer1_exclusion_reason=review1.exclusion_reason,
            reviewer1_confidence=review1.confidence,
            reviewer1_notes=review1.notes,
            reviewer2_id=review2.reviewer_id,
            reviewer2_decision=review2.decision,
            reviewer2_exclusion_reason=review2.exclusion_reason,
            reviewer2_confidence=review2.confidence,
            reviewer2_notes=review2.notes,
            agreement_status=agreement_status
        )

        response_text = self.llm.generate(prompt, self.system_prompt)

        # Parse response
        try:
            # Check if response is empty
            if not response_text or not response_text.strip():
                logger.error("Empty response from adjudicator LLM")
                raise ValueError("Adjudicator: Received empty response from LLM")

            # Log the response for debugging
            logger.debug(f"Adjudicator raw response: {response_text[:200]}...")

            # Parse JSON - improved extraction to handle text before JSON
            json_str = self._extract_json_from_response(response_text)

            if not json_str:
                logger.error("No JSON content found in adjudicator response")
                logger.error(f"Full response: {response_text}")
                raise ValueError("Adjudicator: No valid JSON found in LLM response")

            result = json.loads(json_str)
            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in adjudication: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            # Default to DISCUSS and flag for human review
            return {
                "final_decision": "DISCUSS",
                "final_exclusion_reason": None,
                "adjudication_notes": f"Adjudication JSON parsing failed: {e}. Response: {response_text[:200]}",
                "needs_human_review": True
            }
        except Exception as e:
            logger.error(f"Error in adjudication: {e}")
            if 'response_text' in locals():
                logger.error(f"Response text: {response_text[:500]}")
            # Default to DISCUSS and flag for human review
            return {
                "final_decision": "DISCUSS",
                "final_exclusion_reason": None,
                "adjudication_notes": f"Adjudication failed: {e}",
                "needs_human_review": True
            }


# ============================================================================
# PHASE 2: DATA EXTRACTION AGENTS
# ============================================================================

class ExtractionAssessment(BaseModel):
    """A single extractor's assessment"""
    extractor_id: str
    quantitative_data: Any = Field(default_factory=list)  # Can be list (array format) or dict (legacy)
    qualitative_data: Dict[str, Any] = Field(default_factory=dict)
    confidence: Literal["High", "Medium", "Low"]
    notes: str = ""


class ExtractionResult(BaseModel):
    """Final extraction result after reconciliation"""
    paper_id: str

    # Individual extractor results
    extractor1: ExtractionAssessment
    extractor2: ExtractionAssessment

    # Reconciled results
    final_quantitative_data: Any = Field(default_factory=list)  # Can be list (multiple effect sizes) or dict (legacy)
    final_qualitative_data: Dict[str, Any] = Field(default_factory=dict)

    # Confidence tracking
    field_confidence: Dict[str, str] = Field(default_factory=dict)  # field -> HIGH/MEDIUM/LOW
    overall_confidence: str = "MEDIUM"  # Overall extraction confidence

    # Conflict tracking
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    auto_resolved: List[Dict[str, Any]] = Field(default_factory=list)  # Auto-resolved semantic equivalences
    needs_human_review: bool = False
    reconciliation_notes: str = ""

    # Paper metadata (extracted from PDF)
    paper_title: Optional[str] = None
    paper_authors: Optional[str] = None
    paper_year: Optional[int] = None
    paper_doi: Optional[str] = None

    # Metadata
    extraction_date: str = Field(default_factory=lambda: datetime.now().isoformat())


def extract_first_json(s: str) -> Optional[str]:
    """Extract first complete JSON object from string"""
    start = None
    stack = []
    for i, c in enumerate(s):
        if c == '{':
            if start is None:
                start = i
            stack.append('{')
        elif c == '}' and stack:
            stack.pop()
            if not stack:
                return s[start:i+1]
    return None


def extract_json_from_response(s: str) -> Optional[Dict]:
    """Extract and parse JSON from LLM response"""
    js = extract_first_json(s)
    if not js:
        return None
    try:
        return json.loads(js)
    except json.JSONDecodeError:
        return None


def merge_field_values(values: List[str], embeddings: HuggingFaceEmbeddings, threshold: float = 0.85) -> str:
    """Merge near-duplicate values using cosine similarity"""
    vecs = embeddings.embed_documents(values)
    uniq, uniq_vecs = [], []
    for i, v in enumerate(values):
        if not uniq:
            uniq.append(v)
            uniq_vecs.append(vecs[i])
        else:
            sims = cosine_similarity([vecs[i]], uniq_vecs)[0]
            if max(sims) < threshold:
                uniq.append(v)
                uniq_vecs.append(vecs[i])
    return ", ".join(uniq)


class FigureExtractionAgent:
    """Agent for extracting quantitative data from figures using vision-capable LLM"""

    def __init__(
        self,
        llm_provider: LLMProvider,
        pdf_processor,
        prompt: Optional[str] = None
    ):
        self.llm = llm_provider
        self.pdf_processor = pdf_processor
        self.prompt_template = prompt or self._load_default_prompt()

        if not self.llm.supports_vision:
            logger.warning(f"LLM {self.llm.model_name} does not support vision. Figure extraction will be skipped.")

    @staticmethod
    def _load_default_prompt() -> str:
        """Load the default figure extraction prompt from prompts package."""
        prompts = _get_prompts()
        figure_prompt = prompts["extraction"].get("figure_extraction")
        if not figure_prompt:
            raise ValueError("Figure extraction prompt not found. Please add it to prompts/extraction_agents.")
        return figure_prompt

    def extract_from_pdf(self, pdf_path: str, save_images: bool = False) -> List[Dict[str, Any]]:
        """Extract quantitative data from all figures in a PDF

        Args:
            pdf_path: Path to PDF file
            save_images: If True, save extracted images to disk for debugging

        Returns:
            List of extracted data from each figure
        """
        if not self.llm.supports_vision:
            logger.info("Skipping figure extraction (vision not supported)")
            return []

        # Extract images from PDF
        save_dir = None
        if save_images:
            save_dir = os.path.join(os.path.dirname(pdf_path), "extracted_images", Path(pdf_path).stem)

        images = self.pdf_processor.extract_images_from_pdf(pdf_path, save_dir=save_dir)

        if not images:
            logger.info(f"No images found in {Path(pdf_path).name}")
            return []

        logger.info(f"Extracting data from {len(images)} figures using vision LLM...")

        extractions = []
        for i, img in enumerate(images):
            try:
                logger.info(f"  Processing figure {i+1}/{len(images)} (page {img['page']})")

                # Call vision LLM
                response = self.llm.generate_with_vision(
                    prompt=self.prompt_template,
                    images=[img],
                    system_prompt=None
                )

                # Parse JSON response
                data = extract_json_from_response(response)
                if data:
                    data["source_page"] = img["page"]
                    data["figure_index"] = i
                    extractions.append(data)
                else:
                    logger.warning(f"Failed to parse response for figure {i+1}")

            except Exception as e:
                logger.error(f"Error extracting from figure {i+1}: {e}")
                continue

        logger.info(f"Successfully extracted data from {len(extractions)}/{len(images)} figures")
        return extractions

    def merge_figure_extractions(self, extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge quantitative data from multiple figures"""
        if not extractions:
            return {}

        # Filter out non-quantitative figures
        quant_extractions = [
            ext for ext in extractions
            if ext.get("figure_type") != "non-quantitative"
        ]

        if not quant_extractions:
            return {"notes": "No quantitative figures found"}

        # Merge extracted values (prioritize the most complete extraction)
        merged = {
            "effect_size": "not found",
            "effect_size_unit": "not found",
            "sample_size": "not found",
            "variance_value": "not found",
            "variance_type": "not found",
            "statistical_significance": "not found",
            "notes": []
        }

        for ext in quant_extractions:
            # Update fields if current extraction has values
            for key in ["effect_size", "effect_size_unit", "sample_size", "variance_value", "variance_type", "statistical_significance"]:
                val = ext.get(key, "not found")
                if val and val != "not found" and merged[key] == "not found":
                    merged[key] = val

            # Collect notes
            if ext.get("notes"):
                merged["notes"].append(f"Fig on page {ext['source_page']}: {ext['notes']}")

        merged["notes"] = " | ".join(merged["notes"]) if merged["notes"] else "Extracted from figures"
        merged["num_figures_analyzed"] = len(quant_extractions)

        return merged


class QuantitativeExtractionAgent:
    """Agent for extracting quantitative data from papers"""

    def __init__(self, llm_provider: LLMProvider, embeddings: HuggingFaceEmbeddings):
        self.llm = llm_provider
        self.embeddings = embeddings

        # Load prompts from external files
        prompts = _get_prompts()
        self.prompt_template = prompts["extraction"]["quantitative_extraction"]

    def extract_from_chunk(self, chunk: str) -> Optional[Dict[str, Any]]:
        """Extract quantitative data from a single chunk"""

        # Format prompt with chunk
        prompt = self.prompt_template.format(chunk=chunk)

        try:
            raw = self.llm.generate(prompt, system_prompt=None)
            js = extract_json_from_response(raw)
            return js
        except Exception as e:
            logger.error(f"Error extracting from chunk: {e}")
            return None

    def merge_extractions(self, ex_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple quantitative extractions from different chunks"""
        merged = {}
        keys = [
            "metric", "effect_size", "effect_size_unit", "sample_size",
            "variance_type", "variance_value", "stat_method", "represents",
            "temperature_change", "precipitation_change", "other_climatic_drivers",
            "other_climatic_drivers_value", "latitude", "longitude", "notes"
        ]

        for k in keys:
            vals = []
            for ext in ex_list:
                raw_v = ext.get(k)
                if raw_v is None:
                    continue
                if isinstance(raw_v, list):
                    v = ", ".join(map(str, raw_v))
                else:
                    v = str(raw_v).strip()
                if v and v.lower() not in ("na", "none", "not reported", "not found"):
                    vals.append(v)

            if vals:
                if k not in ("other_climatic_drivers", "other_climatic_drivers_value"):
                    merged[k] = merge_field_values(vals, self.embeddings)
                else:
                    merged[k] = vals[0]
            else:
                merged[k] = "not found"

        return merged

    def extract(self, chunks: List[str]) -> Any:
        """
        Extract quantitative data from all chunks.

        Returns:
            List of effect sizes if the prompt returns arrays, or a single dict if the prompt returns a merged dict
        """
        logger.info(f"Extracting quantitative data from {len(chunks)} chunks")

        all_effect_sizes = []
        for i, chunk in enumerate(chunks):
            logger.info(f"  Processing chunk {i+1}/{len(chunks)}")
            result = self.extract_from_chunk(chunk)
            if result:
                # Check if result is an array (new prompt format) or dict (old format)
                if isinstance(result, list):
                    # New format: result is an array of effect sizes
                    all_effect_sizes.extend(result)
                    logger.info(f"    Found {len(result)} effect size(s) in chunk {i+1}")
                elif isinstance(result, dict):
                    # Old format: result is a single effect size dict
                    all_effect_sizes.append(result)
                    logger.info(f"    Found 1 effect size in chunk {i+1}")

        if all_effect_sizes:
            logger.info(f"Quantitative extraction complete: {len(all_effect_sizes)} total effect size(s) found")
            # Return as list to preserve multiple effect sizes per paper
            return all_effect_sizes
        else:
            logger.info("No quantitative effect sizes found")
            return []


class QualitativeExtractionAgent:
    """Agent for extracting qualitative metadata from papers"""

    def __init__(self, llm_provider: LLMProvider, embeddings: HuggingFaceEmbeddings):
        self.llm = llm_provider
        self.embeddings = embeddings

        # Load prompts from external files
        prompts = _get_prompts()
        self.prompt_template = prompts["extraction"]["qualitative_extraction"]

    def extract_from_chunk(self, chunk: str) -> Optional[Dict[str, Any]]:
        """Extract qualitative data from a single chunk"""

        # Format prompt with chunk
        prompt = self.prompt_template.format(chunk=chunk)

        try:
            raw = self.llm.generate(prompt, system_prompt=None)
            js = extract_json_from_response(raw)
            return js
        except Exception as e:
            logger.error(f"Error extracting from chunk: {e}")
            return None

    def merge_extractions(self, extraction_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple qualitative extractions from different chunks"""
        merged = {}
        keys = [
            "study_location", "dominant_ecosystem", "focal_taxa", "disturbance_type",
            "main_response_variable", "study_type", "niche_metrics", "specialist_vs_generalist",
            "environmental_variability", "niche_change_direction", "trait_info", "primary_climatic_driver",
            "ecosystem_context", "vulnerability_impacts", "theoretical_framework",
            "conservation_implications", "overall_comment"
        ]

        for key in keys:
            vals = []
            for ext in extraction_list:
                val = ext.get(key, "")
                if isinstance(val, list):
                    val = ", ".join(val)
                val = val.strip()
                if val and val.lower() not in ("na", "none", "not reported"):
                    vals.append(val)
            merged[key] = merge_field_values(vals, self.embeddings) if vals else "not reported"

        return merged

    def extract(self, chunks: List[str]) -> Dict[str, Any]:
        """Extract qualitative data from all chunks and merge"""
        logger.info(f"Extracting qualitative data from {len(chunks)} chunks")

        extractions = []
        for i, chunk in enumerate(chunks):
            logger.info(f"  Processing chunk {i+1}/{len(chunks)}")
            result = self.extract_from_chunk(chunk)
            if result:
                extractions.append(result)

        if extractions:
            merged = self.merge_extractions(extractions)
            logger.info(f"Qualitative extraction complete: {len(extractions)} chunks merged")
            return merged
        else:
            logger.info("No qualitative extractions")
            return {}


class ExtractionReconciliationAgent:
    """Agent for reconciling extraction results from two independent extractors"""

    EFFECT_SIZE_NOTE_PATTERNS = [
        "no quantitative effect size",
        "no quantitative effect sizes",
        "no effect size was reported",
        "no effect sizes were reported",
        "does not report any quantitative effect size",
        "does not report any quantitative effect sizes",
        "does not report an effect size",
        "no standardized effect size",
        "no standardized effect sizes",
        "effect size is not reported",
        "effect sizes are not reported",
        "no quantitative results",
        "no effect size reported",
        "no effect sizes reported"
    ]

    def __init__(self, llm_provider: LLMProvider, embeddings: HuggingFaceEmbeddings):
        self.llm = llm_provider
        self.embeddings = embeddings

        # Load prompts from external files
        prompts = _get_prompts()
        self.system_prompt = prompts["extraction"]["reconciliation_system"]
        self.user_prompt_template = prompts["extraction"]["reconciliation_user_template"]

    def _deduplicate_and_renumber_effect_sizes(self, effect_sizes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate effect sizes, filter out qualitative-only comparisons, and ensure unique sequential IDs.

        Duplicates are identified by comparing key fields (effect_size, response_variable,
        climate_variable_type, etc.). After deduplication, IDs are renumbered as ES1, ES2, ES3...
        """
        if not effect_sizes:
            return []

        # First, filter out qualitative-only comparisons (no numeric effect size)
        quantitative_only = []
        for es in effect_sizes:
            effect_size_value = es.get("effect_size", "")
            # Skip if effect_size is "not found", empty, or non-numeric
            if effect_size_value in ["not found", "not applicable", "", None, "CONFLICT"]:
                logger.info(f"Filtering out qualitative comparison: {es.get('response_variable', 'unknown')}")
                continue
            # Also skip if it's a string that doesn't represent a number
            if isinstance(effect_size_value, str):
                try:
                    float(effect_size_value)
                except (ValueError, TypeError):
                    logger.info(f"Filtering out non-numeric effect size: {effect_size_value}")
                    continue
            quantitative_only.append(es)

        if not quantitative_only:
            logger.info("No quantitative effect sizes found after filtering")
            return []

        # Deduplicate based on content similarity
        unique_effect_sizes = []
        seen_signatures = set()

        def normalize_text(text: str) -> str:
            """Normalize text for comparison by removing minor variations"""
            import re
            if not text or text in ["not found", "not applicable", ""]:
                return ""
            # Convert to lowercase
            text = str(text).lower().strip()
            # Remove content in parentheses (often clarifications/details)
            text = re.sub(r'\([^)]*\)', '', text)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            # Remove common punctuation that doesn't change meaning
            text = re.sub(r'[,;:.!?]', '', text)
            # Remove hyphens and dashes
            text = re.sub(r'[-–—]', ' ', text)
            # Final cleanup
            text = text.strip()
            return text

        for es in quantitative_only:
            # Create a signature based on key fields to identify duplicates
            # Normalize text fields to catch minor variations
            signature = (
                str(es.get("effect_size", "")),  # Keep numeric as-is
                normalize_text(es.get("response_variable", "")),
                normalize_text(es.get("climate_variable_type", "")),
                normalize_text(es.get("study_period", "")),
                normalize_text(es.get("stat_method", "")),
                normalize_text(es.get("notes", "")[:200])  # First 200 chars, normalized
            )

            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_effect_sizes.append(es)
            else:
                logger.info(f"Removing duplicate (exact match): effect_size={es.get('effect_size')}, response_variable={es.get('response_variable', 'unknown')[:50]}")

        # Second pass: Use semantic similarity to catch edge cases where text differs but meaning is same
        # Only run if we have multiple effect sizes to compare
        if len(unique_effect_sizes) > 1:
            logger.info(f"Running semantic similarity check on {len(unique_effect_sizes)} effect sizes")
            final_unique = []

            for i, es1 in enumerate(unique_effect_sizes):
                is_duplicate = False

                for es2 in final_unique:
                    # Only check semantic similarity if effect sizes are very close or identical
                    try:
                        val1 = float(es1.get("effect_size", 0))
                        val2 = float(es2.get("effect_size", 0))

                        # Skip semantic check if effect sizes are very different (>10% difference)
                        if abs(val1 - val2) > abs(val1 * 0.1):
                            continue
                    except (ValueError, TypeError):
                        # If can't convert to float, skip this comparison
                        continue

                    # Check semantic similarity of key text fields
                    # Combine key fields into descriptive text for comparison
                    text1 = " ".join([
                        str(es1.get("response_variable", "")),
                        str(es1.get("climate_variable_type", "")),
                        str(es1.get("notes", "")[:100])
                    ])
                    text2 = " ".join([
                        str(es2.get("response_variable", "")),
                        str(es2.get("climate_variable_type", "")),
                        str(es2.get("notes", "")[:100])
                    ])

                    # Skip if either text is too short
                    if len(text1.strip()) < 20 or len(text2.strip()) < 20:
                        continue

                    # Calculate cosine similarity using embeddings
                    try:
                        vec1 = self.embeddings.embed_documents([text1])[0]
                        vec2 = self.embeddings.embed_documents([text2])[0]

                        # Cosine similarity
                        import numpy as np
                        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

                        # If similarity > 0.90, consider as duplicate (very high threshold)
                        if similarity > 0.90:
                            logger.info(f"Removing duplicate (semantic similarity={similarity:.3f}): {es1.get('response_variable', 'unknown')[:50]}")
                            is_duplicate = True
                            break
                    except Exception as e:
                        # If embedding fails, skip semantic check
                        logger.debug(f"Semantic similarity check failed: {e}")
                        continue

                if not is_duplicate:
                    final_unique.append(es1)

            unique_effect_sizes = final_unique
            logger.info(f"After semantic deduplication: {len(unique_effect_sizes)} unique effect sizes")

        # Renumber with sequential IDs
        for i, es in enumerate(unique_effect_sizes, start=1):
            es["effect_size_id"] = f"ES{i}"

        logger.info(f"Processed {len(effect_sizes)} effect sizes: filtered to {len(quantitative_only)} quantitative, deduplicated to {len(unique_effect_sizes)} unique entries")
        return unique_effect_sizes

    def reconcile(
        self,
        extractor1_quant: Any,  # Can be Dict or List
        extractor2_quant: Any,  # Can be Dict or List
        extractor1_qual: Dict[str, Any],
        extractor2_qual: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Reconcile extraction results from two extractors"""

        # Handle new array format for quantitative data
        # If both extractors return arrays, combine unique effect sizes
        if isinstance(extractor1_quant, list) and isinstance(extractor2_quant, list):
            logger.info(f"Reconciling array-based quantitative data: {len(extractor1_quant)} + {len(extractor2_quant)} effect sizes")

            # Combine all effect sizes from both extractors
            combined_effect_sizes = extractor1_quant + extractor2_quant

            # Deduplicate and renumber
            unique_effect_sizes = self._deduplicate_and_renumber_effect_sizes(combined_effect_sizes)

            # Try LLM reconciliation for qualitative data only
            try:
                prompt = self.user_prompt_template.format(
                    extractor1_quant=json.dumps(extractor1_quant, indent=2),
                    extractor2_quant=json.dumps(extractor2_quant, indent=2),
                    extractor1_qual=json.dumps(extractor1_qual, indent=2),
                    extractor2_qual=json.dumps(extractor2_qual, indent=2)
                )

                response_text = self.llm.generate(prompt, self.system_prompt)

                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                else:
                    json_str = response_text

                json_str = repair_json(json_str)
                result = json.loads(json_str)

                # Override quantitative data with deduplicated array
                result["final_quantitative_data"] = unique_effect_sizes

                # Check qualitative completeness
                final_qual = result.get("final_qualitative_data", {})
                completeness_check = self._check_qualitative_completeness(final_qual)

                if completeness_check["is_incomplete"]:
                    logger.warning(f"Qualitative data is incomplete: {completeness_check['completeness_rate']}% complete")
                    result["needs_human_review"] = True
                    result["overall_confidence"] = "LOW"

                    conflicts = result.setdefault("conflicts", [])
                    conflicts.append({
                        "field": "qualitative_data",
                        "conflict_type": "INCOMPLETE_EXTRACTION",
                        "confidence": "LOW",
                        "recommendation": f"Only {completeness_check['completeness_rate']}% of important qualitative fields were extracted. Missing: {', '.join(completeness_check['missing_fields'][:5])}"
                    })

                    notes = result.get("reconciliation_notes", "")
                    extra_note = f"Qualitative extraction incomplete ({completeness_check['completeness_rate']}% complete)."
                    result["reconciliation_notes"] = f"{notes} {extra_note}".strip() if notes else extra_note

                return result

            except Exception as e:
                logger.warning(f"LLM reconciliation failed, using simple merge: {e}")

                # Check qualitative completeness for fallback path too
                completeness_check = self._check_qualitative_completeness(extractor1_qual)

                # Fallback: merge qualitative using first extractor
                result = {
                    "final_quantitative_data": unique_effect_sizes,
                    "final_qualitative_data": extractor1_qual,
                    "field_confidence": {},
                    "conflicts": [],
                    "auto_resolved": [],
                    "needs_human_review": completeness_check["is_incomplete"],
                    "overall_confidence": "LOW" if completeness_check["is_incomplete"] else "MEDIUM",
                    "reconciliation_notes": f"Array-based extraction: combined and deduplicated to {len(unique_effect_sizes)} unique effect sizes from both extractors"
                }

                if completeness_check["is_incomplete"]:
                    result["conflicts"].append({
                        "field": "qualitative_data",
                        "conflict_type": "INCOMPLETE_EXTRACTION",
                        "confidence": "LOW",
                        "recommendation": f"Only {completeness_check['completeness_rate']}% of qualitative fields extracted."
                    })
                    result["reconciliation_notes"] += f" Qualitative extraction incomplete ({completeness_check['completeness_rate']}% complete)."

                return result

        # Legacy dict-based reconciliation
        # Format user prompt with actual data
        prompt = self.user_prompt_template.format(
            extractor1_quant=json.dumps(extractor1_quant, indent=2),
            extractor2_quant=json.dumps(extractor2_quant, indent=2),
            extractor1_qual=json.dumps(extractor1_qual, indent=2),
            extractor2_qual=json.dumps(extractor2_qual, indent=2)
        )

        try:
            response_text = self.llm.generate(prompt, self.system_prompt)

            # Parse response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text

            # Repair and parse JSON
            json_str = repair_json(json_str)
            result = json.loads(json_str)
            self._flag_effect_size_contradictions(
                result,
                extractor1_quant,
                extractor2_quant
            )

            # Check qualitative completeness and flag if mostly empty
            final_qual = result.get("final_qualitative_data", {})
            completeness_check = self._check_qualitative_completeness(final_qual)

            if completeness_check["is_incomplete"]:
                logger.warning(f"Qualitative data is incomplete: {completeness_check['completeness_rate']}% complete ({completeness_check['missing_count']}/{completeness_check['total_fields']} fields missing)")

                # Downgrade confidence and flag for human review
                result["needs_human_review"] = True
                result["overall_confidence"] = "LOW"

                # Add conflict note
                conflicts = result.setdefault("conflicts", [])
                conflicts.append({
                    "field": "qualitative_data",
                    "conflict_type": "INCOMPLETE_EXTRACTION",
                    "confidence": "LOW",
                    "recommendation": f"Only {completeness_check['completeness_rate']}% of important qualitative fields were extracted. Missing fields: {', '.join(completeness_check['missing_fields'][:5])}{'...' if len(completeness_check['missing_fields']) > 5 else ''}"
                })

                # Update reconciliation notes
                notes = result.get("reconciliation_notes", "")
                extra_note = f"Qualitative extraction incomplete ({completeness_check['completeness_rate']}% complete). Needs human review to fill missing fields."
                result["reconciliation_notes"] = f"{notes} {extra_note}".strip() if notes else extra_note

            return result

        except Exception as e:
            logger.error(f"Error in reconciliation: {e}")
            # Default to flagging everything for human review
            return {
                "final_quantitative_data": extractor1_quant,  # Use first extractor as fallback
                "final_qualitative_data": extractor1_qual,
                "field_confidence": {},  # No confidence scores available
                "conflicts": [{
                    "field": "all",
                    "conflict_type": "RECONCILIATION_ERROR",
                    "confidence": "LOW",
                    "recommendation": f"Reconciliation failed: {e}"
                }],
                "auto_resolved": [],
                "needs_human_review": True,
                "overall_confidence": "LOW",
                "reconciliation_notes": f"Automatic reconciliation failed: {e}. Using Extractor 1 results as placeholder."
            }

    def _notes_indicate_missing_effect(self, text: Optional[str]) -> bool:
        """Return True if text contains phrases indicating no effect size was reported."""
        if not text:
            return False
        lower_text = text.lower()
        return any(pattern in lower_text for pattern in self.EFFECT_SIZE_NOTE_PATTERNS)

    def _check_qualitative_completeness(self, qual_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if qualitative data is mostly empty/missing.

        Returns dict with:
        - is_incomplete: bool - True if most fields are "not reported"
        - missing_count: int - Number of fields that are "not reported"
        - total_fields: int - Total number of important fields checked
        - missing_fields: list - List of field names that are missing
        """
        # Important fields that should be extracted
        important_fields = [
            "study_location",
            "dominant_ecosystem",
            "focal_taxa",
            "main_response_variable",
            "study_type",
            "niche_metrics",
            "specialist_vs_generalist",
            "primary_climatic_driver",
            "niche_change_direction"
        ]

        missing_fields = []
        for field in important_fields:
            value = str(qual_data.get(field, "")).strip().lower()
            if value in ["not reported", "not found", "na", "none", ""]:
                missing_fields.append(field)

        missing_count = len(missing_fields)
        total_fields = len(important_fields)

        # Flag as incomplete if more than 60% of important fields are missing
        is_incomplete = (missing_count / total_fields) > 0.6

        return {
            "is_incomplete": is_incomplete,
            "missing_count": missing_count,
            "total_fields": total_fields,
            "missing_fields": missing_fields,
            "completeness_rate": round((total_fields - missing_count) / total_fields * 100, 1)
        }

    def _flag_effect_size_contradictions(
        self,
        reconciliation_result: Dict[str, Any],
        extractor1_quant: Dict[str, Any],
        extractor2_quant: Dict[str, Any]
    ) -> None:
        """Add conflict if notes state no effect size but a value exists."""
        final_quant = reconciliation_result.setdefault("final_quantitative_data", {})
        effect_value = final_quant.get("effect_size")
        if not effect_value or str(effect_value).strip().lower() in {"", "not found"}:
            return

        conflicts = reconciliation_result.setdefault("conflicts", [])
        field_conf = reconciliation_result.setdefault("field_confidence", {})

        flagged_extractors = []
        for idx, quant in enumerate([extractor1_quant, extractor2_quant], start=1):
            if not isinstance(quant, dict):
                continue
            candidate_text = " ".join([
                str(quant.get("notes", "") or ""),
                str(quant.get("represents", "") or "")
            ]).strip()
            if self._notes_indicate_missing_effect(candidate_text):
                flagged_extractors.append((idx, candidate_text))

        if not flagged_extractors:
            return

        # Add one consolidated conflict referencing the first flagged extractor
        flagged_idx, flagged_text = flagged_extractors[0]
        conflicts.append({
            "field": "effect_size",
            "extractor1_value": extractor1_quant.get("effect_size", "not found"),
            "extractor2_value": extractor2_quant.get("effect_size", "not found"),
            "conflict_type": "EFFECT_SIZE_NOTE_CONTRADICTION",
            "confidence": "LOW",
            "recommendation": (
                f"Extractor {flagged_idx} notes state no quantitative effect sizes were reported "
                f"(excerpt: \"{flagged_text[:200]}\"). Verify effect_size={effect_value} manually."
            )
        })

        reconciliation_result["needs_human_review"] = True
        field_conf["effect_size"] = "LOW"
        notes = reconciliation_result.get("reconciliation_notes", "")
        extra_note = (
            "Effect size flagged: extractor notes describe no quantitative effect sizes, "
            "but a numeric value was extracted."
        )
        reconciliation_result["reconciliation_notes"] = (
            f"{notes} {extra_note}".strip() if notes else extra_note
        )


# ============================================================================
# MAIN WORKFLOW ORCHESTRATION
# ============================================================================

class DataExtractionPipeline:
    """Main pipeline orchestrating screening and extraction phases with hybrid LLM support"""

    def __init__(
        self,
        text_llm_provider: LLMProvider,
        pdf_processor: PDFProcessor,
        vision_llm_provider: Optional[LLMProvider] = None,
        use_vision: bool = True,
        save_extracted_images: bool = False
    ):
        """
        Args:
            text_llm_provider: LLM for text-only tasks (screening, text extraction)
            pdf_processor: PDF processing utility
            vision_llm_provider: Optional separate LLM for vision tasks (figure extraction)
                                If None, will use text_llm_provider if it supports vision
            use_vision: Whether to enable figure extraction
            save_extracted_images: Whether to save extracted images to disk for debugging
        """
        self.text_llm = text_llm_provider
        self.pdf_processor = pdf_processor
        self.use_vision = use_vision
        self.save_extracted_images = save_extracted_images

        # Determine vision LLM
        if vision_llm_provider:
            self.vision_llm = vision_llm_provider
            logger.info(f"Using hybrid LLM setup: Text={text_llm_provider.model_name}, Vision={vision_llm_provider.model_name}")
        elif text_llm_provider.supports_vision and use_vision:
            self.vision_llm = text_llm_provider
            logger.info(f"Using single vision-capable LLM: {text_llm_provider.model_name}")
        else:
            self.vision_llm = None
            if use_vision:
                logger.warning("Vision extraction requested but no vision-capable LLM provided. Figure extraction will be skipped.")

        # Phase 1: Screening agents (text-only)
        self.screening_reviewer1 = ScreeningReviewerAgent("Reviewer_1", text_llm_provider)
        self.screening_reviewer2 = ScreeningReviewerAgent("Reviewer_2", text_llm_provider)
        self.screening_adjudicator = ScreeningAdjudicatorAgent(text_llm_provider)

        # Phase 2: Text extraction agents (dual extraction)
        self.quant_extractor1 = QuantitativeExtractionAgent(text_llm_provider, pdf_processor.embeddings)
        self.quant_extractor2 = QuantitativeExtractionAgent(text_llm_provider, pdf_processor.embeddings)
        self.qual_extractor1 = QualitativeExtractionAgent(text_llm_provider, pdf_processor.embeddings)
        self.qual_extractor2 = QualitativeExtractionAgent(text_llm_provider, pdf_processor.embeddings)
        self.extraction_reconciler = ExtractionReconciliationAgent(text_llm_provider, pdf_processor.embeddings)

        # Phase 2: Figure extraction agent (vision-based)
        self.figure_extractor = None
        if self.vision_llm and use_vision:
            prompts = _get_prompts()
            figure_prompt = prompts["extraction"].get("figure_extraction")
            self.figure_extractor = FigureExtractionAgent(
                self.vision_llm,
                pdf_processor,
                prompt=figure_prompt
            )

    def screen_paper(self, pdf_path: str) -> ScreeningResult:
        """Phase 1: Screen a paper for inclusion/exclusion"""

        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 1: SCREENING - {Path(pdf_path).name}")
        logger.info(f"{'='*60}")

        # Prepare PDF for screening
        full_text, metadata, relevant_chunks = self.pdf_processor.prepare_text_for_screening(pdf_path)

        if not relevant_chunks:
            logger.warning("No relevant chunks found, using first page")
            first_page = self.pdf_processor.extract_first_page(pdf_path)
            relevant_chunks = [first_page]

        # Reviewer 1 assessment
        logger.info(f"  {self.screening_reviewer1.agent_id} reviewing...")
        assessment1 = self.screening_reviewer1.review_paper(relevant_chunks, metadata)
        logger.info(f"    Decision: {assessment1.decision}, Confidence: {assessment1.confidence}")

        # Reviewer 2 assessment
        logger.info(f"  {self.screening_reviewer2.agent_id} reviewing...")
        assessment2 = self.screening_reviewer2.review_paper(relevant_chunks, metadata)
        logger.info(f"    Decision: {assessment2.decision}, Confidence: {assessment2.confidence}")

        # Adjudication
        logger.info(f"  Adjudicating...")
        adjudication = self.screening_adjudicator.adjudicate(metadata, assessment1, assessment2)

        # Compile result
        result = ScreeningResult(
            paper_id=metadata.get("paper_id", Path(pdf_path).stem),
            paper_title=metadata.get("title", "Unknown"),
            paper_authors=metadata.get("authors", "Unknown"),
            paper_year=metadata.get("year"),
            reviewer1=assessment1,
            reviewer2=assessment2,
            final_decision=Decision(adjudication["final_decision"]),
            final_exclusion_reason=ExclusionCode(adjudication["final_exclusion_reason"]) if adjudication.get("final_exclusion_reason") else None,
            adjudication_notes=adjudication["adjudication_notes"],
            needs_human_review=adjudication["needs_human_review"],
            pdf_path=str(pdf_path)
        )

        logger.info(f"  FINAL DECISION: {result.final_decision}")
        if result.needs_human_review:
            logger.info(f"  FLAGGED FOR HUMAN REVIEW")

        return result

    def extract_data(
        self,
        pdf_path: str,
        extract_quantitative: bool = True,
        extract_qualitative: bool = True
    ) -> ExtractionResult:
        """Phase 2: Extract data from a paper using dual extraction + optional vision"""

        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 2: DATA EXTRACTION (DUAL EXTRACTORS + VISION) - {Path(pdf_path).name}")
        logger.info(f"{'='*60}")

        # Prepare PDF for extraction
        full_text, sections_dict, quant_chunks, qual_chunks = self.pdf_processor.prepare_text_for_extraction(pdf_path)

        # Extract metadata (title, authors, year) from PDF
        logger.info("Extracting paper metadata...")
        metadata = self.pdf_processor.extract_metadata(pdf_path, full_text)

        # EXTRACTOR 1: Extract quantitative data from text
        logger.info("\n--- Extractor 1: Quantitative Extraction (Text) ---")
        quant_data1 = {}
        if extract_quantitative and quant_chunks:
            quant_data1 = self.quant_extractor1.extract(quant_chunks)
        elif not extract_quantitative:
            logger.info("Quantitative extraction disabled by configuration")
        else:
            logger.info("No chunks available for quantitative extraction")

        # EXTRACTOR 1: Extract qualitative data
        logger.info("\n--- Extractor 1: Qualitative Extraction ---")
        qual_data1 = {}
        if extract_qualitative and qual_chunks:
            qual_data1 = self.qual_extractor1.extract(qual_chunks)
        elif not extract_qualitative:
            logger.info("Qualitative extraction disabled by configuration")
        else:
            logger.info("No chunks available for qualitative extraction")

        # EXTRACTOR 2: Extract quantitative data from text
        logger.info("\n--- Extractor 2: Quantitative Extraction (Text) ---")
        quant_data2 = {}
        if extract_quantitative and quant_chunks:
            quant_data2 = self.quant_extractor2.extract(quant_chunks)
        elif not extract_quantitative:
            logger.info("Quantitative extraction disabled by configuration")
        else:
            logger.info("No chunks available for quantitative extraction")

        # EXTRACTOR 2: Extract qualitative data
        logger.info("\n--- Extractor 2: Qualitative Extraction ---")
        qual_data2 = {}
        if extract_qualitative and qual_chunks:
            qual_data2 = self.qual_extractor2.extract(qual_chunks)
        elif not extract_qualitative:
            logger.info("Qualitative extraction disabled by configuration")
        else:
            logger.info("No chunks available for qualitative extraction")

        # VISION EXTRACTION: Extract quantitative data from figures (if enabled)
        figure_data = {}
        if extract_quantitative and self.figure_extractor:
            logger.info("\n--- Vision Extractor: Figure Analysis ---")
            try:
                figure_extractions = self.figure_extractor.extract_from_pdf(
                    pdf_path,
                    save_images=self.save_extracted_images
                )
                if figure_extractions:
                    figure_data = self.figure_extractor.merge_figure_extractions(figure_extractions)
                    logger.info(f"  Extracted data from {len(figure_extractions)} figures")

                    # Merge figure data with text-extracted quantitative data
                    # Skip merging if using array-based extraction (handled differently)
                    if isinstance(quant_data1, dict) and isinstance(quant_data2, dict):
                        # Legacy dict-based merging
                        # Figures supplement text data (prefer text if both exist, but note figure data)
                        for key in ["effect_size", "sample_size", "variance_value", "variance_type"]:
                            fig_val = figure_data.get(key, "not found")
                            if fig_val and fig_val != "not found":
                                # Add figure data to notes for both extractors
                                if quant_data1.get(key, "not found") == "not found":
                                    quant_data1[key] = fig_val
                                if quant_data2.get(key, "not found") == "not found":
                                    quant_data2[key] = fig_val

                                # Add note about figure source
                                for qdata in [quant_data1, quant_data2]:
                                    existing_notes = qdata.get("notes", "")
                                    fig_note = f" [Figures: {key}={fig_val}]"
                                    qdata["notes"] = existing_notes + fig_note
                    else:
                        logger.info("Skipping figure data merge for array-based quantitative extraction")

                else:
                    logger.info("  No figures found or no quantitative data in figures")
            except Exception as e:
                logger.error(f"Error during figure extraction: {e}")
                logger.info("Continuing with text-only extraction")
        elif self.figure_extractor:
            logger.info("\n--- Vision extractor available but quantitative extraction disabled ---")
        else:
            logger.info("\n--- Vision extraction disabled ---")

        # Create extractor assessments
        assessment1 = ExtractionAssessment(
            extractor_id="Extractor_1",
            quantitative_data=quant_data1,
            qualitative_data=qual_data1,
            confidence="Medium",
            notes="First independent extraction (text + figures)"
        )

        assessment2 = ExtractionAssessment(
            extractor_id="Extractor_2",
            quantitative_data=quant_data2,
            qualitative_data=qual_data2,
            confidence="Medium",
            notes="Second independent extraction (text + figures)"
        )

        # RECONCILIATION: Compare and reconcile results
        logger.info("\n--- Reconciliation Agent: Comparing Extractions ---")
        reconciliation = self.extraction_reconciler.reconcile(
            quant_data1, quant_data2,
            qual_data1, qual_data2
        )

        # Compile final extraction result
        result = ExtractionResult(
            paper_id=metadata.get("paper_id", Path(pdf_path).stem),
            extractor1=assessment1,
            extractor2=assessment2,
            final_quantitative_data=reconciliation.get("final_quantitative_data", []),
            final_qualitative_data=reconciliation.get("final_qualitative_data", {}),
            field_confidence=reconciliation.get("field_confidence", {}),
            overall_confidence=reconciliation.get("overall_confidence", "MEDIUM"),
            conflicts=reconciliation.get("conflicts", []),
            auto_resolved=reconciliation.get("auto_resolved", []),
            needs_human_review=reconciliation.get("needs_human_review", False),
            reconciliation_notes=reconciliation.get("reconciliation_notes", ""),
            paper_title=metadata.get("title"),
            paper_authors=metadata.get("authors"),
            paper_year=metadata.get("year"),
            paper_doi=metadata.get("doi")
        )

        logger.info(f"\n✓ Data extraction complete for {Path(pdf_path).name}")
        logger.info(f"  Overall Confidence: {result.overall_confidence}")
        logger.info(f"  Auto-resolved fields: {len(result.auto_resolved)}")
        logger.info(f"  Conflicts detected: {len(result.conflicts)}")
        if result.needs_human_review:
            logger.info(f"  ⚠️ FLAGGED FOR HUMAN REVIEW")

        return result

    def process_directory(
        self,
        pdf_dir: str,
        output_dir: str,
        screen_only: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Process all PDFs in a directory through the full pipeline

        Args:
            pdf_dir: Directory containing PDF files
            output_dir: Directory for output files
            screen_only: If True, only run screening phase

        Returns:
            (screening_df, extraction_df) - DataFrames with results
        """

        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING {len(pdf_files)} PDF FILES")
        logger.info(f"{'='*60}")

        screening_results = []
        extraction_results = []

        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")

            try:
                # Phase 1: Screening
                screening_result = self.screen_paper(str(pdf_path))
                screening_results.append(screening_result)

                # Phase 2: Extraction (only if INCLUDE)
                if not screen_only and screening_result.final_decision == Decision.INCLUDE:
                    extraction_data = self.extract_data(str(pdf_path))
                    extraction_results.append(extraction_data)

            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                continue

        # Save screening results
        os.makedirs(output_dir, exist_ok=True)

        screening_df = self._screening_results_to_dataframe(screening_results)
        today_str = date.today().strftime("%d%b")
        screening_csv = os.path.join(output_dir, f"screening_results_{today_str}.csv")
        screening_df.to_csv(screening_csv, index=False)
        logger.info(f"\n✓ Screening results saved to {screening_csv}")

        # Save extraction results
        extraction_df = None
        if extraction_results:
            extraction_df = self._extraction_results_to_dataframe(extraction_results)
            extraction_csv = os.path.join(output_dir, f"extraction_results_{today_str}.csv")
            extraction_df.to_csv(extraction_csv, index=False)
            logger.info(f"✓ Extraction results saved to {extraction_csv}")

            # Also save detailed extraction results with conflicts as JSON
            extraction_json = os.path.join(output_dir, f"extraction_detailed_{today_str}.json")
            detailed_results = [result.model_dump() for result in extraction_results]
            with open(extraction_json, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Detailed extraction results (with conflicts) saved to {extraction_json}")

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total papers processed: {len(screening_results)}")
        logger.info(f"INCLUDE: {sum(1 for r in screening_results if r.final_decision == Decision.INCLUDE)}")
        logger.info(f"EXCLUDE: {sum(1 for r in screening_results if r.final_decision == Decision.EXCLUDE)}")
        logger.info(f"DISCUSS: {sum(1 for r in screening_results if r.final_decision == Decision.DISCUSS)}")
        logger.info(f"Flagged for human review (screening): {sum(1 for r in screening_results if r.needs_human_review)}")
        if extraction_df is not None:
            logger.info(f"Data extracted from: {len(extraction_results)} papers")
            logger.info(f"Extraction conflicts detected: {sum(len(r.conflicts) for r in extraction_results)}")
            logger.info(f"Flagged for human review (extraction): {sum(1 for r in extraction_results if r.needs_human_review)}")
        logger.info(f"{'='*60}")

        return screening_df, extraction_df

    def _screening_results_to_dataframe(self, results: List[ScreeningResult]) -> pd.DataFrame:
        """Convert screening results to a flat DataFrame"""
        rows = []

        for result in results:
            row = {
                "paper_id": result.paper_id,
                "title": result.paper_title,
                "authors": result.paper_authors,
                "year": result.paper_year,
                "pdf_path": result.pdf_path,

                # Reviewer 1
                "r1_decision": result.reviewer1.decision.value,
                "r1_exclusion_reason": result.reviewer1.exclusion_reason.value if result.reviewer1.exclusion_reason else None,
                "r1_confidence": result.reviewer1.confidence,
                "r1_notes": result.reviewer1.notes,

                # Reviewer 2
                "r2_decision": result.reviewer2.decision.value,
                "r2_exclusion_reason": result.reviewer2.exclusion_reason.value if result.reviewer2.exclusion_reason else None,
                "r2_confidence": result.reviewer2.confidence,
                "r2_notes": result.reviewer2.notes,

                # Final
                "final_decision": result.final_decision.value,
                "final_exclusion_reason": result.final_exclusion_reason.value if result.final_exclusion_reason else None,
                "adjudication_notes": result.adjudication_notes,
                "needs_human_review": result.needs_human_review,

                "screening_date": result.screening_date
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def _extraction_results_to_dataframe(self, results: List[ExtractionResult]) -> pd.DataFrame:
        """Convert extraction results to a flat DataFrame"""
        rows = []

        for result in results:
            # Combine final quantitative and qualitative data
            combined_data = {
                "paper_id": result.paper_id,
                "Filename": f"{result.paper_id}.pdf",
                **result.final_qualitative_data,
                **result.final_quantitative_data,

                # Add extraction metadata
                "overall_confidence": result.overall_confidence,
                "needs_human_review": result.needs_human_review,
                "num_conflicts": len(result.conflicts),
                "num_auto_resolved": len(result.auto_resolved),
                "reconciliation_notes": result.reconciliation_notes,
                "extraction_date": result.extraction_date,

                # Add both extractor confidence scores
                "extractor1_confidence": result.extractor1.confidence,
                "extractor2_confidence": result.extractor2.confidence,
            }

            rows.append(combined_data)

        return pd.DataFrame(rows)


# ============================================================================
# CLI
# ============================================================================

def create_llm_provider(
    provider: str,
    model: str,
    ollama_base_url: str = "http://localhost:11434",
    ollama_cloud: bool = False
) -> LLMProvider:
    """Factory function to create LLM provider"""

    if provider == "anthropic":
        return AnthropicProvider(model_name=model)
    elif provider == "openai":
        return OpenAIProvider(model_name=model)
    elif provider == "ollama":
        if ollama_cloud:
            api_key = os.environ.get("OLLAMA_API_KEY")
            return OllamaProvider(
                model_name=model,
                base_url="https://ollama.com",
                api_key=api_key
            )
        else:
            return OllamaProvider(
                model_name=model,
                base_url=ollama_base_url
            )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidated Data Extraction and Screening Pipeline for Meta-Analysis with Vision Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hybrid: Gemma3:4b for text + Gemma3:27b for vision (RECOMMENDED for GPU)
  python data_extraction.py --pdf-dir ./papers --output-dir ./results \\
    --provider ollama --model gemma3:4b \\
    --vision-provider ollama --vision-model gemma3:27b

  # Single vision-capable model (Claude with vision)
  python data_extraction.py --pdf-dir ./papers --output-dir ./results \\
    --provider anthropic --model claude-3-5-sonnet-20241022

  # No vision (text-only extraction)
  python data_extraction.py --pdf-dir ./papers --output-dir ./results \\
    --provider ollama --model llama3.2 --no-vision

  # Screening only (no extraction)
  python data_extraction.py --pdf-dir ./papers --output-dir ./results \\
    --provider ollama --model gemma3:4b --screen-only

  # Save extracted images for debugging
  python data_extraction.py --pdf-dir ./papers --output-dir ./results \\
    --provider ollama --model gemma3:4b --vision-model gemma3:27b --save-images
        """
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=DEFAULT_PDF_DIR,
        help=f"Directory containing PDF files to process (default: {DEFAULT_PDF_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})"
    )

    # Text LLM configuration
    parser.add_argument(
        "--provider",
        type=str,
        default=DEFAULT_PROVIDER,
        choices=["anthropic", "openai", "ollama"],
        help=f"LLM provider for text processing (default: {DEFAULT_PROVIDER})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,  # Will use DEFAULT_TEXT_MODEL if not specified
        help=f"Text model name (default: {DEFAULT_TEXT_MODEL} for ollama)"
    )

    # Vision LLM configuration (optional hybrid setup)
    parser.add_argument(
        "--vision-provider",
        type=str,
        choices=["anthropic", "openai", "ollama"],
        default=None,  # Will use same as --provider if not specified
        help="Separate LLM provider for vision (optional, for hybrid setup)"
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default=None,  # Will use DEFAULT_VISION_MODEL if not specified
        help=f"Vision model name (default: {DEFAULT_VISION_MODEL} for ollama)"
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        default=not DEFAULT_USE_VISION,
        help=f"Disable vision-based figure extraction (default: {'disabled' if not DEFAULT_USE_VISION else 'enabled'})"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        default=DEFAULT_SAVE_IMAGES,
        help=f"Save extracted images to disk for debugging (default: {DEFAULT_SAVE_IMAGES})"
    )

    # Ollama-specific options
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=DEFAULT_OLLAMA_BASE_URL,
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_BASE_URL})"
    )
    parser.add_argument(
        "--ollama-cloud",
        action="store_true",
        default=DEFAULT_OLLAMA_CLOUD,
        help=f"Use Ollama Cloud instead of local (default: {DEFAULT_OLLAMA_CLOUD})"
    )

    # Pipeline options
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="Only run screening phase (skip data extraction)"
    )

    args = parser.parse_args()

    # Set default models if not specified
    if not args.model:
        if args.provider == "anthropic":
            args.model = "claude-3-5-haiku-20241022"
        elif args.provider == "openai":
            args.model = "gpt-4o"
        elif args.provider == "ollama":
            args.model = DEFAULT_TEXT_MODEL  # Default to Gemma3 4B for text

    # Create text LLM provider
    logger.info(f"\n{'='*60}")
    logger.info(f"CONFIGURING LLM PROVIDERS")
    logger.info(f"{'='*60}")
    logger.info(f"Text LLM: {args.provider} / {args.model}")

    if args.provider == "ollama":
        if args.ollama_cloud:
            logger.info("  Using Ollama Cloud (requires OLLAMA_API_KEY)")
        else:
            logger.info(f"  Using local Ollama at {args.ollama_base_url}")

    text_llm_provider = create_llm_provider(
        args.provider,
        args.model,
        ollama_base_url=args.ollama_base_url,
        ollama_cloud=args.ollama_cloud
    )

    # Create vision LLM provider (if specified)
    vision_llm_provider = None
    use_vision = not args.no_vision

    if use_vision and not args.vision_model:
        if (args.vision_provider in (None, "ollama")) and args.provider == "ollama":
            args.vision_provider = args.vision_provider or "ollama"
            args.vision_model = DEFAULT_VISION_MODEL
            logger.info(f"Vision model not specified; defaulting to {DEFAULT_VISION_MODEL} on Ollama")

    if use_vision and args.vision_provider and args.vision_model:
        # Hybrid setup: separate vision LLM
        logger.info(f"Vision LLM: {args.vision_provider} / {args.vision_model}")
        vision_llm_provider = create_llm_provider(
            args.vision_provider,
            args.vision_model,
            ollama_base_url=args.ollama_base_url,
            ollama_cloud=args.ollama_cloud
        )
    elif use_vision and text_llm_provider.supports_vision:
        # Single vision-capable LLM
        logger.info(f"Vision: Using same model as text ({args.model} supports vision)")
    elif use_vision:
        logger.warning("Vision extraction requested but no vision-capable LLM configured.")
        logger.warning("Use --vision-provider and --vision-model for hybrid setup, or use a vision-capable text model.")
    else:
        logger.info("Vision extraction disabled (--no-vision)")

    # Create PDF processor
    pdf_processor = PDFProcessor()

    # Create pipeline with hybrid LLM support
    pipeline = DataExtractionPipeline(
        text_llm_provider=text_llm_provider,
        pdf_processor=pdf_processor,
        vision_llm_provider=vision_llm_provider,
        use_vision=use_vision,
        save_extracted_images=args.save_images
    )

    # Process directory
    logger.info(f"\n{'='*60}")
    logger.info(f"STARTING PIPELINE")
    logger.info(f"{'='*60}\n")

    screening_df, extraction_df = pipeline.process_directory(
        args.pdf_dir,
        args.output_dir,
        screen_only=args.screen_only
    )


if __name__ == "__main__":
    main()
