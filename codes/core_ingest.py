#!/usr/bin/env python3
"""
CORE Full-Text Ingestion Pipeline
AI × Civil Engineering Academic Paper Collection

Searches CORE (core.ac.uk) for open-access papers at the intersection of 
AI and Civil Engineering, downloads PDFs, extracts full text, validates 
content, and outputs normalized JSON.

Usage:
    export CORE_API_KEY="your-api-key"
    python core_ingest.py --max-papers 50 --output data/core_corpus.jsonl
"""

import argparse
import json
import logging
import os
import re
import ssl
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import urllib.request

import requests
import fitz  # PyMuPDF
import yaml

# Fix SSL certificate verification on macOS
try:
    import certifi
    ssl_context = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_DIR = Path(__file__).parent
TAXONOMY_FILE = PROJECT_DIR / "taxonomy.yml"
PDF_CACHE_DIR = PROJECT_DIR / "data" / "core_pdfs"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "core_corpus.jsonl"
MIN_TEXT_LENGTH = 3000
API_DELAY_SECONDS = 2.0  # CORE rate limit friendly
CORE_API_BASE = "https://api.core.ac.uk/v3"


@dataclass
class CorePaper:
    """Represents a processed CORE paper."""
    doc_id: str
    source: str = "CORE"
    repository: str = ""
    url: str = ""
    full_text_url: str = ""
    title: str = ""
    authors: list = field(default_factory=list)
    year: str = ""
    document_type: str = ""
    full_text: str = ""
    summary: str = ""
    ai_methods: list = field(default_factory=list)
    civil_engineering_domain: list = field(default_factory=list)
    data_sources: list = field(default_factory=list)
    keywords: list = field(default_factory=list)


def get_api_key() -> str:
    """Get CORE API key from environment."""
    key = os.environ.get("CORE_API_KEY", "")
    if not key:
        raise ValueError(
            "CORE_API_KEY environment variable not set. "
            "Register at https://core.ac.uk/register and request API access."
        )
    return key


def load_taxonomy(taxonomy_path: Path) -> dict:
    """Load taxonomy keywords from YAML file."""
    if not taxonomy_path.exists():
        logger.warning(f"Taxonomy file not found: {taxonomy_path}")
        return {"civil_engineering_areas": {}, "ai_technologies": {}}
    
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_search_queries() -> list[str]:
    """Build CORE search queries combining CE domains with AI terms."""
    queries = [
        # Structural Engineering + AI
        '"structural health monitoring" AND "deep learning"',
        '"structural engineering" AND "machine learning"',
        '"crack detection" AND "neural network"',
        '"bridge inspection" AND "machine learning"',
        '"seismic" AND "deep learning"',
        '"damage detection" AND "convolutional neural network"',
        
        # Geotechnical Engineering + AI
        '"slope stability" AND "machine learning"',
        '"geotechnical" AND "neural network"',
        '"landslide prediction" AND "deep learning"',
        '"soil" AND "machine learning" AND "prediction"',
        
        # Transportation Engineering + AI
        '"traffic prediction" AND "deep learning"',
        '"autonomous vehicle" AND "reinforcement learning"',
        '"pavement" AND "computer vision"',
        '"transportation" AND "graph neural network"',
        
        # Construction Management + AI
        '"construction" AND "computer vision" AND "safety"',
        '"BIM" AND "machine learning"',
        '"construction site" AND "object detection"',
        '"project scheduling" AND "optimization"',
        
        # Water Resources / Environmental + AI
        '"flood prediction" AND "deep learning"',
        '"water quality" AND "machine learning"',
        '"hydrological" AND "neural network"',
        
        # Infrastructure / Smart Cities + AI
        '"infrastructure monitoring" AND "deep learning"',
        '"smart city" AND "machine learning"',
        '"digital twin" AND "civil engineering"',
        
        # Materials + AI
        '"concrete" AND "machine learning" AND "strength"',
        '"asphalt" AND "deep learning"',
    ]
    return queries


def search_core(api_key: str, queries: list[str], max_results_per_query: int = 20) -> list[dict]:
    """Search CORE API with multiple queries and deduplicate results."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    seen_ids = set()
    all_results = []
    
    for query in queries:
        logger.info(f"Searching CORE: {query[:60]}...")
        try:
            # Use CORE APIv3 search endpoint
            params = {
                "q": query,
                "limit": max_results_per_query,
                "scroll": "false"
            }
            
            response = requests.get(
                f"{CORE_API_BASE}/search/works",
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 401:
                logger.error("API authentication failed. Check your CORE_API_KEY.")
                return []
            elif response.status_code == 429:
                logger.warning("Rate limited. Waiting 60 seconds...")
                time.sleep(60)
                continue
            elif response.status_code != 200:
                logger.warning(f"API error {response.status_code}: {response.text[:200]}")
                continue
            
            data = response.json()
            results = data.get("results", [])
            
            for result in results:
                core_id = str(result.get("id", ""))
                if not core_id or core_id in seen_ids:
                    continue
                
                # Filter: must have download URL (full text access)
                download_url = result.get("downloadUrl", "")
                if not download_url:
                    continue
                
                # Filter: year 2015+
                year_published = result.get("yearPublished")
                if year_published and int(year_published) < 2015:
                    continue
                
                # Filter: must be open access
                # CORE typically only returns OA content, but double-check
                
                seen_ids.add(core_id)
                all_results.append(result)
            
            time.sleep(API_DELAY_SECONDS)  # Respect rate limits
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout for query: {query[:40]}...")
            continue
        except Exception as e:
            logger.error(f"Search failed for query '{query[:40]}...': {e}")
            continue
    
    logger.info(f"Found {len(all_results)} unique papers with full-text URLs from {len(queries)} queries")
    return all_results


def download_pdf(download_url: str, core_id: str, cache_dir: Path) -> Optional[Path]:
    """Download PDF to cache directory using custom SSL context."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = cache_dir / f"core_{core_id}.pdf"
    
    if pdf_path.exists():
        logger.debug(f"Using cached PDF: {pdf_path.name}")
        return pdf_path
    
    try:
        req = urllib.request.Request(
            download_url,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; core-ingest/1.0)'}
        )
        
        with urllib.request.urlopen(req, context=ssl_context, timeout=60) as response:
            content_type = response.headers.get('Content-Type', '')
            
            # Check if it's actually a PDF
            if 'pdf' not in content_type.lower() and not download_url.endswith('.pdf'):
                # Try to download anyway, might still be PDF
                pass
            
            pdf_data = response.read()
        
        # Verify it's a PDF (check magic bytes)
        if not pdf_data.startswith(b'%PDF'):
            logger.debug(f"Not a valid PDF: {core_id}")
            return None
        
        with open(pdf_path, 'wb') as f:
            f.write(pdf_data)
        
        logger.info(f"Downloaded: {pdf_path.name}")
        time.sleep(1)  # Brief delay after download
        return pdf_path
        
    except Exception as e:
        logger.error(f"Failed to download PDF for {core_id}: {e}")
        return None


def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """Extract text from PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page in doc:
            text_parts.append(page.get_text())
        
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"PDF extraction failed for {pdf_path.name}: {e}")
        return None


def remove_references(text: str) -> str:
    """Remove references/bibliography section from text."""
    ref_patterns = [
        r'\n\s*References?\s*\n',
        r'\n\s*REFERENCES?\s*\n',
        r'\n\s*Bibliography\s*\n',
        r'\n\s*BIBLIOGRAPHY\s*\n',
        r'\n\s*Works Cited\s*\n',
    ]
    
    for pattern in ref_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            break
    
    return text


def remove_appendices(text: str) -> str:
    """Remove appendices section from text."""
    appendix_patterns = [
        r'\n\s*Appendix\s*[A-Z]?\s*\n',
        r'\n\s*APPENDIX\s*[A-Z]?\s*\n',
        r'\n\s*Appendices\s*\n',
        r'\n\s*APPENDICES\s*\n',
        r'\n\s*Supplementary\s+Materials?\s*\n',
    ]
    
    for pattern in appendix_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            break
    
    return text


def parse_sections(text: str) -> dict[str, int]:
    """Parse text into sections based on common headers."""
    sections = {}
    
    section_patterns = [
        (r'(?:^|\n)\s*(?:1\.?\s*)?Introduction\s*\n', 'introduction'),
        (r'(?:^|\n)\s*(?:2\.?\s*)?(?:Related Work|Background|Literature Review)\s*\n', 'background'),
        (r'(?:^|\n)\s*(?:3\.?\s*)?(?:Methodology|Methods?|Approach|Proposed Method)\s*\n', 'methods'),
        (r'(?:^|\n)\s*(?:4\.?\s*)?(?:Experiments?|Results?|Evaluation)\s*\n', 'results'),
        (r'(?:^|\n)\s*(?:5\.?\s*)?(?:Discussion)\s*\n', 'discussion'),
        (r'(?:^|\n)\s*(?:6\.?\s*)?(?:Conclusion|Conclusions?|Summary)\s*\n', 'conclusion'),
    ]
    
    for pattern, section_name in section_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            sections[section_name] = match.start()
    
    return sections


def validate_content(text: str, abstract: str = "") -> bool:
    """Validate extracted text meets quality requirements."""
    if not text:
        return False
    
    # Check minimum length
    if len(text) < MIN_TEXT_LENGTH:
        logger.debug(f"Text too short: {len(text)} < {MIN_TEXT_LENGTH}")
        return False
    
    # Check it's not just the abstract repeated
    if abstract and text.strip() == abstract.strip():
        logger.debug("Text is abstract-only")
        return False
    
    # Check for presence of typical paper content
    sections = parse_sections(text)
    if not sections:
        # At least check for some substance beyond abstract
        if abstract and len(text) < len(abstract) * 3:
            logger.debug("Text appears to be abstract-only (length check)")
            return False
    
    # Check for OCR-only PDFs (very low text density)
    # Heuristic: if mostly whitespace/symbols, likely scanned
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.3:
        logger.debug(f"Low alpha ratio ({alpha_ratio:.2f}), possibly scanned PDF")
        return False
    
    return True


def extract_ai_methods(text: str, taxonomy: dict) -> list[str]:
    """Extract AI methods mentioned in text using taxonomy keywords."""
    methods = set()
    text_lower = text.lower()
    
    ai_tech = taxonomy.get("ai_technologies", {})
    for category, data in ai_tech.items():
        keywords = data.get("keywords", [])
        for keyword in keywords:
            if keyword.lower() in text_lower:
                methods.add(category.replace("_", " "))
                break
    
    return list(methods)[:10]


def extract_ce_domains(text: str, taxonomy: dict) -> list[str]:
    """Extract civil engineering domains mentioned in text."""
    domains = set()
    text_lower = text.lower()
    
    ce_areas = taxonomy.get("civil_engineering_areas", {})
    for category, data in ce_areas.items():
        keywords = data.get("keywords", [])
        for keyword in keywords:
            if keyword.lower() in text_lower:
                domains.add(category.replace("_", " "))
                break
    
    return list(domains)


def extract_data_sources(text: str) -> list[str]:
    """Extract data sources used in the paper."""
    data_sources = set()
    text_lower = text.lower()
    
    source_patterns = {
        "sensor data": ["sensor data", "accelerometer", "strain gauge", "vibration data", "iot sensor", "monitoring data"],
        "images": ["image data", "images", "photographs", "visual inspection", "camera", "imagery"],
        "video": ["video", "footage", "surveillance", "cctv"],
        "point cloud": ["point cloud", "lidar", "3d scan", "laser scanning"],
        "satellite imagery": ["satellite", "remote sensing", "aerial imagery", "uav imagery", "drone"],
        "simulations": ["simulation", "finite element", "numerical model", "synthetic data", "fem", "fea"],
        "tabular data": ["tabular", "csv", "database", "structured data", "spreadsheet", "dataset"],
        "text data": ["text data", "documents", "reports", "nlp", "text mining"],
        "time series": ["time series", "temporal data", "sequential data", "sensor readings"],
        "survey data": ["survey", "questionnaire", "interview", "expert opinion"],
        "mixed": ["multimodal", "multi-source", "heterogeneous data"],
    }
    
    for source, patterns in source_patterns.items():
        for pattern in patterns:
            if pattern in text_lower:
                data_sources.add(source)
                break
    
    return list(data_sources)


def extract_keywords(text: str, title: str, max_keywords: int = 10) -> list[str]:
    """Extract important keywords from paper."""
    keywords = set()
    
    # Common AI/ML keywords
    ai_keywords = [
        "deep learning", "machine learning", "neural network", "CNN", "RNN", "LSTM",
        "transformer", "attention mechanism", "reinforcement learning", "computer vision",
        "object detection", "semantic segmentation", "classification", "regression",
        "optimization", "genetic algorithm", "particle swarm", "random forest", "SVM",
        "GAN", "autoencoder", "transfer learning", "ensemble learning"
    ]
    
    # Common CE keywords
    ce_keywords = [
        "structural health monitoring", "crack detection", "bridge inspection",
        "traffic prediction", "autonomous vehicle", "construction safety",
        "BIM", "digital twin", "flood prediction", "infrastructure",
        "concrete", "pavement", "geotechnical", "earthquake", "seismic",
        "smart city", "transportation", "hydraulic", "foundation"
    ]
    
    text_lower = text.lower()
    title_lower = title.lower()
    
    for kw in ai_keywords + ce_keywords:
        if kw.lower() in text_lower or kw.lower() in title_lower:
            keywords.add(kw)
    
    return list(keywords)[:max_keywords]


def generate_summary(result: dict, text: str, ai_methods: list, ce_domains: list) -> str:
    """Generate a technical summary of the paper."""
    abstract = result.get("abstract", "") or ""
    abstract = abstract.replace("\n", " ").strip()
    
    # Build enhanced summary based on abstract and extracted signals
    summary_parts = []
    
    # Research objective / domain
    if ce_domains:
        summary_parts.append(f"This paper addresses {', '.join(ce_domains[:2])}.")
    
    # Core abstract (truncated)
    if abstract:
        if len(abstract) > 400:
            abstract_short = abstract[:400].rsplit(" ", 1)[0] + "..."
        else:
            abstract_short = abstract
        summary_parts.append(abstract_short)
    
    # AI methodology
    if ai_methods:
        summary_parts.append(f"The methodology employs {', '.join(ai_methods[:3])}.")
    
    summary = " ".join(summary_parts)
    
    # Ensure within 150-250 word range
    words = summary.split()
    if len(words) > 250:
        summary = " ".join(words[:250]) + "..."
    
    return summary


def process_paper(result: dict, taxonomy: dict, cache_dir: Path) -> Optional[CorePaper]:
    """Process a single CORE paper through the full pipeline."""
    core_id = str(result.get("id", ""))
    title = result.get("title", "")
    logger.info(f"Processing: {core_id} - {title[:50]}...")
    
    download_url = result.get("downloadUrl", "")
    if not download_url:
        logger.warning(f"DISCARDED (no download URL): {core_id}")
        return None
    
    # Step 1: Download PDF
    pdf_path = download_pdf(download_url, core_id, cache_dir)
    if not pdf_path:
        logger.warning(f"DISCARDED (download failed): {core_id}")
        return None
    
    # Step 2: Extract text
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        logger.warning(f"DISCARDED (extraction failed): {core_id}")
        return None
    
    # Step 3: Clean text (remove references and appendices)
    clean_text = remove_references(raw_text)
    clean_text = remove_appendices(clean_text)
    
    # Step 4: Validate content
    abstract = result.get("abstract", "") or ""
    if not validate_content(clean_text, abstract):
        logger.warning(f"DISCARDED (validation failed): {core_id}")
        return None
    
    # Step 5: Extract signals
    ai_methods = extract_ai_methods(clean_text, taxonomy)
    ce_domains = extract_ce_domains(clean_text, taxonomy)
    data_sources = extract_data_sources(clean_text)
    keywords = extract_keywords(clean_text, title)
    
    # Step 6: Generate summary
    summary = generate_summary(result, clean_text, ai_methods, ce_domains)
    
    # Extract metadata
    authors = result.get("authors", [])
    if authors:
        author_names = [a.get("name", "") if isinstance(a, dict) else str(a) for a in authors]
    else:
        author_names = []
    
    year = str(result.get("yearPublished", ""))
    
    # Get repository/data provider info
    data_provider = result.get("dataProvider", {})
    if isinstance(data_provider, dict):
        repository = data_provider.get("name", "")
    else:
        repository = str(data_provider) if data_provider else ""
    
    # Document type
    doc_type = result.get("documentType", "") or "article"
    
    # Build paper object
    paper = CorePaper(
        doc_id=f"core:{core_id}",
        repository=repository,
        url=f"https://core.ac.uk/works/{core_id}",
        full_text_url=download_url,
        title=title,
        authors=author_names,
        year=year,
        document_type=doc_type,
        full_text=clean_text,
        summary=summary,
        ai_methods=ai_methods,
        civil_engineering_domain=ce_domains,
        data_sources=data_sources,
        keywords=keywords
    )
    
    logger.info(f"SUCCESS: {core_id} ({len(clean_text)} chars, {len(ai_methods)} AI methods, {len(ce_domains)} CE domains)")
    return paper


def main():
    parser = argparse.ArgumentParser(description="CORE Full-Text Ingestion Pipeline")
    parser.add_argument("--max-papers", type=int, default=50,
                        help="Maximum number of papers to collect (default: 50)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output JSONL file path")
    parser.add_argument("--results-per-query", type=int, default=20,
                        help="Max results per search query (default: 20)")
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("CORE Full-Text Ingestion Pipeline")
    logger.info(f"Target: {args.max_papers} papers")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)
    
    # Get API key
    try:
        api_key = get_api_key()
    except ValueError as e:
        logger.error(str(e))
        return
    
    # Load taxonomy
    taxonomy = load_taxonomy(TAXONOMY_FILE)
    logger.info(f"Loaded taxonomy: {len(taxonomy.get('civil_engineering_areas', {}))} CE areas, "
                f"{len(taxonomy.get('ai_technologies', {}))} AI categories")
    
    # Build and run search queries
    queries = build_search_queries()
    results = search_core(api_key, queries, args.results_per_query)
    
    if not results:
        logger.error("No papers found. Check your API key and queries.")
        return
    
    # Process papers
    processed = 0
    successful = 0
    discarded = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            if successful >= args.max_papers:
                break
            
            processed += 1
            paper = process_paper(result, taxonomy, PDF_CACHE_DIR)
            
            if paper:
                json_line = json.dumps(asdict(paper), ensure_ascii=False)
                f.write(json_line + "\n")
                successful += 1
            else:
                discarded += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Processed: {processed}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Discarded: {discarded}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
