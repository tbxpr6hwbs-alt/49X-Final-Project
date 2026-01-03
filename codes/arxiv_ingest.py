#!/usr/bin/env python3
"""
arXiv Full-Text Ingestion Pipeline
AI × Civil Engineering Academic Paper Collection

Searches arXiv for papers at the intersection of AI and Civil Engineering,
downloads PDFs, extracts full text, validates content, and outputs normalized JSON.

Usage:
    python arxiv_ingest.py --max-papers 50 --output data/arxiv_corpus.jsonl
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

import arxiv
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
PDF_CACHE_DIR = PROJECT_DIR / "data" / "arxiv_pdfs"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "arxiv_corpus.jsonl"
MIN_TEXT_LENGTH = 3000
API_DELAY_SECONDS = 3.0


@dataclass
class ArxivPaper:
    """Represents a processed arXiv paper."""
    doc_id: str
    source: str = "arXiv"
    url: str = ""
    pdf_url: str = ""
    title: str = ""
    authors: list = field(default_factory=list)
    year: str = ""
    categories: list = field(default_factory=list)
    full_text: str = ""
    summary: str = ""
    ai_methods: list = field(default_factory=list)
    civil_engineering_domain: list = field(default_factory=list)
    data_types: list = field(default_factory=list)
    keywords: list = field(default_factory=list)


def load_taxonomy(taxonomy_path: Path) -> dict:
    """Load taxonomy keywords from YAML file."""
    if not taxonomy_path.exists():
        logger.warning(f"Taxonomy file not found: {taxonomy_path}")
        return {"civil_engineering_areas": {}, "ai_technologies": {}}
    
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_search_queries() -> list[str]:
    """Build arXiv search queries combining CE domains with AI terms."""
    queries = [
        # Structural Engineering + AI
        'all:"structural health monitoring" AND all:"deep learning"',
        'all:"structural engineering" AND all:"machine learning"',
        'all:"crack detection" AND all:"neural network"',
        'all:"bridge" AND all:"computer vision" AND all:"inspection"',
        'all:"seismic" AND all:"machine learning"',
        
        # Geotechnical Engineering + AI
        'all:"slope stability" AND all:"machine learning"',
        'all:"geotechnical" AND all:"neural network"',
        'all:"landslide prediction" AND all:"deep learning"',
        'all:"soil" AND all:"machine learning" AND all:"prediction"',
        
        # Transportation Engineering + AI
        'all:"traffic prediction" AND all:"deep learning"',
        'all:"autonomous vehicle" AND all:"reinforcement learning"',
        'all:"pavement" AND all:"computer vision"',
        'all:"transportation" AND all:"graph neural network"',
        
        # Construction Management + AI
        'all:"construction" AND all:"computer vision" AND all:"safety"',
        'all:"BIM" AND all:"machine learning"',
        'all:"construction site" AND all:"object detection"',
        'all:"project scheduling" AND all:"optimization"',
        
        # Water Resources / Environmental + AI
        'all:"flood prediction" AND all:"deep learning"',
        'all:"water quality" AND all:"machine learning"',
        'all:"hydrological" AND all:"neural network"',
        
        # Infrastructure / Smart Cities + AI
        'all:"infrastructure monitoring" AND all:"deep learning"',
        'all:"smart city" AND all:"machine learning"',
        'all:"digital twin" AND all:"civil engineering"',
        
        # Materials + AI
        'all:"concrete" AND all:"machine learning" AND all:"strength"',
        'all:"asphalt" AND all:"deep learning"',
    ]
    return queries


def search_arxiv(queries: list[str], max_results_per_query: int = 20) -> list[arxiv.Result]:
    """Search arXiv with multiple queries and deduplicate results."""
    client = arxiv.Client()
    seen_ids = set()
    all_results = []
    
    for query in queries:
        logger.info(f"Searching: {query[:60]}...")
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results_per_query,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            for result in client.results(search):
                arxiv_id = result.entry_id.split("/")[-1]
                if arxiv_id not in seen_ids:
                    # Filter by year (2015+)
                    pub_year = result.published.year
                    if pub_year >= 2015:
                        seen_ids.add(arxiv_id)
                        all_results.append(result)
            
            time.sleep(API_DELAY_SECONDS)  # Respect rate limits
            
        except Exception as e:
            logger.error(f"Search failed for query '{query[:40]}...': {e}")
            continue
    
    logger.info(f"Found {len(all_results)} unique papers from {len(queries)} queries")
    return all_results


def download_pdf(result: arxiv.Result, cache_dir: Path) -> Optional[Path]:
    """Download PDF to cache directory using custom SSL context."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    arxiv_id = result.entry_id.split("/")[-1].replace("/", "_")
    pdf_path = cache_dir / f"{arxiv_id}.pdf"
    
    if pdf_path.exists():
        logger.debug(f"Using cached PDF: {pdf_path.name}")
        return pdf_path
    
    try:
        # Use urllib with our SSL context to avoid macOS certificate issues
        pdf_url = result.pdf_url
        req = urllib.request.Request(
            pdf_url,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; arxiv-ingest/1.0)'}
        )
        
        with urllib.request.urlopen(req, context=ssl_context, timeout=60) as response:
            pdf_data = response.read()
        
        with open(pdf_path, 'wb') as f:
            f.write(pdf_data)
        
        logger.info(f"Downloaded: {pdf_path.name}")
        time.sleep(1)  # Brief delay after download
        return pdf_path
    except Exception as e:
        logger.error(f"Failed to download PDF for {arxiv_id}: {e}")
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
    # Common reference section headers
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


def parse_sections(text: str) -> dict[str, str]:
    """Parse text into sections based on common headers."""
    sections = {}
    
    # Common section patterns
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


def validate_content(text: str, abstract: str) -> bool:
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
        if len(text) < len(abstract) * 3:
            logger.debug("Text appears to be abstract-only (length check)")
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


def extract_data_types(text: str) -> list[str]:
    """Extract data types used in the paper."""
    data_types = set()
    text_lower = text.lower()
    
    data_patterns = {
        "sensor data": ["sensor data", "accelerometer", "strain gauge", "vibration data", "iot sensor"],
        "images": ["image data", "images", "photographs", "visual inspection", "camera"],
        "video": ["video", "footage", "surveillance"],
        "point cloud": ["point cloud", "lidar", "3d scan"],
        "satellite imagery": ["satellite", "remote sensing", "aerial imagery"],
        "simulations": ["simulation", "finite element", "numerical model", "synthetic data"],
        "tabular data": ["tabular", "csv", "database", "structured data", "spreadsheet"],
        "text data": ["text data", "documents", "reports", "nlp", "text mining"],
        "time series": ["time series", "temporal data", "sequential data"],
    }
    
    for dtype, patterns in data_patterns.items():
        for pattern in patterns:
            if pattern in text_lower:
                data_types.add(dtype)
                break
    
    return list(data_types)


def extract_keywords(text: str, title: str, max_keywords: int = 10) -> list[str]:
    """Extract important keywords from paper."""
    keywords = set()
    
    # Common AI/ML keywords
    ai_keywords = [
        "deep learning", "machine learning", "neural network", "CNN", "RNN", "LSTM",
        "transformer", "attention mechanism", "reinforcement learning", "computer vision",
        "object detection", "semantic segmentation", "classification", "regression",
        "optimization", "genetic algorithm", "particle swarm", "random forest", "SVM"
    ]
    
    # Common CE keywords
    ce_keywords = [
        "structural health monitoring", "crack detection", "bridge inspection",
        "traffic prediction", "autonomous vehicle", "construction safety",
        "BIM", "digital twin", "flood prediction", "infrastructure",
        "concrete", "pavement", "geotechnical", "earthquake", "seismic"
    ]
    
    text_lower = text.lower()
    title_lower = title.lower()
    
    for kw in ai_keywords + ce_keywords:
        if kw.lower() in text_lower or kw.lower() in title_lower:
            keywords.add(kw)
    
    return list(keywords)[:max_keywords]


def generate_summary(result: arxiv.Result, text: str, ai_methods: list, ce_domains: list) -> str:
    """Generate a technical summary of the paper."""
    abstract = result.summary.replace("\n", " ").strip()
    
    # Build enhanced summary based on abstract and extracted signals
    summary_parts = []
    
    # Problem/domain
    if ce_domains:
        summary_parts.append(f"This paper addresses {', '.join(ce_domains[:2])}.")
    
    # Core abstract (truncated)
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


def process_paper(result: arxiv.Result, taxonomy: dict, cache_dir: Path) -> Optional[ArxivPaper]:
    """Process a single arXiv paper through the full pipeline."""
    arxiv_id = result.entry_id.split("/")[-1]
    logger.info(f"Processing: {arxiv_id} - {result.title[:50]}...")
    
    # Step 1: Download PDF
    pdf_path = download_pdf(result, cache_dir)
    if not pdf_path:
        logger.warning(f"DISCARDED (no PDF): {arxiv_id}")
        return None
    
    # Step 2: Extract text
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        logger.warning(f"DISCARDED (extraction failed): {arxiv_id}")
        return None
    
    # Step 3: Clean text (remove references)
    clean_text = remove_references(raw_text)
    
    # Step 4: Validate content
    if not validate_content(clean_text, result.summary):
        logger.warning(f"DISCARDED (validation failed): {arxiv_id}")
        return None
    
    # Step 5: Extract signals
    ai_methods = extract_ai_methods(clean_text, taxonomy)
    ce_domains = extract_ce_domains(clean_text, taxonomy)
    data_types = extract_data_types(clean_text)
    keywords = extract_keywords(clean_text, result.title)
    
    # Step 6: Generate summary
    summary = generate_summary(result, clean_text, ai_methods, ce_domains)
    
    # Build paper object
    paper = ArxivPaper(
        doc_id=f"arxiv:{arxiv_id}",
        url=result.entry_id,
        pdf_url=result.pdf_url,
        title=result.title,
        authors=[author.name for author in result.authors],
        year=str(result.published.year),
        categories=list(result.categories),
        full_text=clean_text,
        summary=summary,
        ai_methods=ai_methods,
        civil_engineering_domain=ce_domains,
        data_types=data_types,
        keywords=keywords
    )
    
    logger.info(f"SUCCESS: {arxiv_id} ({len(clean_text)} chars, {len(ai_methods)} AI methods, {len(ce_domains)} CE domains)")
    return paper


def main():
    parser = argparse.ArgumentParser(description="arXiv Full-Text Ingestion Pipeline")
    parser.add_argument("--max-papers", type=int, default=50,
                        help="Maximum number of papers to collect (default: 50)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output JSONL file path")
    parser.add_argument("--results-per-query", type=int, default=20,
                        help="Max results per search query (default: 20)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip papers needing download (use cache only)")
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("arXiv Full-Text Ingestion Pipeline")
    logger.info(f"Target: {args.max_papers} papers")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)
    
    # Load taxonomy
    taxonomy = load_taxonomy(TAXONOMY_FILE)
    logger.info(f"Loaded taxonomy: {len(taxonomy.get('civil_engineering_areas', {}))} CE areas, "
                f"{len(taxonomy.get('ai_technologies', {}))} AI categories")
    
    # Build and run search queries
    queries = build_search_queries()
    results = search_arxiv(queries, args.results_per_query)
    
    if not results:
        logger.error("No papers found. Check your queries or internet connection.")
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
