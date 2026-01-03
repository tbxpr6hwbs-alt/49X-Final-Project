#!/usr/bin/env python3
"""
Article Collection Pipeline for AI × Civil Engineering Trend Analysis
Automated scraping using search queries from search_queries.txt

This script provides batch collection of articles from multiple sources.
Run with: python collect_articles.py
"""

import json
import hashlib
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

# Configuration
PROJECT_DIR = Path(__file__).parent
QUERIES_FILE = PROJECT_DIR / "search_queries.txt"
URL_QUEUE_FILE = PROJECT_DIR / "url_queue.jsonl"
DATA_DESC_FILE = PROJECT_DIR / "data_description.md"

def load_queries():
    """Load and parse queries from search_queries.txt"""
    queries = []
    current_group = None
    query_counter = {"CORE": 0, "OBJ": 0, "SRC": 0}
    
    with open(QUERIES_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "CORE_CE_X_AI":
                current_group = "CORE"
            elif line == "HIGH_YIELD_OBJECT":
                current_group = "OBJ"
            elif line == "SOURCE_TARGETED":
                current_group = "SRC"
            elif current_group:
                query_counter[current_group] += 1
                query_id = f"{current_group}_{query_counter[current_group]:02d}"
                queries.append({
                    "query_id": query_id,
                    "query_string": line,
                    "group": current_group
                })
    return queries

def build_google_news_url(query_string):
    """Build Google News search URL from query string"""
    encoded = urllib.parse.quote(query_string)
    return f"https://news.google.com/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"

def generate_doc_id(url, title):
    """Generate deterministic doc_id from url + title"""
    combined = f"{url}{title}"
    return hashlib.sha1(combined.encode('utf-8')).hexdigest()

def create_article_record(query_id, query_string, engine, rank, url, title, source, published_at=None):
    """Create a single article record for url_queue.jsonl"""
    return {
        "query_id": query_id,
        "query_string": query_string,
        "engine_used": engine,
        "rank": rank,
        "url": url,
        "title": title,
        "source": source,
        "published_at": published_at
    }

def append_to_queue(records):
    """Append records to url_queue.jsonl"""
    with open(URL_QUEUE_FILE, 'a') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

def main():
    """Main execution flow"""
    print("=" * 60)
    print("AI × Civil Engineering Article Collection Pipeline")
    print("=" * 60)
    
    # Load queries
    queries = load_queries()
    print(f"\nLoaded {len(queries)} queries:")
    for group in ["CORE", "OBJ", "SRC"]:
        count = len([q for q in queries if q["group"] == group])
        print(f"  - {group}: {count} queries")
    
    # Display query URLs for manual collection
    print("\n" + "=" * 60)
    print("GOOGLE NEWS SEARCH URLS")
    print("Copy these URLs to browser for manual collection:")
    print("=" * 60)
    
    for q in queries:
        url = build_google_news_url(q["query_string"])
        print(f"\n[{q['query_id']}]")
        print(f"Query: {q['query_string']}")
        print(f"URL: {url}")
    
    print("\n" + "=" * 60)
    print("INSTRUCTIONS FOR DATA COLLECTION")
    print("=" * 60)
    print("""
1. Open each URL above in your browser
2. Scroll down to load more results
3. For each article, record:
   - URL
   - Title  
   - Source (domain name)
   - Published date (if visible)
4. Add records to url_queue.jsonl using this format:
   {"query_id":"...", "query_string":"...", "engine_used":"google_news", 
    "rank":1, "url":"...", "title":"...", "source":"...", "published_at":"YYYY-MM-DD"}

Alternative: Use NewsAPI.org for programmatic access
- Sign up at https://newsapi.org
- Use the /v2/everything endpoint with query parameters
- Free tier allows 100 requests/day
""")

if __name__ == "__main__":
    main()
