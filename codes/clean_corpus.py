#!/usr/bin/env python3
"""
Clean and Enrich Corpus Script
Filters classified corpus (score >= 60) and adds full text/summary from original corpus.
"""

import json
from pathlib import Path

# Paths
ORIGINAL_CORPUS = Path("data/combined_corpus.jsonl")
CLASSIFIED_CORPUS = Path("data/classified_corpus.jsonl")
OUTPUT_CORPUS = Path("data/clean_corpus.jsonl")

def load_jsonl_map(filepath, key_field):
    data_map = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                record = json.loads(line)
                data_map[record[key_field]] = record
            except json.JSONDecodeError:
                continue
    return data_map

def main():
    print(f"Loading original corpus from {ORIGINAL_CORPUS}...")
    original_data = load_jsonl_map(ORIGINAL_CORPUS, 'doc_id')
    
    print(f"Loading classified corpus from {CLASSIFIED_CORPUS}...")
    
    valid_count = 0
    skipped_count = 0
    
    print(f"Processing and writing to {OUTPUT_CORPUS}...")
    with open(CLASSIFIED_CORPUS, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_CORPUS, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip(): continue
            try:
                classified_record = json.loads(line)
                
                # Filter: relevance_score < 60
                if classified_record.get('relevance_score', 0) < 60:
                    skipped_count += 1
                    continue
                
                doc_id = classified_record.get('doc_id')
                
                # Get fields from original corpus
                original_record = original_data.get(doc_id)
                
                if original_record:
                    # Enrich record
                    # User requested: "add full text, summarization columns"
                    # We will create a clean record merging classification info + original content
                    
                    # Start with original record to keep all metadata (authors, year, title etc)
                    clean_record = original_record.copy()
                    
                    # Ensure specific naming requested
                    clean_record['summarization'] = original_record.get('summary', '') 
                    # (Keep origin 'summary' as well or just replace? I'll keep both to comprise logic)
                    
                    # Add classification metadata
                    clean_record['is_relevant'] = classified_record.get('is_relevant')
                    clean_record['relevance_score'] = classified_record.get('relevance_score')
                    clean_record['ai_methods_detected'] = classified_record.get('ai_methods_detected')
                    clean_record['civil_domains_detected'] = classified_record.get('civil_domains_detected')
                    clean_record['primary_reason'] = classified_record.get('primary_reason')
                    
                    # Add Token Usage and Cost Metadata
                    # Estimation: ~4 characters per token
                    full_text = clean_record.get('full_text', '')
                    token_count = len(full_text) // 4 if full_text else 0
                    
                    clean_record['token_count'] = token_count
                    
                    # Cost: 0.00 since we used a local rule-based classifier, not a paid API
                    clean_record['processing_cost'] = 0.00
                    
                    fout.write(json.dumps(clean_record) + "\n")
                    valid_count += 1
                else:
                    print(f"Warning: Original record not found for {doc_id}")
                    
            except json.JSONDecodeError:
                continue

    print("="*40)
    print("Cleaning Complete.")
    print(f"Kept (Score >= 60): {valid_count}")
    print(f"Removed (Score < 60): {skipped_count}")
    print(f"Output saved to: {OUTPUT_CORPUS}")
    print("="*40)

if __name__ == "__main__":
    main()
