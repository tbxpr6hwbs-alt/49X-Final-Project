import sqlite3
import json
import os

def jsonl_to_db(jsonl_path, db_path):
    print(f"Converting {jsonl_path} to {db_path}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop table if exists to ensure fresh schema
    cursor.execute("DROP TABLE IF EXISTS articles")
    
    # Create table
    cursor.execute("""
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            full_text TEXT,
            summary TEXT,
            ai_methods TEXT,
            civil_engineering_domain TEXT,
            data_types TEXT,
            keywords TEXT,
            summarization TEXT,
            is_relevant BOOLEAN,
            relevance_score INTEGER,
            ai_methods_detected TEXT,
            civil_domains_detected TEXT,
            primary_reason TEXT,
            token_count INTEGER,
            processing_cost REAL,
            doc_id TEXT,
            source TEXT,
            pdf_url TEXT,
            authors TEXT,
            year TEXT,
            categories TEXT
        )
    """)
    conn.commit()
    
    inserted_count = 0
    skipped_count = 0
    
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found.")
        return

    with open(jsonl_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                
                # Extract fields with defaults
                url = data.get('url')
                title = data.get('title')
                full_text = data.get('full_text')
                summary = data.get('summary')
                
                # Serialize list/dict fields
                ai_methods = json.dumps(data.get('ai_methods', []))
                civil_engineering_domain = json.dumps(data.get('civil_engineering_domain', []))
                data_types = json.dumps(data.get('data_types', []))
                keywords = json.dumps(data.get('keywords', []))
                ai_methods_detected = json.dumps(data.get('ai_methods_detected', []))
                civil_domains_detected = json.dumps(data.get('civil_domains_detected', []))
                authors = json.dumps(data.get('authors', []))
                categories = json.dumps(data.get('categories', []))

                summarization = data.get('summarization')
                is_relevant = data.get('is_relevant')
                relevance_score = data.get('relevance_score')
                primary_reason = data.get('primary_reason')
                token_count = data.get('token_count')
                processing_cost = data.get('processing_cost')
                doc_id = data.get('doc_id')
                source = data.get('source')
                pdf_url = data.get('pdf_url')
                year = str(data.get('year')) if data.get('year') is not None else None

                try:
                    cursor.execute("""
                        INSERT INTO articles (
                            url, title, full_text, summary,
                            ai_methods, civil_engineering_domain, data_types, keywords,
                            summarization, is_relevant, relevance_score, 
                            ai_methods_detected, civil_domains_detected, primary_reason,
                            token_count, processing_cost,
                            doc_id, source, pdf_url, authors, year, categories
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        url, title, full_text, summary,
                        ai_methods, civil_engineering_domain, data_types, keywords,
                        summarization, is_relevant, relevance_score,
                        ai_methods_detected, civil_domains_detected, primary_reason,
                        token_count, processing_cost,
                        doc_id, source, pdf_url, authors, year, categories
                    ))
                    inserted_count += 1
                except sqlite3.IntegrityError:
                    skipped_count += 1
                    
            except json.JSONDecodeError:
                print("Skipping invalid JSON line")
                
    conn.commit()
    conn.close()
    
    print(f"Done.")
    print(f"Inserted: {inserted_count}")
    print(f"Skipped (Duplicates): {skipped_count}")

if __name__ == "__main__":
    JSONL_FILE = "data/merged_corpus.jsonl"
    DB_FILE = "data/corpus.db"
    jsonl_to_db(JSONL_FILE, DB_FILE)
