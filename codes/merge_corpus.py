import json
import os

def normalize_text(text):
    """Normalize text for comparison (lower case, strip whitespace)."""
    if not text:
        return ""
    return text.lower().strip()

def merge_corpus(file1_path, file2_path, output_path):
    seen_urls = set()
    seen_titles = set()
    total_written = 0
    duplicates_skipped = 0
    file1_count = 0
    file2_count = 0

    print(f"Merging {file1_path} and {file2_path} into {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Process file 1 (Priority)
        if os.path.exists(file1_path):
            with open(file1_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        url = data.get('url')
                        title = data.get('title')
                        
                        norm_title = normalize_text(title)
                        
                        # Add to seen sets
                        if url:
                            seen_urls.add(url)
                        if norm_title:
                            seen_titles.add(norm_title)
                            
                        outfile.write(line + '\n')
                        total_written += 1
                        file1_count += 1
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line in {file1_path}")
        else:
            print(f"Warning: {file1_path} not found.")

        # Process file 2
        if os.path.exists(file2_path):
            with open(file2_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        url = data.get('url')
                        title = data.get('title')
                        
                        norm_title = normalize_text(title)
                        
                        is_duplicate = False
                        if url and url in seen_urls:
                            is_duplicate = True
                        if norm_title and norm_title in seen_titles:
                            is_duplicate = True
                            
                        if not is_duplicate:
                            outfile.write(line + '\n')
                            if url:
                                seen_urls.add(url)
                            if norm_title:
                                seen_titles.add(norm_title)
                            total_written += 1
                            file2_count += 1
                        else:
                            duplicates_skipped += 1
                            
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line in {file2_path}")
        else:
             print(f"Warning: {file2_path} not found.")

    print(f"Done.")
    print(f"Records from {file1_path}: {file1_count}")
    print(f"Records from {file2_path} (new): {file2_count}")
    print(f"Duplicates skipped: {duplicates_skipped}")
    print(f"Total records in {output_path}: {total_written}")

if __name__ == "__main__":
    FILE_CLEAN = "data/clean_corpus.jsonl"
    FILE_DETAILED = "data/ce_ai_articles_detailed.jsonl"
    FILE_OUTPUT = "data/merged_corpus.jsonl"
    
    merge_corpus(FILE_CLEAN, FILE_DETAILED, FILE_OUTPUT)
