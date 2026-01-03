import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from pathlib import Path
import numpy as np
from collections import Counter

# Configuration
INPUT_FILE = 'data/clean_corpus.jsonl'
OUTPUT_DIR = Path('analysis_output/diagnostics')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

def load_data(filepath):
    """Load JSONL data into a DataFrame."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)

def plot_relevance_distribution(df):
    """1. Relevance Score Distribution"""
    plt.figure(figsize=(10, 6))
    
    # Check if we have variation in is_relevant
    dataset_has_irrelevant = df['is_relevant'].nunique() > 1
    palette = {True: '#2ecc71', False: '#e74c3c'} if dataset_has_irrelevant else None
    
    sns.histplot(
        data=df,
        x='relevance_score',
        hue='is_relevant' if dataset_has_irrelevant else None,
        palette=palette,
        multiple="stack",
        bins=range(0, 105, 5),
        edgecolor='white'
    )
    
    plt.title('Diagnostic 1: Relevance Score Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Relevance Score (0-100)')
    plt.ylabel('Count of Articles')
    plt.axvline(60, color='red', linestyle='--', alpha=0.9, label='Cutoff (60)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_relevance_dist.png', dpi=300)
    plt.close()
    print("Generated: 1_relevance_dist.png")

def plot_heatmap(df):
    """2. AI Methods x Civil Domains Heatmap"""
    # Explode lists
    df_exploded = df.explode('ai_methods_detected').explode('civil_domains_detected')
    
    # Filter out NaNs if any
    df_exploded = df_exploded.dropna(subset=['ai_methods_detected', 'civil_domains_detected'])
    
    # Create crosstab
    ct = pd.crosstab(
        df_exploded['ai_methods_detected'],
        df_exploded['civil_domains_detected']
    )
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(ct, annot=True, fmt='d', cmap='viridis', cbar_kws={'label': 'Article Count'})
    
    plt.title('Diagnostic 2: AI Methods \u00d7 Civil Domains Intersection', fontsize=14, fontweight='bold')
    plt.xlabel('Civil Engineering Domain')
    plt.ylabel('AI Method')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_interaction_heatmap.png', dpi=300)
    plt.close()
    print("Generated: 2_interaction_heatmap.png")

def plot_embeddings(df):
    """3. Text Embedding Projection (TF-IDF + t-SNE)"""
    # Use summarization field
    texts = df['summarization'].fillna('')
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    matrix = tfidf.fit_transform(texts)
    
    # Dimensionality Reduction (t-SNE)
    # Perplexity should be smaller than number of samples
    n_samples = matrix.shape[0]
    perplexity = min(30, max(5, n_samples / 4))
    
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=perplexity)
    embeddings = tsne.fit_transform(matrix.toarray())
    
    df['x_tsne'] = embeddings[:, 0]
    df['y_tsne'] = embeddings[:, 1]
    
    plt.figure(figsize=(12, 8))
    
    sns.scatterplot(
        data=df,
        x='x_tsne',
        y='y_tsne',
        hue='relevance_score', 
        palette='viridis',
        size='relevance_score',
        sizes=(20, 200),
        alpha=0.8
    )
    
    plt.title('Diagnostic 3: Text Embedding Projection (t-SNE of Summaries)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Relevance Score')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_text_embeddings.png', dpi=300)
    plt.close()
    print("Generated: 3_text_embeddings.png")

def plot_leakage_analysis(df):
    """4. Irrelevant-Article Leakage Analysis"""
    irrelevant_df = df[df['is_relevant'] == False]
    
    if irrelevant_df.empty:
        # Create a placeholder if no irrelevant articles found (expected for clean corpus)
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No "False" Relevance Articles in Dataset\n(Clean Corpus contains only Score >= 60)', 
                 ha='center', va='center', fontsize=14)
        plt.title('Diagnostic 4: Irrelevant-Article Leakage Analysis', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.savefig(OUTPUT_DIR / '4_leakage_analysis.png', dpi=300)
        plt.close()
        print("Generated: 4_leakage_analysis.png (No Data)")
        return

    # Process text
    text = ' '.join(irrelevant_df['summarization'].fillna('').astype(str))
    words = [w.lower() for w in text.split() if w.isalpha() and len(w) > 3]
    
    # Filter stopwords (simple list)
    stop_words = set(['this', 'that', 'with', 'from', 'have', 'which', 'were', 'their', 'model', 'data', 'using', 'study', 'results', 'paper', 'proposed', 'method', 'analysis', 'based'])
    words = [w for w in words if w not in stop_words]
    
    common_words = Counter(words).most_common(20)
    
    if not common_words:
        return

    words, counts = zip(*common_words)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(counts), y=list(words), palette='Reds_r')
    
    plt.title('Diagnostic 4: Top Terms in REJECTED Articles', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency')
    plt.ylabel('Term')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_leakage_analysis.png', dpi=300)
    plt.close()
    print("Generated: 4_leakage_analysis.png")

def create_borderline_table(df):
    """5. Borderline Audit Table (Visual)"""
    # Filter scores 50-70
    mask = (df['relevance_score'] >= 50) & (df['relevance_score'] <= 70)
    borderline = df[mask][['doc_id', 'relevance_score', 'ai_methods_detected', 'civil_domains_detected', 'primary_reason']].head(15) # Limit to 15 for visibility
    
    if borderline.empty:
         # Create placeholder
        plt.figure(figsize=(10, 2))
        plt.text(0.5, 0.5, 'No Borderline Articles (50-70) Found', ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.savefig(OUTPUT_DIR / '5_borderline_table.png', dpi=300)
        plt.close()
        print("Generated: 5_borderline_table.png (No Data)")
        return
        
    # Simplify lists for display
    borderline['ai_methods_detected'] = borderline['ai_methods_detected'].apply(lambda x: ', '.join(x[:2]) + '...' if len(x)>2 else ', '.join(x))
    borderline['civil_domains_detected'] = borderline['civil_domains_detected'].apply(lambda x: ', '.join(x[:2]) + '...' if len(x)>2 else ', '.join(x))
    borderline['primary_reason'] = borderline['primary_reason'].apply(lambda x: (x[:40] + '...') if len(str(x)) > 40 else str(x))

    # Plot table
    fig, ax = plt.subplots(figsize=(14, len(borderline)*0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=borderline.values, colLabels=borderline.columns, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title('Diagnostic 5: Borderline Audit (Score 50-70)', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / '5_borderline_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: 5_borderline_table.png")

def plot_cost_value(df):
    """6. Cost vs Value Check"""
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(
        data=df,
        x='token_count',
        y='relevance_score',
        alpha=0.6,
        s=80,
        color='purple'
    )
    
    plt.title('Diagnostic 6: Cost vs Value (Token Count vs Relevance)', fontsize=14, fontweight='bold')
    plt.xlabel('Token Count (Estimated)')
    plt.ylabel('Relevance Score')
    plt.axvline(df['token_count'].mean(), color='orange', linestyle='--', label='Avg Tokens')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '6_cost_value.png', dpi=300)
    plt.close()
    print("Generated: 6_cost_value.png")

def main():
    print(f"Loading data from {INPUT_FILE}...")
    df = load_data(INPUT_FILE)
    print(f"Loaded {len(df)} records.")
    
    print("Generating visualizations...")
    plot_relevance_distribution(df)
    plot_heatmap(df)
    plot_embeddings(df)
    plot_leakage_analysis(df)
    create_borderline_table(df)
    plot_cost_value(df)
    
    print(f"All diagnostics saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
