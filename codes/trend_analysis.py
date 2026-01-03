#!/usr/bin/env python3
"""
Trend Analysis and Visualization for AI in Civil Engineering Corpus
Creates interactive and static visualizations for research trends.
"""

import json
import sqlite3
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
import numpy as np
from datetime import datetime

# Set style
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11

# Paths
DB_PATH = Path("data/corpus.db")
OUTPUT_DIR = Path("analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Color palettes
NEON_COLORS = ['#00ff88', '#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3', 
               '#f38181', '#aa96da', '#fcbad3', '#a8d8ea', '#ff9a3c']
GRADIENT_CMAP = plt.cm.viridis

def load_data():
    """Load data from SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT doc_id, source, title, year, ai_methods, 
               civil_engineering_domain, keywords, data_types
        FROM articles
    """)
    
    articles = []
    for row in cursor.fetchall():
        articles.append({
            'doc_id': row[0],
            'source': row[1],
            'title': row[2],
            'year': row[3],
            'ai_methods': json.loads(row[4]) if row[4] else [],
            'domains': json.loads(row[5]) if row[5] else [],
            'keywords': json.loads(row[6]) if row[6] else [],
            'data_types': json.loads(row[7]) if row[7] else []
        })
    
    conn.close()
    return articles

def create_radial_bar_chart(data, title, filename, top_n=10):
    """Create a beautiful radial bar chart."""
    counter = Counter(data)
    top_items = counter.most_common(top_n)
    
    if not top_items:
        return
    
    labels = [item[0][:25] + '...' if len(item[0]) > 25 else item[0] for item in top_items]
    values = [item[1] for item in top_items]
    
    # Create radial chart
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Normalize values for visual appeal
    max_val = max(values)
    normalized = [v / max_val for v in values]
    
    # Create gradient colors
    colors = [plt.cm.plasma(0.3 + 0.6 * n) for n in normalized]
    
    # Draw bars
    bars = ax.bar(angles, values, width=0.5, bottom=0,
                  color=colors, edgecolor='white', linewidth=1.5, alpha=0.8)
    
    # Add glow effect
    for bar, val in zip(bars, values):
        bar.set_alpha(0.9)
    
    # Customize
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=9, color='white')
    ax.set_title(title, size=18, color='white', pad=30, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['polar'].set_color('#4a4a6a')
    ax.set_ylim(0, max(values) * 1.2)
    
    # Add value annotations
    for angle, val, label in zip(angles, values, labels):
        ax.annotate(str(val), xy=(angle, val + max(values)*0.05),
                   ha='center', va='bottom', color='#00ff88', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, facecolor='#1a1a2e', 
                bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"  ✓ Saved: {filename}")

def create_timeline_heatmap(articles, filename):
    """Create a timeline showing research intensity over years."""
    # Count by year and source
    year_source = {}
    for art in articles:
        year = art['year']
        source = art['source']
        if year:
            if year not in year_source:
                year_source[year] = {}
            year_source[year][source] = year_source[year].get(source, 0) + 1
    
    if not year_source:
        return
    
    years = sorted(year_source.keys())
    sources = list(set(s for ys in year_source.values() for s in ys.keys()))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Stacked area chart
    bottom = np.zeros(len(years))
    for i, source in enumerate(sources):
        values = [year_source.get(y, {}).get(source, 0) for y in years]
        ax.fill_between(range(len(years)), bottom, bottom + values,
                       alpha=0.7, color=NEON_COLORS[i % len(NEON_COLORS)],
                       label=source, linewidth=2)
        ax.plot(range(len(years)), bottom + values, color='white', linewidth=1, alpha=0.5)
        bottom += values
    
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, color='white')
    ax.set_ylabel('Number of Publications', color='white')
    ax.set_xlabel('Year', color='white')
    ax.set_title('📊 Publication Timeline by Source', size=18, color='white', fontweight='bold')
    ax.legend(loc='upper left', facecolor='#2a2a4e', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white')
    
    for spine in ax.spines.values():
        spine.set_color('#4a4a6a')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")

def create_network_visualization(articles, filename):
    """Create a chord-like diagram showing AI methods to CE domains connections."""
    # Build connections
    connections = Counter()
    for art in articles:
        for method in art['ai_methods'][:3]:  # Top 3 methods per article
            for domain in art['domains'][:2]:  # Top 2 domains per article
                connections[(method, domain)] += 1
    
    if not connections:
        return
    
    # Get top connections
    top_connections = connections.most_common(20)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Get unique methods and domains
    methods = list(set(c[0][0] for c in top_connections))
    domains = list(set(c[0][1] for c in top_connections))
    
    # Position methods on left, domains on right
    method_y = {m: i for i, m in enumerate(methods)}
    domain_y = {d: i for i, d in enumerate(domains)}
    
    # Scale y positions
    method_scale = len(domains) / len(methods) if methods else 1
    
    # Draw connections
    max_weight = max(c[1] for c in top_connections)
    for (method, domain), weight in top_connections:
        y1 = method_y[method] * method_scale
        y2 = domain_y[domain]
        alpha = 0.3 + 0.7 * (weight / max_weight)
        lw = 1 + 4 * (weight / max_weight)
        
        # Bezier curve
        x_points = np.array([0, 0.3, 0.7, 1])
        y_points = np.array([y1, y1, y2, y2])
        t = np.linspace(0, 1, 50)
        
        # Cubic bezier
        curve_x = (1-t)**3 * x_points[0] + 3*(1-t)**2*t * x_points[1] + 3*(1-t)*t**2 * x_points[2] + t**3 * x_points[3]
        curve_y = (1-t)**3 * y_points[0] + 3*(1-t)**2*t * y_points[1] + 3*(1-t)*t**2 * y_points[2] + t**3 * y_points[3]
        
        color_idx = methods.index(method) % len(NEON_COLORS)
        ax.plot(curve_x, curve_y, color=NEON_COLORS[color_idx], 
               alpha=alpha, linewidth=lw)
    
    # Draw method nodes (left)
    for method, y in method_y.items():
        y_scaled = y * method_scale
        ax.scatter(-0.05, y_scaled, s=200, c=NEON_COLORS[methods.index(method) % len(NEON_COLORS)],
                  zorder=5, edgecolors='white', linewidth=2)
        ax.text(-0.1, y_scaled, method[:30], ha='right', va='center', 
               color='white', fontsize=9, fontweight='bold')
    
    # Draw domain nodes (right)
    for domain, y in domain_y.items():
        ax.scatter(1.05, y, s=200, c='#ff6b6b', zorder=5, 
                  edgecolors='white', linewidth=2)
        ax.text(1.1, y, domain[:30], ha='left', va='center',
               color='white', fontsize=9, fontweight='bold')
    
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-1, max(len(methods) * method_scale, len(domains)) + 1)
    ax.axis('off')
    ax.set_title('🔗 AI Methods ↔ Civil Engineering Domains Network', 
                size=18, color='white', fontweight='bold', pad=20)
    
    # Add labels
    ax.text(-0.3, -0.5, 'AI Methods', ha='center', color='#00ff88', 
           fontsize=14, fontweight='bold')
    ax.text(1.3, -0.5, 'CE Domains', ha='center', color='#ff6b6b',
           fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")

def create_keyword_wordcloud_alternative(articles, filename):
    """Create a bubble chart as word cloud alternative."""
    all_keywords = []
    for art in articles:
        all_keywords.extend(art['keywords'])
    
    counter = Counter(all_keywords)
    top_keywords = counter.most_common(30)
    
    if not top_keywords:
        return
    
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Create bubble positions
    np.random.seed(42)
    n = len(top_keywords)
    
    # Use force-directed layout simulation
    positions = np.random.rand(n, 2) * 10
    sizes = np.array([kw[1] for kw in top_keywords])
    sizes_normalized = 200 + (sizes / sizes.max()) * 3000
    
    # Simple force simulation
    for _ in range(50):
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                min_dist = (np.sqrt(sizes_normalized[i]) + np.sqrt(sizes_normalized[j])) / 100
                if dist < min_dist and dist > 0:
                    force = diff / dist * (min_dist - dist) * 0.5
                    positions[i] += force
                    positions[j] -= force
    
    # Draw bubbles
    for i, (keyword, count) in enumerate(top_keywords):
        color = NEON_COLORS[i % len(NEON_COLORS)]
        ax.scatter(positions[i, 0], positions[i, 1], 
                  s=sizes_normalized[i], c=color, alpha=0.6,
                  edgecolors='white', linewidth=2)
        ax.annotate(f"{keyword}\n({count})", 
                   xy=(positions[i, 0], positions[i, 1]),
                   ha='center', va='center',
                   fontsize=8 + count/max(sizes)*6,
                   color='white', fontweight='bold')
    
    ax.axis('off')
    ax.set_title('💡 Research Keywords Landscape', size=20, color='white', 
                fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")

def create_data_types_donut(articles, filename):
    """Create a modern donut chart for data types."""
    all_types = []
    for art in articles:
        all_types.extend(art['data_types'])
    
    counter = Counter(all_types)
    if not counter:
        return
    
    labels = list(counter.keys())
    values = list(counter.values())
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Create donut
    colors = NEON_COLORS[:len(labels)]
    wedges, texts, autotexts = ax.pie(values, labels=None, autopct='%1.1f%%',
                                       colors=colors, pctdistance=0.75,
                                       wedgeprops=dict(width=0.5, edgecolor='#1a1a2e', linewidth=3),
                                       textprops=dict(color='white', fontweight='bold'))
    
    # Add center circle
    centre_circle = plt.Circle((0, 0), 0.4, fc='#1a1a2e')
    ax.add_patch(centre_circle)
    
    # Center text
    total = sum(values)
    ax.text(0, 0.05, f'{total}', ha='center', va='center', 
           fontsize=36, color='#00ff88', fontweight='bold')
    ax.text(0, -0.15, 'Total', ha='center', va='center',
           fontsize=14, color='white')
    
    # Legend
    ax.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0.5),
             facecolor='#2a2a4e', edgecolor='white', labelcolor='white',
             fontsize=11)
    
    ax.set_title('📁 Data Types Distribution', size=18, color='white', 
                fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")

def create_source_comparison(articles, filename):
    """Create comparison visualization between sources."""
    source_methods = {}
    for art in articles:
        source = art['source']
        if source not in source_methods:
            source_methods[source] = Counter()
        source_methods[source].update(art['ai_methods'])
    
    if not source_methods:
        return
    
    # Get top methods across all sources
    all_methods = Counter()
    for methods in source_methods.values():
        all_methods.update(methods)
    top_methods = [m[0] for m in all_methods.most_common(8)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    x = np.arange(len(top_methods))
    width = 0.25
    
    for i, (source, methods) in enumerate(source_methods.items()):
        values = [methods.get(m, 0) for m in top_methods]
        bars = ax.bar(x + i * width, values, width, label=source,
                     color=NEON_COLORS[i], alpha=0.8, edgecolor='white', linewidth=1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(val), ha='center', va='bottom', color='white', fontsize=9)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels([m[:20] + '...' if len(m) > 20 else m for m in top_methods], 
                       rotation=45, ha='right', color='white')
    ax.set_ylabel('Frequency', color='white')
    ax.set_title('📈 AI Methods Comparison by Source', size=18, color='white', fontweight='bold')
    ax.legend(facecolor='#2a2a4e', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, axis='y', color='white')
    
    for spine in ax.spines.values():
        spine.set_color('#4a4a6a')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")

def generate_html_report(articles):
    """Generate an interactive HTML report."""
    # Calculate statistics
    sources = Counter(art['source'] for art in articles)
    years = Counter(art['year'] for art in articles if art['year'])
    
    all_methods = []
    all_domains = []
    all_keywords = []
    for art in articles:
        all_methods.extend(art['ai_methods'])
        all_domains.extend(art['domains'])
        all_keywords.extend(art['keywords'])
    
    top_methods = Counter(all_methods).most_common(10)
    top_domains = Counter(all_domains).most_common(10)
    top_keywords = Counter(all_keywords).most_common(15)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI in Civil Engineering - Trend Analysis</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #ffffff;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 2rem;
            background: linear-gradient(90deg, #00ff88, #4ecdc4, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(0,255,136,0.5);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(0,255,136,0.3);
        }}
        .stat-number {{
            font-size: 3rem;
            font-weight: bold;
            color: #00ff88;
            text-shadow: 0 0 20px rgba(0,255,136,0.5);
        }}
        .stat-label {{ color: #aaa; margin-top: 0.5rem; }}
        .section {{
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        h2 {{
            color: #4ecdc4;
            margin-bottom: 1.5rem;
            font-size: 1.5rem;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
        }}
        .image-card {{
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .image-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .tag-cloud {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            justify-content: center;
        }}
        .tag {{
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            transition: all 0.3s;
        }}
        .tag:hover {{
            transform: scale(1.1);
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        }}
        .bar {{
            display: flex;
            align-items: center;
            margin-bottom: 0.75rem;
        }}
        .bar-label {{
            width: 250px;
            font-size: 0.9rem;
            color: #ddd;
        }}
        .bar-fill {{
            height: 25px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            padding-left: 10px;
            font-size: 0.85rem;
            font-weight: bold;
            transition: width 0.5s ease;
        }}
        .footer {{
            text-align: center;
            margin-top: 3rem;
            color: #666;
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 AI in Civil Engineering - Research Trends</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{len(articles)}</div>
                <div class="stat-label">Total Articles</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(sources)}</div>
                <div class="stat-label">Data Sources</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(set(all_methods))}</div>
                <div class="stat-label">AI Methods</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(set(all_domains))}</div>
                <div class="stat-label">CE Domains</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(set(all_keywords))}</div>
                <div class="stat-label">Unique Keywords</div>
            </div>
        </div>

        <div class="section">
            <h2>📊 Top AI Methods</h2>
            {"".join(f'''<div class="bar">
                <span class="bar-label">{m[0][:35]}</span>
                <div class="bar-fill" style="width: {m[1]/top_methods[0][1]*100}%; background: linear-gradient(90deg, #00ff88, #4ecdc4);">{m[1]}</div>
            </div>''' for m in top_methods)}
        </div>

        <div class="section">
            <h2>🏗️ Civil Engineering Domains</h2>
            {"".join(f'''<div class="bar">
                <span class="bar-label">{d[0][:35]}</span>
                <div class="bar-fill" style="width: {d[1]/top_domains[0][1]*100}%; background: linear-gradient(90deg, #ff6b6b, #feca57);">{d[1]}</div>
            </div>''' for d in top_domains)}
        </div>

        <div class="section">
            <h2>🏷️ Trending Keywords</h2>
            <div class="tag-cloud">
                {"".join(f'<span class="tag" style="background: {NEON_COLORS[i % len(NEON_COLORS)]}; font-size: {0.8 + k[1]/top_keywords[0][1]*0.8}rem;">{k[0]} ({k[1]})</span>' for i, k in enumerate(top_keywords))}
            </div>
        </div>

        <div class="section">
            <h2>📈 Visualizations</h2>
            <div class="image-grid">
                <div class="image-card"><img src="ai_methods_radial.png" alt="AI Methods"></div>
                <div class="image-card"><img src="ce_domains_radial.png" alt="CE Domains"></div>
                <div class="image-card"><img src="timeline.png" alt="Timeline"></div>
                <div class="image-card"><img src="network.png" alt="Network"></div>
                <div class="image-card"><img src="keywords_bubble.png" alt="Keywords"></div>
                <div class="image-card"><img src="data_types_donut.png" alt="Data Types"></div>
                <div class="image-card"><img src="source_comparison.png" alt="Source Comparison"></div>
            </div>
        </div>

        <div class="footer">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} | AI + Civil Engineering Trend Analysis
        </div>
    </div>
</body>
</html>"""
    
    with open(OUTPUT_DIR / 'report.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  ✓ Saved: report.html")

def main():
    print("=" * 60)
    print("🔬 AI in Civil Engineering - Trend Analysis")
    print("=" * 60)
    
    print("\n📂 Loading data from database...")
    articles = load_data()
    print(f"   Loaded {len(articles)} articles")
    
    print("\n🎨 Generating visualizations...")
    
    # Extract all data
    all_methods = []
    all_domains = []
    for art in articles:
        all_methods.extend(art['ai_methods'])
        all_domains.extend(art['domains'])
    
    # Generate visualizations
    create_radial_bar_chart(all_methods, "🤖 AI Methods Distribution", "ai_methods_radial.png", 12)
    create_radial_bar_chart(all_domains, "🏗️ Civil Engineering Domains", "ce_domains_radial.png", 10)
    create_timeline_heatmap(articles, "timeline.png")
    create_network_visualization(articles, "network.png")
    create_keyword_wordcloud_alternative(articles, "keywords_bubble.png")
    create_data_types_donut(articles, "data_types_donut.png")
    create_source_comparison(articles, "source_comparison.png")
    
    print("\n📄 Generating HTML report...")
    generate_html_report(articles)
    
    print("\n" + "=" * 60)
    print(f"✅ Analysis complete! Output saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
