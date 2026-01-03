[README.md](https://github.com/user-attachments/files/24416327/README.md)
# 🤖 Artificial Intelligence Applications in Civil Engineering
## Systematic Literature Review and Trend Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project performs a **large-scale, automated systematic literature review** analyzing **505 academic articles** published between 2015 and 2025 to map the intersection of Artificial Intelligence and Civil Engineering.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Visualizations](#-visualizations)
- [Key Findings](#-key-findings)
- [Authors](#-authors)

---

## 🎯 Overview

The rapid integration of AI into Civil Engineering has created a vast but fragmented body of literature. This project addresses the challenge of synthesizing this knowledge by:

- **Constructing a Unified Corpus**: Aggregating diverse academic sources (arXiv, CORE, Semantic Scholar) into a single, high-quality dataset
- **Automating Classification**: Rule-based taxonomy system to categorize papers by AI methodology and Civil Engineering domain
- **Visualizing Trends**: Interactive dashboard revealing evolution patterns, hot topics, and cross-disciplinary momentum
- **Identifying Research Gaps**: Detecting under-explored intersections between AI capabilities and engineering problems

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📊 **Multi-Source Data Collection** | Ingest from arXiv, CORE, Semantic Scholar APIs |
| 🏷️ **Automated Classification** | Rule-based taxonomy for AI methods & CE domains |
| 📈 **Interactive Dashboard** | 20+ advanced visualizations for trend analysis |
| 🔍 **Relevance Scoring** | 0-100 scoring system to filter quality articles |
| 📁 **SQLite Database** | Structured storage with full-text search capability |

---

## 📊 Dataset

### Sources

| Source | Articles | Percentage |
|--------|----------|------------|
| CORE | 193 | 38.2% |
| arXiv | 65 | 12.9% |
| Semantic Scholar | 58 | 11.5% |
| Nature | 29 | 5.7% |
| Other Sources | 160 | 31.7% |
| **Total** | **505** | **100%** |

### Characteristics

- **Year Range**: 2015 - 2025
- **Total Tokens**: ~4.2 Million
- **Average Relevance Score**: 89.1/100

### AI Methods Detected

- Machine Learning, Deep Learning, Computer Vision
- Natural Language Processing (NLP)
- Optimization Algorithms, Robotics
- IoT/Digital Twin, Generative AI

### Civil Engineering Domains

- Structural Health Monitoring
- Construction Management & BIM
- Transportation & Traffic Analysis
- Geotechnical Engineering
- Water/Environmental Engineering
- Materials Science

---

## 📁 Project Structure

```
49X-Final-Project/
├── data/
│   ├── Project_Report_EN.html    # English project report
│   ├── Proje_Raporu_TR.html      # Turkish project report
│   ├── clean_corpus.jsonl        # Cleaned article corpus
│   └── charts/                   # Visualization images
├── dashboard/                    # Interactive web dashboard
├── trend_project/
│   └── scripts/
│       ├── ingest_jsonl.py       # JSONL ingestion script
│       └── visualize.py          # Visualization generation
├── arxiv_ingest.py               # arXiv API data collection
├── core_ingest.py                # CORE API data collection
├── s2_ingest.py                  # Semantic Scholar ingestion
├── relevance_classifier.py       # Rule-based classifier
├── clean_corpus.py               # Corpus cleaning pipeline
├── merge_corpus.py               # Corpus merging utility
├── taxonomy.yml                  # Classification taxonomy
├── trend.db                      # SQLite database
└── requirements_*.txt            # Python dependencies
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/tbxpr6hwbs-alt/49X-Final-Project.git
cd 49X-Final-Project

# Install dependencies for each module
pip install -r requirements_arxiv.txt
pip install -r requirements_core.txt
pip install -r requirements_s2.txt
```

---

## 💻 Usage

### 1. Data Collection

```bash
# Collect from arXiv
python arxiv_ingest.py

# Collect from CORE
python core_ingest.py

# Collect from Semantic Scholar
python s2_ingest.py
```

### 2. Process & Classify

```bash
# Clean and merge corpus
python clean_corpus.py
python merge_corpus.py

# Classify articles
python relevance_classifier.py
```

### 3. Generate Database

```bash
# Create SQLite database
python jsonl_to_db.py
```

### 4. View Dashboard

Open `dashboard/index.html` in a web browser to explore the interactive visualizations.

---

## 📈 Visualizations

The project includes 20+ advanced visualizations:

1. **Coverage & Bias Analysis**
   - Year Coverage Strip
   - Source Bias Waterfall

2. **Domain & Method Dynamics**
   - Domain/Method Momentum Leaderboards
   - Power Grid Heatmap
   - First Appearance Matrix

3. **Text/Keyword Intelligence**
   - Rising/Declining Keywords
   - Keyword-Domain Heatmap

4. **Data Quality Metrics**
   - Cost vs Value Analysis
   - Relevance Score Distribution

---

## 🔬 Key Findings

### Fastest Growing Methods (2023-2025 vs. prior)

| Method | Growth Rate |
|--------|-------------|
| Robotics | 39x |
| Generative AI | 27x |
| Automation | 23x |
| Optimization | 11x |
| BIM Integration | 7x |

### Strongest Method-Domain Combinations

| Rank | Combination | Articles |
|------|-------------|----------|
| 1 | Deep Learning × Transportation | 199 |
| 2 | Deep Learning × Structural | 164 |
| 3 | Machine Learning × Transportation | 159 |
| 4 | Machine Learning × Structural | 128 |
| 5 | Computer Vision × Transportation | 61 |

### Research Gap Opportunities

- NLP × Geotechnical Engineering
- Optimization × Water/Environmental
- Robotics × Materials Science
- IoT × Structural Health Monitoring

---

## 👥 Authors

- **Ömer Faruk Ersin**
- **Arben Üstün**

**Date**: December 2025

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{ersin2025aice,
  author = {Ersin, Ömer Faruk and Üstün, Arben},
  title = {Artificial Intelligence Applications in Civil Engineering: Systematic Literature Review and Trend Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/tbxpr6hwbs-alt/49X-Final-Project}
}
```
