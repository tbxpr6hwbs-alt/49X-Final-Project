#!/usr/bin/env python3
"""
Strict Relevance Classifier for Academic JSONL Corpus
Classification Agent: intersection of AI AND Civil Engineering

CRITICAL RULE:
Decision must be based PRIMARILY on the "summary" field.
If "summary" is missing -> is_relevant = false.
"""

import json
import re
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict

# Paths
DEFAULT_INPUT = "data/combined_corpus.jsonl"
DEFAULT_OUTPUT = "data/classified_corpus.jsonl"

@dataclass
class ClassificationResult:
    doc_id: str
    is_relevant: bool
    relevance_score: int
    ai_methods_detected: list = field(default_factory=list)
    civil_domains_detected: list = field(default_factory=list)
    primary_reason: str = ""
    exclusion_reason: str = None  # None if relevant

class RelevanceClassifier:
    def __init__(self):
        # AI Keywords (Must appear in summary)
        self.ai_patterns = {
            "Deep Learning": [r"deep learning", r"neural network", r"CNN", r"RNN", r"LSTM", r"transformer", r"autoencoder", r"GAN"],
            "Machine Learning": [r"machine learning", r"supervised learning", r"unsupervised learning", r"reinforcement learning", r"support vector machine", r"SVM", r"random forest", r"decision tree", r"gradient boosting", r"XGBoost"],
            "Computer Vision": [r"computer vision", r"object detection", r"semantic segmentation", r"image processing", r"image classification", r"YOLO"],
            "Optimization": [r"genetic algorithm", r"particle swarm", r"optimization", r"evolutionary"],
            "NLP": [r"natural language processing", r"NLP", r"text mining", r"bert", r"gpt"],
            "General AI": [r"artificial intelligence", r"AI-based", r"data-driven approach", r"predictive model", r"classification model"]
        }

        # Civil Engineering Keywords (Must appear in summary)
        self.ce_patterns = {
            "Structural": [r"structural health", r"SHM", r"damage detection", r"crack detection", r"structural analysis", r"bridge", r"building"],
            "Geotechnical": [r"geotechnical", r"soil", r"rock", r"landslide", r"slope stability", r"tunnel", r"foundation"],
            "Transportation": [r"traffic", r"transportation", r"vehicle", r"pavement", r"road", r"accident prediction"],
            "Construction": [r"construction", r"project management", r"BIM", r"building information modeling", r"safety", r"site monitoring"],
            "Water/Env": [r"hydrology", r"water resource", r"flood", r"environmental", r"river", r"dam"],
            "Materials": [r"concrete", r"asphalt", r"steel", r"material property", r"cement"],
            "Infrastructure": [r"infrastructure", r"smart city", r"urban planning"]
        }
        
        # Exclusion patterns (Immediate disqualification)
        self.exclusion_patterns = [
            r"survey", r"review", r"overview", r"state-of-the-art",  # Reviews
            r"future potential", r"concept only", r"theoretical framework", # Not applied
            r"descriptive analysis", r"statistical analysis" # Classical stats only
        ]

    def check_pattern(self, text, patterns_dict):
        detected = []
        text_lower = text.lower()
        for category, patterns in patterns_dict.items():
            for pattern in patterns:
                if re.search(r'\b' + pattern + r'\b', text_lower):
                    detected.append(category)
                    break 
        return list(set(detected))

    def classify(self, record):
        doc_id = record.get('doc_id', 'unknown')
        summary = record.get('summary', '')

        # Rule 1: Missing Summary -> Irrelevant
        if not summary or len(summary.strip()) < 10:
            return ClassificationResult(
                doc_id=doc_id,
                is_relevant=False,
                relevance_score=0,
                exclusion_reason="Missing or empty summary"
            )

        # Rule 2: Automatic Exclusion
        summary_lower = summary.lower()
        for excl in self.exclusion_patterns:
            if re.search(r'\b' + excl + r'\b', summary_lower):
                # Check if it's a "literature review" or just mentioning current literature
                
                # Careful: "We review..." vs "Literature review"
                if "literature review" in summary_lower or "survey of" in summary_lower or "review of" in summary_lower:
                     return ClassificationResult(
                        doc_id=doc_id,
                        is_relevant=False,
                        relevance_score=10,
                        exclusion_reason=f"Detected exclusion term: {excl}"
                    )

        # Rule 3: AI Requirement
        ai_detected = self.check_pattern(summary, self.ai_patterns)
        if not ai_detected:
            return ClassificationResult(
                doc_id=doc_id,
                is_relevant=False,
                relevance_score=20,
                exclusion_reason="No explicit AI method found in summary"
            )

        # Rule 4: CE Requirement
        ce_detected = self.check_pattern(summary, self.ce_patterns)
        if not ce_detected:
            return ClassificationResult(
                doc_id=doc_id,
                is_relevant=False,
                relevance_score=30,
                ai_methods_detected=ai_detected,
                exclusion_reason="No explicit Civil Engineering domain found in summary"
            )

        # Rule 5: Relevance & Scoring
        # Both present!
        score = 80 # Base score for having both
        
        # Boost score if multiple specific methods/domains found
        if len(ai_detected) > 1: score += 5
        if len(ce_detected) > 1: score += 5
        
        # Check for specific phrase linking valid AI to CE (simple heuristic)
        if "using" in summary_lower or "based on" in summary_lower or "proposed" in summary_lower:
             score += 5

        score = min(score, 100)
        
        # Primary reason construction
        primary_reason = f"Applies {', '.join(ai_detected)} to {', '.join(ce_detected)} problems."

        return ClassificationResult(
            doc_id=doc_id,
            is_relevant=True,
            relevance_score=score,
            ai_methods_detected=ai_detected,
            civil_domains_detected=ce_detected,
            primary_reason=primary_reason,
            exclusion_reason=None
        )

def main():
    parser = argparse.ArgumentParser(description="Strict Relevance Classifier")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSONL file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        return

    classifier = RelevanceClassifier()
    
    classification_counts = {"relevant": 0, "irrelevant": 0}
    
    print(f"Processing {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip(): continue
            try:
                record = json.loads(line)
                result = classifier.classify(record)
                
                # Convert dataclass to dict for JSON serialization
                result_dict = asdict(result)
                
                # Write to output
                fout.write(json.dumps(result_dict) + "\n")
                
                if result.is_relevant:
                    classification_counts["relevant"] += 1
                else:
                    classification_counts["irrelevant"] += 1
                    
            except json.JSONDecodeError:
                continue
    
    print("="*40)
    print(f"Classification Complete.")
    print(f"Relevant Papers: {classification_counts['relevant']}")
    print(f"Irrelevant Papers: {classification_counts['irrelevant']}")
    print(f"Output saved to: {output_path}")
    print("="*40)

if __name__ == "__main__":
    main()
