# GitHub Issue Summarization and Priority Prediction

## Member 4 Module

This module provides two core capabilities for GitHub issue intelligence:
1. **Extractive Issue Summarization & Context Extraction**: Distills long issue descriptions into concise 2–3 sentence summaries (~70 words max), extracts key traceback exceptions (`key_error`), discovers software runtime versions (`environment`), and extracts 2-sentence comment discussions with a recency bias (`discussion_summary`).
2. **Four-Level Priority Prediction (Low, Medium, High, Critical)**: A weak-supervision pipeline combining heuristic rule-based scoring with a machine learning classifier (TF-IDF + structured numeric signals) into a hybrid consensus predictor with hard overrides for critical risks (CVEs, RCE, data loss, startup crashes).

---

## Directory Structure

```text
summarization/
├── __init__.py                 # Package exports
├── summarizer.py               # Extractive TextRank summarizer & metadata extractor
├── priority_prediction.py      # Rule scorer, ML model training, hybrid priority predictor, dev/test sweep, bootstrap CIs
├── evaluate_summaries.py       # Proxy metric evaluation and pure-Python ROUGE calculator
├── evaluate_ratings.py         # Blinded A/B human summary ratings evaluator and unblinder
├── compute_kappa.py            # Cohen's Kappa inter-annotator agreement evaluation script
├── analyze_issue.py            # Unified integration module for Member 5 (CLI + API)
├── data/
│   ├── priority_gold.csv       # 130 hand-curated gold issues (60 random + 30 severity-enriched + 40 metadata-sampled)
│   ├── gold_second_annotator.csv # 40 issues stripped of labels/comments for double-blind teammate review
│   ├── annotator_instructions.md # Detailed priority definitions (Critical, High, Medium, Low) for raters
│   ├── summary_references.csv  # 15 reference issues for ROUGE evaluation
│   ├── summary_ratings.csv     # Blinded A/B summary ratings file (30 differing issues) for 1-5 Likert scoring
│   └── summary_ratings_key.csv # Secret key mapping Summary A and Summary B to TextRank / lead-3
├── models/
│   └── priority_model.pkl      # Trained LogisticRegression priority model bundle
└── results/
    ├── issue_priorities.csv    # Generated priority predictions for all 2,769 dataset issues
    ├── summary_evaluation.txt  # Summary proxy evaluation metrics report
    └── priority_evaluation.txt # Precision, recall, confusion matrix, dev/test threshold sweep, bootstrap CIs
```

---

## Installation & Requirements

Only standard libraries and `numpy`, `scipy`, `pandas`, `scikit-learn` are used. No external NLP model downloads or paid APIs are required.

```bash
pip install numpy scipy pandas scikit-learn
```

---

## How to Run

### 1. Test Single-Issue Summarization
```bash
python summarization/summarizer.py
```

### 2. Train Priority Model & Predict All Issues
```bash
python summarization/priority_prediction.py
```
Outputs:
- Model bundle: `summarization/models/priority_model.pkl`
- Complete predictions: `summarization/results/issue_priorities.csv`

### 3. Evaluate Summarization Quality
```bash
python summarization/evaluate_summaries.py
```
Computes proxy metrics (mean length, compression, title keyword coverage, traceback extraction rate) across all issues with substantive prose (>40 words) and ROUGE metrics on annotated reference summaries.

### 4. Run Unified Issue Analysis (CLI for Member 5)
```bash
python summarization/analyze_issue.py --title "Flask crashes on startup" --desc "PermissionError when starting server"
```
Or with JSON output:
```bash
python summarization/analyze_issue.py --title "Typo in tutorial" --desc "Missing import os in setup" --json
```

---

## Integration Guide for Member 5 (FastAPI & Streamlit)

Member 5 can import `analyze_issue` directly. It automatically combines Member 4's summarization and priority prediction with Member 2's category classification (gracefully degraded if Member 2's model is absent).

### Python API Example:
```python
from summarization import analyze_issue

result = analyze_issue(
    title="Application crashes on startup with TypeError",
    description="When starting Flask on Windows 11, it raises TypeError: cannot import name 'request' from 'flask'. Python 3.11, Flask 3.0.0.",
    labels="bug",
    comments=["Reproduced on Windows 11.", "Fixed in commit 12345."]
)

print(result["priority"])            # "Critical"
print(result["priority_reasons"])    # ['Hard-Critical trigger: Breaks fresh installs...', ...]
print(result["summary"])             # Extractive summary
print(result["key_error"])           # Extracted exception line
print(result["environment"])         # {'python': '3.11', 'flask': '3.0.0'}
print(result["discussion_summary"])  # 2-sentence summary of comments
print(result["category"])            # "Bug" (from Member 2)
print(result["suggested_label"])     # "bug" (from Member 2)
```

