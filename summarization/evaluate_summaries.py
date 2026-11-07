"""
Evaluation of Extractive Summarization
Member 4: Summarization + Priority Prediction

Computes:
1. Proxy metrics across all dataset issues with > 40 words of prose:
   - Mean summary length (words)
   - Mean compression ratio (% reduction)
   - Title keyword coverage (% of title words captured in summary)
   - Traceback key_error extraction rate (% of tracebacks with exception extracted)
2. Standalone ROUGE-1, ROUGE-2, and ROUGE-L metrics computed against
   human reference summaries (when populated in data/summary_references.csv).
"""

import os
import re
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Safe compatibility patch
if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = int

if not hasattr(np, "_core"):
    try:
        import numpy.core as _core
        sys.modules["numpy._core"] = _core
        sys.modules["numpy._core.numeric"] = _core.numeric
        sys.modules["numpy._core.multiarray"] = _core.multiarray
        if hasattr(_core, "_multiarray_umath"):
            sys.modules["numpy._core._multiarray_umath"] = _core._multiarray_umath
    except Exception:
        pass

_orig_np_array = np.array
def _compat_np_array(*args, **kwargs):
    if kwargs.get("copy") is None and "copy" in kwargs:
        kwargs["copy"] = False
    return _orig_np_array(*args, **kwargs)
np.array = _compat_np_array

try:
    from summarization.summarizer import summarize_issue, clean_markdown, extract_key_error
except ImportError:
    from summarizer import summarize_issue, clean_markdown, extract_key_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_FILE = os.path.join(REPO_ROOT, "cleaned_issues.csv")
REF_FILE = os.path.join(BASE_DIR, "data", "summary_references.csv")
RESULT_DIR = os.path.join(BASE_DIR, "results")
REPORT_FILE = os.path.join(RESULT_DIR, "summary_evaluation.txt")

STOPWORDS = {
    "a", "an", "the", "in", "on", "of", "to", "for", "with", "by", "at",
    "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "and", "or", "but", "if",
    "not", "this", "that", "it", "its", "they", "their", "what", "which",
    "when", "where", "why", "how", "all", "any", "both", "each", "few"
}


# ============================================================
# ROUGE METRICS IMPLEMENTATION (PURE PYTHON)
# ============================================================
def tokenize(text):
    """Tokenize text into lowercase words while preserving snake_case."""
    return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_]+\b", text or "")]


def get_ngrams(tokens, n):
    """Generate n-grams list from token sequence."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n(candidate_text, reference_text, n=1):
    """Compute ROUGE-N precision, recall, and F1."""
    cand_tokens = tokenize(candidate_text)
    ref_tokens = tokenize(reference_text)
    
    cand_ngrams = get_ngrams(cand_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)
    
    if not ref_ngrams or not cand_ngrams:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    ref_counts = {}
    for g in ref_ngrams:
        ref_counts[g] = ref_counts.get(g, 0) + 1
        
    cand_counts = {}
    for g in cand_ngrams:
        cand_counts[g] = cand_counts.get(g, 0) + 1
        
    overlap = 0
    for g, count in cand_counts.items():
        if g in ref_counts:
            overlap += min(count, ref_counts[g])
            
    p = overlap / len(cand_ngrams)
    r = overlap / len(ref_ngrams)
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def lcs_length(x, y):
    """Longest common subsequence length via dynamic programming."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]


def rouge_l(candidate_text, reference_text):
    """Compute ROUGE-L precision, recall, and F1 based on LCS."""
    cand_tokens = tokenize(candidate_text)
    ref_tokens = tokenize(reference_text)
    if not cand_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    lcs = lcs_length(cand_tokens, ref_tokens)
    p = lcs / len(cand_tokens)
    r = lcs / len(ref_tokens)
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


# ============================================================
# PROXY METRICS OVER ALL PROSE ISSUES (> 40 words)
# ============================================================
def evaluate_proxy_metrics(df):
    """Compute proxy quality metrics across all issues with substantive prose."""
    lengths = []
    compressions = []
    coverages = []
    
    tb_count = 0
    tb_extracted = 0
    
    substantive_count = 0
    
    for _, row in df.iterrows():
        title = str(row.get("title", "") or "")
        desc = str(row.get("description", "") or "")
        
        # Check traceback extraction (all issues containing 'traceback (most recent call last)')
        if "traceback (most recent call last)" in desc.lower():
            tb_count += 1
            err = extract_key_error(desc)
            if err:
                tb_extracted += 1
                
        cleaned_prose = clean_markdown(desc)
        prose_words = cleaned_prose.split()
        
        # Only evaluate proxy metrics on issues with > 40 words of cleaned prose
        if len(prose_words) > 40:
            substantive_count += 1
            res = summarize_issue(title, desc)
            sum_words = res["summary_words"]
            lengths.append(sum_words)
            compressions.append(res["compression"])
            
            # Title keyword coverage
            title_keywords = [
                w.lower() for w in re.findall(r"\b[a-zA-Z0-9_]+\b", title)
                if w.lower() not in STOPWORDS and len(w) > 2
            ]
            if title_keywords:
                sum_text_lower = res["summary"].lower()
                matches = sum(1 for kw in title_keywords if kw in sum_text_lower)
                coverages.append(matches / len(title_keywords))
                
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    std_len = float(np.std(lengths)) if lengths else 0.0
    mean_comp = float(np.mean(compressions)) if compressions else 0.0
    mean_cov = float(np.mean(coverages)) if coverages else 0.0
    tb_rate = (tb_extracted / max(1, tb_count)) if tb_count > 0 else 0.0
    
    return {
        "substantive_issues_count": substantive_count,
        "mean_summary_length": mean_len,
        "std_summary_length": std_len,
        "mean_compression_ratio": mean_comp,
        "title_keyword_coverage": mean_cov,
        "traceback_total": tb_count,
        "traceback_extracted": tb_extracted,
        "traceback_extraction_rate": tb_rate
    }


# ============================================================
# MAIN EVALUATION RUNNER
# ============================================================
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("EVALUATING ISSUE SUMMARIES")
    print("=" * 70)
    
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded dataset: {len(df):,} total issues.")
    
    print("Computing proxy quality metrics across all issues with > 40 words of prose...")
    proxy = evaluate_proxy_metrics(df)
    
    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)
        
    log("=" * 70)
    log("GITHUB ISSUE SUMMARIZATION EVALUATION REPORT")
    log("=" * 70)
    log(f"Total issues analyzed in dataset    : {len(df):,}")
    log(f"Issues with substantive prose (>40w): {proxy['substantive_issues_count']:,}\n")
    
    log("--- 1. PROXY QUALITY METRICS ---")
    log(f"Mean summary length                 : {proxy['mean_summary_length']:.1f} words (std: {proxy['std_summary_length']:.1f})")
    log(f"Mean compression ratio              : {proxy['mean_compression_ratio']:.1%} reduction")
    log(f"Title-keyword coverage              : {proxy['title_keyword_coverage']:.1%}")
    log(f"Tracebacks encountered in dataset   : {proxy['traceback_total']}")
    log(f"Tracebacks with key_error extracted : {proxy['traceback_extracted']} ({proxy['traceback_extraction_rate']:.1%})\n")
    
    # 2. Check reference summaries
    log("--- 2. ROUGE EVALUATION ON REFERENCE SUMMARIES ---")
    ref_df = None
    if os.path.exists(REF_FILE):
        ref_df = pd.read_csv(REF_FILE)
    else:
        alt_ref = os.path.join(REPO_ROOT, "data", "summary_references.csv")
        if os.path.exists(alt_ref):
            ref_df = pd.read_csv(alt_ref)
            
    if ref_df is not None:
        valid_refs = ref_df[ref_df["reference_summary"].fillna("").str.strip() != ""]
        if len(valid_refs) > 0:
            log(f"Found {len(valid_refs)} populated human reference summaries.")
            r1_scores, r2_scores, rl_scores = [], [], []
            for _, r in valid_refs.iterrows():
                cand = str(r["generated_summary"])
                ref = str(r["reference_summary"])
                r1_scores.append(rouge_n(cand, ref, 1))
                r2_scores.append(rouge_n(cand, ref, 2))
                rl_scores.append(rouge_l(cand, ref))
                
            mean_r1 = {k: np.mean([s[k] for s in r1_scores]) for k in ["precision", "recall", "f1"]}
            mean_r2 = {k: np.mean([s[k] for s in r2_scores]) for k in ["precision", "recall", "f1"]}
            mean_rl = {k: np.mean([s[k] for s in rl_scores]) for k in ["precision", "recall", "f1"]}
            
            log(f"ROUGE-1: Precision: {mean_r1['precision']:.3f} | Recall: {mean_r1['recall']:.3f} | F1: {mean_r1['f1']:.3f}")
            log(f"ROUGE-2: Precision: {mean_r2['precision']:.3f} | Recall: {mean_r2['recall']:.3f} | F1: {mean_r2['f1']:.3f}")
            log(f"ROUGE-L: Precision: {mean_rl['precision']:.3f} | Recall: {mean_rl['recall']:.3f} | F1: {mean_rl['f1']:.3f}")
        else:
            log(f"Reference file '{REF_FILE}' is present with {len(ref_df)} sampled issues,")
            log("but the 'reference_summary' column is currently empty.")
            log("ACTION REQUIRED: Annotate reference summaries by hand to compute ROUGE scores.")
    else:
        log("No summary_references.csv file found.")
        
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        
    print(f"\nReport written to: {REPORT_FILE}")


if __name__ == "__main__":
    main()

