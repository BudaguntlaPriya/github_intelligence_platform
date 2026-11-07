"""
Unified Issue Analyzer (Integration for Member 5: FastAPI + Streamlit)
Member 4: Summarization + Priority Prediction

Exposes:
- analyze_issue(title, description, labels="", comments=None) -> dict
  Returns unified dict containing:
    summary, key_error, environment, discussion_summary,
    priority, priority_confidence, priority_reasons,
    category, suggested_label (dynamically imported from Member 2 if available).

- CLI:
  python analyze_issue.py --title "..." --desc "..." [--labels "..."]
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

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

# Import Member 4 components
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from summarization.summarizer import summarize_issue
    from summarization.priority_prediction import predict_priority, load_priority_bundle
except ImportError:
    from summarizer import summarize_issue
    from priority_prediction import predict_priority, load_priority_bundle

# Optional integration with Member 2 (Issue Classification)
_MEMBER2_AVAILABLE = False
_classify_issue_fn = None

try:
    from classification.issue_classification import classify_issue
    _classify_issue_fn = classify_issue
    _MEMBER2_AVAILABLE = True
except ImportError:
    try:
        MEMBER2_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "classification"))
        if os.path.exists(MEMBER2_DIR) and MEMBER2_DIR not in sys.path:
            sys.path.insert(0, MEMBER2_DIR)
        import importlib
        _m2_mod = importlib.import_module("issue_classification")
        _classify_issue_fn = getattr(_m2_mod, "classify_issue", None)
        _MEMBER2_AVAILABLE = _classify_issue_fn is not None
    except Exception:
        _MEMBER2_AVAILABLE = False
        _classify_issue_fn = None
except Exception:
    _MEMBER2_AVAILABLE = False
    _classify_issue_fn = None


def analyze_issue(title, description="", labels="", comments=None, priority_mode="hybrid"):
    """Comprehensive analysis of a GitHub issue for backend and dashboard integration.
    
    Parameters:
    -----------
    title : str
        Issue title.
    description : str, optional
        Issue description or body markdown.
    labels : str, optional
        Pre-existing GitHub labels (comma-separated or string).
    comments : list or str, optional
        Issue comments or discussion thread.
    priority_mode : str, optional
        'hybrid', 'rules', or 'model' (default: 'hybrid').
        
    Returns:
    --------
    dict with unified analysis:
        summary : str
        key_error : str or None
        environment : dict
        discussion_summary : str or None
        priority : str ("Low", "Medium", "High", "Critical")
        priority_confidence : float or None
        priority_reasons : list of str
        priority_method : str
        category : str or None (from Member 2 if available)
        suggested_label : str or None (from Member 2 if available)
        classification_method : str or None
    """
    title = (title or "").strip()
    description = (description or "").strip()
    labels = (labels or "").strip()
    
    # 1. Summarization & metadata extraction
    sum_res = summarize_issue(title, description, comments=comments)
    
    # Count comments
    n_comments = 0
    if isinstance(comments, list):
        n_comments = len(comments)
    elif isinstance(comments, str) and comments.strip():
        n_comments = 1
        
    # 2. Priority prediction
    prio_res = predict_priority(
        title=title,
        description=description,
        labels=labels,
        n_comments=n_comments,
        n_commenters=min(n_comments, 2),
        mode=priority_mode
    )
    
    # 3. Category & label recommendation from Member 2 (if available)
    category = None
    suggested_label = None
    classification_method = None
    
    if _MEMBER2_AVAILABLE and _classify_issue_fn is not None:
        try:
            m2_res = _classify_issue_fn(title, description)
            if m2_res and isinstance(m2_res, dict):
                category = m2_res.get("category")
                suggested_label = m2_res.get("label")
                classification_method = m2_res.get("method")
        except Exception:
            category = None
            suggested_label = None
            classification_method = None
            
    return {
        "title": title,
        "summary": sum_res["summary"],
        "key_error": sum_res["key_error"],
        "environment": sum_res["environment"],
        "discussion_summary": sum_res["discussion_summary"],
        "summary_method": sum_res["method"],
        "compression": sum_res["compression"],
        "priority": prio_res["priority"],
        "priority_agreement": prio_res.get("agreement", prio_res.get("confidence")),
        "agreement": prio_res.get("agreement", prio_res.get("confidence")),
        "priority_reasons": prio_res["reasons"],
        "priority_method": prio_res["method"],
        "priority_score": prio_res["score"],
        "priority_probabilities": prio_res["probabilities"],
        "category": category,
        "suggested_label": suggested_label,
        "classification_method": classification_method,
    }


def main():
    parser = argparse.ArgumentParser(
        description="GitHub Issue Intelligence - Issue Analyzer (Member 4 Integration)"
    )
    parser.add_argument("--title", type=str, required=True, help="Issue title")
    parser.add_argument("--desc", "--description", type=str, default="", help="Issue description")
    parser.add_argument("--labels", type=str, default="", help="Existing GitHub labels")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "rules", "model"], help="Priority mode")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    
    args = parser.parse_args()
    
    result = analyze_issue(
        title=args.title,
        description=args.desc,
        labels=args.labels,
        priority_mode=args.mode
    )
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 70)
        print("GITHUB ISSUE INTELLIGENCE: COMPREHENSIVE ANALYSIS")
        print("=" * 70)
        print(f"Title            : {result['title']}")
        print(f"Priority         : {result['priority']} (Method: {result['priority_method']})")
        if result.get('priority_agreement'):
            print(f"Agreement        : {result['priority_agreement']}")
        print("Priority Reasons :")
        for r in result['priority_reasons']:
            print(f"  - {r}")
        print("-" * 70)
        print(f"Category (M2)    : {result['category'] or 'Not available'} (Suggested Label: {result['suggested_label'] or 'None'})")
        print("-" * 70)
        print(f"Extractive Summary (Method: {result['summary_method']}, Comp: {result['compression']:.1%}):")
        print(f"  {result['summary']}")
        if result['key_error']:
            print(f"Key Error        : {result['key_error']}")
        if result['environment']:
            print(f"Environment      : {result['environment']}")
        if result['discussion_summary']:
            print(f"Discussion       : {result['discussion_summary']}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

