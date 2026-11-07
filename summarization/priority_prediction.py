"""
GitHub Issue Priority Prediction (Weak Supervision + ML + Hybrid)
Member 4: Summarization + Priority Prediction

Priority Levels: Low, Medium, High, Critical

Architecture:
1. Rule Scorer:
   - Interpretable heuristic rules awarding points for severity signals
     (regression, breaking changes, security, crashes, startup/import failure,
      tracebacks, blocker labels, high comment/commenter counts, production impact)
   - Negative points for docs/typos, questions, feature requests, good first issue,
     cosmetic changes, spam.
   - Hard-Critical overrides (CVE, RCE, auth bypass/injection, data loss,
     crashes on startup / breaks fresh installs).
   - Generates human-readable explanation strings ('reasons') for the dashboard.
2. Machine Learning Model:
   - Feature fusion: TF-IDF (1-2 grams) on text + scaled numeric signals
     (log comments, distinct commenters, log text length, has_traceback,
      code blocks count, label indicator flags).
   - Multi-class Logistic Regression with balanced class weights.
   - Trained strictly on weakly-labeled data excluding manual gold set issues.
3. Hybrid Predictor:
   - Evaluates rule level and model expected level, blending them into a consensus
     prediction with hard overrides for Critical signals.
   - predict_priority(title, description, labels="", n_comments=0, n_commenters=0, mode="hybrid")
"""

import os
import re
import math
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Compatibility patch for numpy/scipy versions
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

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

# ============================================================
# CONFIGURATION & PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_FILE = os.path.join(REPO_ROOT, "cleaned_issues.csv")
COMMENTS_FILE = os.path.join(REPO_ROOT, "raw_comments.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_FILE = os.path.join(MODEL_DIR, "priority_model.pkl")
GOLD_FILE = os.path.join(DATA_DIR, "priority_gold.csv")
RESULT_PRIORITIES_FILE = os.path.join(RESULT_DIR, "issue_priorities.csv")
EVAL_REPORT_FILE = os.path.join(RESULT_DIR, "priority_evaluation.txt")

PRIORITY_LEVELS = ["Low", "Medium", "High", "Critical"]
LEVEL_TO_NUM = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
NUM_TO_LEVEL = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}

RANDOM_STATE = 42

# ============================================================
# RULE DEFINITIONS
# ============================================================
HARD_CRITICAL_TRIGGERS = [
    (r"\b(remote code execution|rce)\b", "Remote code execution (RCE) vulnerability detected"),
    (r"\bcve[- ]\d{4}[- ]\d+\b", "Known CVE security vulnerability identifier present"),
    (r"\b(authentication bypass|auth bypass)\b", "Authentication bypass vulnerability detected"),
    (r"\b(sql injection|command injection|code injection)\b", "Injection vulnerability detected"),
    (r"\b(data loss|corrupt(?:s|ed|ion)?\s+data|silent data corruption|loss of data)\b", "Data loss or silent corruption risk"),
    (r"\b(breaks? fresh installs?|crashes? on startup|crashes? immediately on launch)\b", "Breaks fresh installs or crashes immediately on startup"),
]

SEVERITY_RULES = [
    (r"\bregression\b", 3, "Regression identified (+3)"),
    (r"\b(breaking change|backward[s]? incompatible|breaks backward)\b", 3, "Breaking change / backward incompatibility (+3)"),
    (r"\b(security|vulnerab\w*|xss|csrf|ssrf|exploit\w*|denial of service|open redirect)\b", 3, "Security vulnerability signal (+3)"),
    (r"\b(crash\w*|segfault|panic|fatal|unhandled exception)\b", 3, "Crash or fatal error signal (+3)"),
    (r"\b(fails?|unable|cannot|can't)\s+to\s+(import|install|start|boot|load)\b", 3, "Fails to import / install / start (+3)"),
    (r"\b(production|prod|outage|downtime|live system|service down)\b", 3, "Production outage or live system impact (+3)"),
]

NEGATIVE_RULES = [
    (r"\b(docs?|documentation|typos?|readme|tutorial|changelog|docstrings?)\b", -3, "Documentation or typo (-3)"),
    (r"\b(how to|how do|question|help wanted|wondering if|is it possible)\b", -3, "User question or inquiry (-3)"),
    (r"\b(feature request|enhancement|would be nice|proposal)\b", -2, "Feature request or enhancement proposal (-2)"),
    (r"\b(minor|cosmetic|cleanup|refactor|lint|nit)\b", -2, "Minor / cosmetic / cleanup (-2)"),
    (r"<spam>|\bspam\b", -5, "Spam detected (-5)"),
]


# ============================================================
# RULE SCORER
# ============================================================
def rule_based_priority(title, description="", labels="", n_comments=0, n_commenters=0):
    """Compute heuristic rule-based priority score and explanation reasons.
    
    Returns:
    --------
    level : str ("Low", "Medium", "High", "Critical")
    score : int
    reasons : list of str
    hard_critical : bool
    """
    title = str(title) if pd.notna(title) else ""
    desc = str(description) if pd.notna(description) else ""
    labels = str(labels).strip().lower() if pd.notna(labels) else ""
    full_text = f"{title}\n{desc}".lower()
    title_lower = title.lower()
    
    score = 0
    reasons = []
    hard_critical = False
    
    # 1. Hard-Critical Triggers
    for pat, r_text in HARD_CRITICAL_TRIGGERS:
        if re.search(pat, full_text):
            hard_critical = True
            reasons.append(f"Hard-Critical trigger: {r_text}")
            
    # 2. GitHub Labels
    if "blocker" in labels or "critical" in labels:
        score += 4
        reasons.append("Labeled as 'blocker' or 'critical' (+4)")
    if "docs" in labels or "documentation" in labels:
        score -= 3
        reasons.append("Labeled as 'documentation' (-3)")
    if "good first issue" in labels or "easy" in labels:
        score -= 2
        reasons.append("Labeled as 'good first issue' (-2)")
    if "enhancement" in labels or "feature" in labels:
        score -= 2
        reasons.append("Labeled as 'enhancement' (-2)")
        
    # 3. Title Signals
    if re.search(r"\b(error|exception|crash\w*|fail\w*|traceback|broken|cannot|can't)\b", title_lower):
        score += 2
        reasons.append("Error or failure indicator in title (+2)")
        
    # 4. Traceback in description
    if "traceback (most recent call last):" in desc.lower():
        score += 2
        reasons.append("Contains Python traceback (+2)")
        
    # 5. Positive Severity Rules
    for pat, pts, r_text in SEVERITY_RULES:
        if re.search(pat, full_text):
            score += pts
            reasons.append(r_text)
            
    # 6. Negative Rules
    for pat, pts, r_text in NEGATIVE_RULES:
        if re.search(pat, title_lower):
            score += pts
            reasons.append(f"Title: {r_text}")
        elif re.search(pat, full_text) and score < 4:
            score += pts
            reasons.append(r_text)
            
    # 7. Comment and Community Activity Signals
    try:
        n_c = int(n_comments) if pd.notna(n_comments) else 0
    except:
        n_c = 0
    try:
        n_u = int(n_commenters) if pd.notna(n_commenters) else 0
    except:
        n_u = 0
        
    if n_c >= 8 or n_u >= 5:
        score += 2
        reasons.append(f"High community activity ({n_c} comments, {n_u} commenters) (+2)")
    elif n_c >= 4 or n_u >= 3:
        score += 1
        reasons.append(f"Moderate community activity ({n_c} comments) (+1)")
        
    # Threshold Mapping
    if hard_critical:
        level = "Critical"
    elif score >= 7:
        level = "Critical"
    elif score >= 4:
        level = "High"
    elif score >= 1:
        level = "Medium"
    else:
        level = "Low"
        
    if not reasons:
        reasons.append("Default baseline score; no strong positive or negative triggers")
        
    return level, score, reasons, hard_critical


# ============================================================
# FEATURE EXTRACTION FOR MACHINE LEARNING
# ============================================================
def extract_numeric_features(df_in, comm_user_map=None):
    """Extract structured numeric features from an issues DataFrame or list of dicts."""
    if comm_user_map is None:
        comm_user_map = {}
        
    num_feats = []
    texts = []
    
    for _, row in df_in.iterrows():
        title = str(row.get("title", "") or "")
        desc = str(row.get("description", "") or "")
        labels = str(row.get("labels", "") or "").lower()
        
        # Weighted text representation
        text = f"{title} {title} {desc}"
        texts.append(text)
        
        # Comments count
        n_comm = row.get("comments", 0)
        try:
            n_comm = float(n_comm) if pd.notna(n_comm) else 0.0
        except:
            n_comm = 0.0
        log_comm = math.log1p(max(0.0, n_comm))
        
        # Distinct commenters
        issue_id = row.get("issue_id")
        n_users = comm_user_map.get(issue_id, 1.0 if n_comm > 0 else 0.0)
        log_users = math.log1p(float(n_users))
        
        # Text length
        text_len = math.log1p(len(desc))
        
        # Has traceback
        has_tb = 1.0 if "traceback (most recent call last):" in desc.lower() else 0.0
        
        # Code blocks count
        code_blocks = float(len(re.findall(r"```", desc))) / 2.0
        
        # Label indicator flags
        has_bug = 1.0 if "bug" in labels else 0.0
        has_docs = 1.0 if ("docs" in labels or "documentation" in labels) else 0.0
        has_gfi = 1.0 if ("good first issue" in labels or "easy" in labels) else 0.0
        has_feat = 1.0 if ("enhancement" in labels or "feature" in labels) else 0.0
        has_blocker = 1.0 if ("blocker" in labels or "critical" in labels) else 0.0
        
        num_feats.append([
            log_comm, log_users, text_len, has_tb, code_blocks,
            has_bug, has_docs, has_gfi, has_feat, has_blocker
        ])
        
    return texts, np.array(num_feats, dtype=np.float32)


# ============================================================
# ML MODEL TRAINING & SERIALIZATION
# ============================================================
def train_priority_model(exclude_gold_file=GOLD_FILE, save_path=MODEL_FILE):
    """Train the TF-IDF + LogisticRegression model on rule-based weak labels,
    strictly excluding gold-set issues from training.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print("\n" + "=" * 70)
    print("TRAINING PRIORITY PREDICTION MODEL (WEAK SUPERVISION)")
    print("=" * 70)
    
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
        
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded dataset: {len(df):,} total issues.")
    
    # Load comments mapping for distinct commenters
    comm_user_map = {}
    if os.path.exists(COMMENTS_FILE):
        cdf = pd.read_csv(COMMENTS_FILE)
        comm_user_map = cdf.groupby("issue_id")["username"].nunique().to_dict()
        print(f"Loaded comments data: {len(cdf):,} comments across {len(comm_user_map):,} issues.")
        
    # Generate weak priority labels for all issues
    weak_levels = []
    weak_scores = []
    for _, row in df.iterrows():
        issue_id = row.get("issue_id")
        n_comm = row.get("comments", 0)
        n_users = comm_user_map.get(issue_id, 1 if n_comm > 0 else 0)
        lvl, sc, _, _ = rule_based_priority(
            row.get("title", ""),
            row.get("description", ""),
            row.get("labels", ""),
            n_comm,
            n_users
        )
        weak_levels.append(lvl)
        weak_scores.append(sc)
        
    df["weak_priority"] = weak_levels
    df["weak_score"] = weak_scores
    
    # Exclude gold set issues from training
    gold_ids = set()
    if os.path.exists(exclude_gold_file):
        gold_df = pd.read_csv(exclude_gold_file)
        gold_ids = set(gold_df["issue_id"].dropna().astype(int))
        print(f"Excluding {len(gold_ids)} gold-set issues from training.")
        
    train_pool = df[~df["issue_id"].isin(gold_ids)].copy().reset_index(drop=True)
    print(f"Training pool size: {len(train_pool):,} issues.")
    print("Weak Label Distribution in training pool:")
    print(train_pool["weak_priority"].value_counts().to_string())
    
    # Feature extraction
    texts, num_feats = extract_numeric_features(train_pool, comm_user_map)
    labels = train_pool["weak_priority"].tolist()
    
    # Stratified train/test split on the weak labels
    X_train_txt, X_val_txt, num_train, num_val, y_train, y_val = train_test_split(
        texts, num_feats, labels, test_size=0.20, random_state=RANDOM_STATE, stratify=labels
    )
    
    # 1. TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        sublinear_tf=True
    )
    tfidf_train = vectorizer.fit_transform(X_train_txt)
    tfidf_val = vectorizer.transform(X_val_txt)
    
    # 2. Scaler for numeric features
    scaler = StandardScaler()
    num_train_scaled = scaler.fit_transform(num_train)
    num_val_scaled = scaler.transform(num_val)
    
    # 3. Combined feature matrices
    X_train_comb = hstack([tfidf_train, csr_matrix(num_train_scaled)]).tocsr()
    X_val_comb = hstack([tfidf_val, csr_matrix(num_val_scaled)]).tocsr()
    
    # 4. Multi-class Logistic Regression with balanced weights
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=2.0,
        random_state=RANDOM_STATE
    )
    model.fit(X_train_comb, y_train)
    
    # Validation evaluation (AGREEMENT with weak labels, NOT true accuracy)
    y_pred = model.predict(X_val_comb)
    val_agreement = accuracy_score(y_val, y_pred)
    val_macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    report = classification_report(y_val, y_pred, zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=PRIORITY_LEVELS)
    
    print("\n" + "=" * 70)
    print("ML MODEL VALIDATION (AGREEMENT WITH WEAK LABELS ON HELD-OUT SPLIT)")
    print("=" * 70)
    print(f"Agreement Rate: {val_agreement:.2%}")
    print(f"Macro-F1 (Weak Agreement): {val_macro_f1:.4f}")
    print("NOTE: Evaluated against rule-generated weak labels. Reflects rule-learning fidelity, not human accuracy.\n")
    print(report)
    print("Confusion Matrix against weak labels (rows=weak label, cols=predicted):")
    print(PRIORITY_LEVELS)
    print(cm)
    
    bundle = {
        "vectorizer": vectorizer,
        "scaler": scaler,
        "model": model,
        "classes": list(model.classes_),
        "val_agreement": float(val_agreement),
        "val_macro_f1": float(val_macro_f1),
        "comm_user_map": comm_user_map
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nModel saved successfully to: {save_path}")
    
    return bundle


def load_priority_bundle(path=MODEL_FILE):
    """Load cached model bundle, training on-the-fly if not found."""
    if not os.path.exists(path):
        return train_priority_model(save_path=path)
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# PUBLIC API: PREDICT PRIORITY (HYBRID)
# ============================================================
def predict_priority(title, description="", labels="", n_comments=0, n_commenters=0, mode="hybrid", bundle=None):
    """Predict issue priority using rules, ML model, or hybrid consensus.
    
    Parameters:
    -----------
    title : str
    description : str
    labels : str
    n_comments : int
    n_commenters : int
    mode : str ("hybrid", "rules", "model")
    bundle : dict, optional
        Pre-loaded model bundle.
        
    Returns:
    --------
    dict with keys:
        priority : str ("Low", "Medium", "High", "Critical")
        confidence : float or None
            Prediction confidence (None for rule overrides or pure rule mode).
        method : str
            Method name explaining how prediction was derived.
        score : float
            Numeric score or composite level.
        reasons : list of str
            Human-readable explanation of triggers.
        probabilities : dict
            Class probability distribution.
    """
    title = str(title) if pd.notna(title) else ""
    desc = str(description) if pd.notna(description) else ""
    labels = str(labels).strip() if pd.notna(labels) else ""
    
    # 1. Rule evaluation
    rule_level, rule_score, reasons, hard_critical = rule_based_priority(
        title, desc, labels, n_comments, n_commenters
    )
    
    # 2. ML Model evaluation
    if bundle is None:
        bundle = load_priority_bundle()
        
    vectorizer = bundle["vectorizer"]
    scaler = bundle["scaler"]
    model = bundle["model"]
    classes = list(model.classes_)
    
    # Feature extraction for single instance
    text = f"{title} {title} {desc}"
    tfidf_feat = vectorizer.transform([text])
    
    log_c = math.log1p(max(0.0, float(n_comments)))
    log_u = math.log1p(max(0.0, float(n_commenters)))
    text_len = math.log1p(len(desc))
    has_tb = 1.0 if "traceback (most recent call last):" in desc.lower() else 0.0
    code_blocks = float(len(re.findall(r"```", desc))) / 2.0
    labels_l = labels.lower()
    has_bug = 1.0 if "bug" in labels_l else 0.0
    has_docs = 1.0 if ("docs" in labels_l or "documentation" in labels_l) else 0.0
    has_gfi = 1.0 if ("good first issue" in labels_l or "easy" in labels_l) else 0.0
    has_feat = 1.0 if ("enhancement" in labels_l or "feature" in labels_l) else 0.0
    has_blocker = 1.0 if ("blocker" in labels_l or "critical" in labels_l) else 0.0
    
    num_vec = np.array([[
        log_c, log_u, text_len, has_tb, code_blocks,
        has_bug, has_docs, has_gfi, has_feat, has_blocker
    ]], dtype=np.float32)
    num_scaled = scaler.transform(num_vec)
    
    X_inst = hstack([tfidf_feat, csr_matrix(num_scaled)]).tocsr()
    
    probs = model.predict_proba(X_inst)[0]
    prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}
    
    # Model expected numeric level
    model_expected_val = sum(LEVEL_TO_NUM.get(c, 0) * p for c, p in zip(classes, probs))
    best_ml_idx = int(np.argmax(probs))
    best_ml_class = str(classes[best_ml_idx])
    
    # Agreement string between rules and model
    if rule_level == best_ml_class:
        agreement = "rules and model agree"
    else:
        agreement = f"rules and model disagree (rule: {rule_level}, model: {best_ml_class})"
    
    # Hard-Critical rule override always takes precedence in hybrid and rules mode
    if hard_critical:
        hc_agreement = "rules and model agree" if best_ml_class == "Critical" else f"rules and model disagree (rule: Critical override, model: {best_ml_class})"
        return {
            "priority": "Critical",
            "agreement": hc_agreement,
            "confidence": hc_agreement,
            "method": "Rule override (Hard-Critical)",
            "score": float(rule_score),
            "reasons": reasons,
            "probabilities": {"Low": 0.0, "Medium": 0.0, "High": 0.0, "Critical": 1.0}
        }
        
    if mode == "rules":
        return {
            "priority": rule_level,
            "agreement": f"Rule-based evaluation (rule: {rule_level})",
            "confidence": f"Rule-based evaluation (rule: {rule_level})",
            "method": "Rule-based",
            "score": float(rule_score),
            "reasons": reasons,
            "probabilities": {lvl: (1.0 if lvl == rule_level else 0.0) for lvl in PRIORITY_LEVELS}
        }
    
    if mode == "model":
        return {
            "priority": best_ml_class,
            "agreement": agreement,
            "confidence": agreement,
            "method": "ML (Logistic Regression)",
            "score": round(model_expected_val, 2),
            "reasons": [f"ML highest probability class: {best_ml_class}"],
            "probabilities": {lvl: round(prob_dict.get(lvl, 0.0), 3) for lvl in PRIORITY_LEVELS}
        }
        
    # 3. Hybrid Combination: average of rule level and model expected level
    rule_num = LEVEL_TO_NUM[rule_level]
    hybrid_val = (rule_num + model_expected_val) / 2.0
    
    # Continuous threshold mapping with rounding
    # < 0.5: Low, 0.5 - 1.5: Medium, 1.5 - 2.5: High, >= 2.5: Critical
    if hybrid_val >= 2.5:
        hybrid_level = "Critical"
    elif hybrid_val >= 1.5:
        hybrid_level = "High"
    elif hybrid_val >= 0.5:
        hybrid_level = "Medium"
    else:
        hybrid_level = "Low"
        
    reasons.append(
        f"Hybrid consensus: rule level '{rule_level}' ({rule_num}) + model expected ({model_expected_val:.2f}) -> {hybrid_level}"
    )
    
    return {
        "priority": hybrid_level,
        "agreement": agreement,
        "confidence": agreement,
        "method": "Hybrid (Rules + Model)",
        "score": round(hybrid_val, 2),
        "reasons": reasons,
        "probabilities": {lvl: round(prob_dict.get(lvl, 0.0), 3) for lvl in PRIORITY_LEVELS}
    }


# ============================================================
# BATCH PREDICTION FOR ALL DATASET ISSUES
# ============================================================
def batch_predict_all_issues(output_file=RESULT_PRIORITIES_FILE):
    """Predict priorities for all 2,769 issues and save to CSV."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    
    comm_user_map = {}
    if os.path.exists(COMMENTS_FILE):
        cdf = pd.read_csv(COMMENTS_FILE)
        comm_user_map = cdf.groupby("issue_id")["username"].nunique().to_dict()
        
    bundle = load_priority_bundle()
    
    records = []
    print(f"Generating priorities for all {len(df):,} issues...")
    for _, row in df.iterrows():
        issue_id = row.get("issue_id")
        n_c = row.get("comments", 0)
        n_u = comm_user_map.get(issue_id, 1 if n_c > 0 else 0)
        
        pred = predict_priority(
            row.get("title", ""),
            row.get("description", ""),
            row.get("labels", ""),
            n_c,
            n_u,
            mode="hybrid",
            bundle=bundle
        )
        
        records.append({
            "issue_id": issue_id,
            "issue_number": row.get("issue_number"),
            "title": row.get("title"),
            "predicted_priority": pred["priority"],
            "confidence": pred["confidence"],
            "method": pred["method"],
            "score": pred["score"],
            "reasons": "; ".join(pred["reasons"]),
            "prob_low": pred["probabilities"].get("Low", 0.0),
            "prob_medium": pred["probabilities"].get("Medium", 0.0),
            "prob_high": pred["probabilities"].get("High", 0.0),
            "prob_critical": pred["probabilities"].get("Critical", 0.0)
        })
        
    out_df = pd.DataFrame(records)
    out_df.to_csv(output_file, index=False)
    print(f"Priorities successfully saved to: {output_file}")
    print("\nPredicted Priority Distribution across all issues:")
    print(out_df["predicted_priority"].value_counts().to_string())
    return out_df


# ============================================================
# EVALUATION AGAINST GOLD SET (STANDALONE RUNNER)
# ============================================================
def evaluate_gold_set(gold_file=GOLD_FILE, output_file=EVAL_REPORT_FILE):
    """Evaluate Rules, ML Model, and Hybrid system against hand-labeled Gold Set.
    
    Computes:
    - Accuracy
    - Precision, Recall, F1 per class
    - Macro-F1
    - Within-one-level accuracy
    - Mean Absolute Error (MAE) in levels
    - Confusion Matrix
    Evaluates both on the full gold set (130 issues) and the random subset (60 issues).
    """
    if not os.path.exists(gold_file):
        raise FileNotFoundError(f"Gold set file not found: {gold_file}")
        
    gold_df = pd.read_csv(gold_file)
    print(f"\nLoaded Gold Set: {len(gold_df)} examples.")
    
    # Load issue text and comments to match gold issues
    df = pd.read_csv(DATA_FILE)
    issue_map = df.set_index("issue_id").to_dict("index")
    
    comm_user_map = {}
    if os.path.exists(COMMENTS_FILE):
        cdf = pd.read_csv(COMMENTS_FILE)
        comm_user_map = cdf.groupby("issue_id")["username"].nunique().to_dict()
        
    bundle = load_priority_bundle()
    
    # Run predictions under each mode
    modes = ["rules", "model", "hybrid"]
    results = {m: [] for m in modes}
    
    for _, grow in gold_df.iterrows():
        iid = grow["issue_id"]
        info = issue_map.get(iid, {})
        title = info.get("title", grow.get("title", ""))
        desc = info.get("description", "")
        labels = info.get("labels", "")
        n_c = info.get("comments", 0)
        n_u = comm_user_map.get(iid, 1 if n_c > 0 else 0)
        
        for m in modes:
            pred = predict_priority(title, desc, labels, n_c, n_u, mode=m, bundle=bundle)
            results[m].append(pred["priority"])
            
    for m in modes:
        gold_df[f"pred_{m}"] = results[m]
        
    # Evaluation metrics generator
    def compute_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        prec = precision_score(y_true, y_pred, labels=PRIORITY_LEVELS, average=None, zero_division=0)
        rec = recall_score(y_true, y_pred, labels=PRIORITY_LEVELS, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=PRIORITY_LEVELS, average=None, zero_division=0)
        
        # Numeric level differences
        y_true_num = np.array([LEVEL_TO_NUM[y] for y in y_true])
        y_pred_num = np.array([LEVEL_TO_NUM[y] for y in y_pred])
        diff = np.abs(y_true_num - y_pred_num)
        
        within_one = np.mean(diff <= 1)
        mae = np.mean(diff)
        cm = confusion_matrix(y_true, y_pred, labels=PRIORITY_LEVELS)
        
        per_class = {}
        for idx, lvl in enumerate(PRIORITY_LEVELS):
            support = int(np.sum(np.array(y_true) == lvl))
            per_class[lvl] = {
                "precision": float(prec[idx]),
                "recall": float(rec[idx]),
                "f1": float(f1[idx]),
                "support": support
            }
            
        # Recompute Macro-F1 over classes with n >= 5 only (dropping Critical n=0)
        evaluable_classes = [lvl for lvl in ["Low", "Medium", "High"] if per_class[lvl]["support"] >= 5]
        if evaluable_classes:
            macro_f1 = float(np.mean([per_class[lvl]["f1"] for lvl in evaluable_classes]))
        else:
            macro_f1 = 0.0
            
        # Also compute 3-class Macro-F1 across Low, Medium, High (dropping Critical n=0)
        macro_f1_3class = float(np.mean([per_class["Low"]["f1"], per_class["Medium"]["f1"], per_class["High"]["f1"]]))
            
        # Merged High+Critical binary detection metrics
        y_true_hc = np.array([1 if y in ("High", "Critical") else 0 for y in y_true])
        y_pred_hc = np.array([1 if y in ("High", "Critical") else 0 for y in y_pred])
        tp = int(np.sum((y_true_hc == 1) & (y_pred_hc == 1)))
        fp = int(np.sum((y_true_hc == 0) & (y_pred_hc == 1)))
        fn = int(np.sum((y_true_hc == 1) & (y_pred_hc == 0)))
        tn = int(np.sum((y_true_hc == 0) & (y_pred_hc == 0)))
        hc_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        hc_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        hc_f1 = (2 * hc_prec * hc_rec) / (hc_prec + hc_rec) if (hc_prec + hc_rec) > 0 else 0.0
        merged_hc = {
            "precision": float(hc_prec),
            "recall": float(hc_rec),
            "f1": float(hc_f1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": int(np.sum(y_true_hc == 1))
        }
            
        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "macro_f1_3class": float(macro_f1_3class),
            "evaluable_classes": evaluable_classes,
            "within_one_level": float(within_one),
            "mae": float(mae),
            "per_class": per_class,
            "merged_hc": merged_hc,
            "confusion_matrix": cm
        }
        
    subsets = {
        "ALL GOLD ISSUES (N=130)": gold_df,
        "RANDOM SUBSET ONLY (N=60)": gold_df[gold_df["sample"] == "random"]
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    lines = []
    
    def log(msg=""):
        print(msg)
        lines.append(msg)
        
    log("=" * 78)
    log("GITHUB ISSUE PRIORITY PREDICTION EVALUATION REPORT")
    log("=" * 78)
    log("Evaluation against hand-annotated Gold Set (N=130 issues).")
    log("DISCLAIMER: Weak supervision was used for ML training. Training excluded all gold issues.")
    log("NOTE ON CLASS EVALUATION: Per-class metrics are reported only for classes with n >= 5.")
    log("Classes with n < 5 are marked as 'not evaluable (n=X)' due to small-sample unreliability.\n")
    
    for sub_name, s_df in subsets.items():
        log("=" * 78)
        log(f"SUBSET: {sub_name}")
        log("=" * 78)
        
        y_t = s_df["gold_priority"].tolist()
        
        # Always-Low Majority Baseline
        y_base = ["Low"] * len(y_t)
        base_metrics = compute_metrics(y_t, y_base)
        log("\n--- Baseline: ALWAYS-LOW MAJORITY CLASS ---")
        log(f"Overall Accuracy         : {base_metrics['accuracy']:.2%}")
        log(f"Within-One-Level Accuracy: {base_metrics['within_one_level']:.2%}")
        log(f"Mean Absolute Error (MAE): {base_metrics['mae']:.3f} levels")
        log(f"Macro-F1 (n>=5 classes)  : {base_metrics['macro_f1']:.4f} (classes: {', '.join(base_metrics['evaluable_classes'])})")
        log(f"Macro-F1 (Low, Med, High): {base_metrics['macro_f1_3class']:.4f} (Critical dropped)")
        b_hc = base_metrics["merged_hc"]
        b_counts = f"TP={b_hc['tp']}, FP={b_hc['fp']}, FN={b_hc['fn']}, TN={b_hc['tn']}"
        log(f"Merged High+Critical     : Precision={b_hc['precision']:.2%}, Recall={b_hc['recall']:.2%}, F1={b_hc['f1']:.4f} ({b_counts}, Support={b_hc['support']})\n")
        
        for m in modes:
            m_title = "RULE-BASED" if m == "rules" else ("ML MODEL (LOGISTIC REGRESSION)" if m == "model" else "HYBRID CONSENSUS")
            log(f"\n--- Method: {m_title} ---")
            
            y_p = s_df[f"pred_{m}"].tolist()
            metrics = compute_metrics(y_t, y_p)
            
            log(f"Overall Accuracy         : {metrics['accuracy']:.2%}")
            log(f"Within-One-Level Accuracy: {metrics['within_one_level']:.2%}")
            log(f"Mean Absolute Error (MAE): {metrics['mae']:.3f} levels")
            log(f"Macro-F1 (n>=5 classes)  : {metrics['macro_f1']:.4f} (classes: {', '.join(metrics['evaluable_classes'])})")
            log(f"Macro-F1 (Low, Med, High): {metrics['macro_f1_3class']:.4f} (Critical dropped)\n")
            
            log("Per-Class Metrics:")
            log(f"  {'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
            log("  " + "-" * 48)
            for lvl in PRIORITY_LEVELS:
                pc = metrics["per_class"][lvl]
                if pc["support"] < 5:
                    eval_msg = f"not evaluable (n={pc['support']})"
                    log(f"  {lvl:<10} {eval_msg:<32} {pc['support']:<8}")
                else:
                    log(f"  {lvl:<10} {pc['precision']:<10.2%} {pc['recall']:<10.2%} {pc['f1']:<10.4f} {pc['support']:<8}")
            
            # Merged High+Critical row with raw counts
            m_hc = metrics["merged_hc"]
            counts_str = f"TP={m_hc['tp']}, FP={m_hc['fp']}, FN={m_hc['fn']}, TN={m_hc['tn']}"
            log("  " + "-" * 75)
            log(f"  {'High+Critical':<10} {m_hc['precision']:<10.2%} {m_hc['recall']:<10.2%} {m_hc['f1']:<10.4f} {counts_str:<26} {m_hc['support']:<8}")
                
            log("\nConfusion Matrix (Rows = Ground Truth Gold, Columns = Predicted):")
            log(f"  Labels order: {PRIORITY_LEVELS}")
            for r_idx, row in enumerate(metrics["confusion_matrix"]):
                row_str = "  ".join(f"{val:>6}" for val in row)
                log(f"  {PRIORITY_LEVELS[r_idx]:<10}: {row_str}")
            log()
            
    # Add Baseline prediction column for uniform bootstrap evaluation
    gold_df["pred_base"] = ["Low"] * len(gold_df)

    # Add held-out weak label agreement section
    log("=" * 78)
    log("HELD-OUT WEAK-LABEL AGREEMENT (TRAINING FIDELITY)")
    log("=" * 78)
    log(f"Weak-label agreement on 20% validation split: {bundle['val_agreement']:.2%}")
    log(f"Macro-F1 on weak labels: {bundle['val_macro_f1']:.4f}")
    log("LABELLED CLEARLY AS AGREEMENT, NOT HUMAN ACCURACY.")
    log("=" * 78)

    # ============================================================
    # DEV / TEST STRATIFIED SPLIT & SCORE THRESHOLD OPTIMIZATION
    # ============================================================
    log("\n" + "=" * 78)
    log("DEV (N=65) / TEST (N=65) STRATIFIED SPLIT & THRESHOLD OPTIMIZATION")
    log("=" * 78)
    log("130 gold issues split 50/50 into dev and test, stratified by gold label (seed=42).")
    log("Medium (1..3) and High (3..6) score thresholds swept strictly on DEV.")
    log("Selection criterion: dev Macro-F1 across Low, Medium, High.\n")

    dev_df, test_df = train_test_split(
        gold_df, test_size=0.5, random_state=42, stratify=gold_df["gold_priority"]
    )
    dev_df = dev_df.copy()
    test_df = test_df.copy()

    log(f"Dev class distribution : {dict(dev_df['gold_priority'].value_counts())}")
    log(f"Test class distribution: {dict(test_df['gold_priority'].value_counts())}\n")

    # Pre-extract rule score and hard_critical for dev issues
    dev_records = []
    for _, grow in dev_df.iterrows():
        iid = grow["issue_id"]
        info = issue_map.get(iid, {})
        title = info.get("title", grow.get("title", ""))
        desc = info.get("description", "")
        labels = info.get("labels", "")
        n_c = info.get("comments", 0)
        n_u = comm_user_map.get(iid, 1 if n_c > 0 else 0)
        lvl, score, reasons, hard_crit = rule_based_priority(title, desc, labels, n_c, n_u)
        dev_records.append({
            "iid": iid, "gold": grow["gold_priority"],
            "score": score, "hard_crit": hard_crit
        })

    # Grid sweep on DEV
    dev_sweep_results = []
    log("--- THRESHOLD SWEEP TABLE ON DEV (65 ISSUES) ---")
    log(f"{'T_med':<7} {'T_high':<8} {'Dev Macro-F1':<14} {'High+ Prec':<12} {'High+ Rec':<11} {'High+ F1':<10} {'TP':<4} {'FP':<4}")
    log("-" * 72)

    for t_med in [1, 2, 3]:
        for t_high in [3, 4, 5, 6]:
            if t_high <= t_med:
                continue
            y_true_dev = [d["gold"] for d in dev_records]
            y_pred_dev = []
            for d in dev_records:
                if d["hard_crit"] or d["score"] >= 7:
                    p = "Critical"
                elif d["score"] >= t_high:
                    p = "High"
                elif d["score"] >= t_med:
                    p = "Medium"
                else:
                    p = "Low"
                y_pred_dev.append(p)

            f1_per = f1_score(y_true_dev, y_pred_dev, labels=["Low", "Medium", "High"], average=None, zero_division=0)
            macro_f1 = float(np.mean(f1_per))

            y_true_hc = np.array([1 if y in ("High", "Critical") else 0 for y in y_true_dev])
            y_pred_hc = np.array([1 if y in ("High", "Critical") else 0 for y in y_pred_dev])
            tp = int(np.sum((y_true_hc == 1) & (y_pred_hc == 1)))
            fp = int(np.sum((y_true_hc == 0) & (y_pred_hc == 1)))
            fn = int(np.sum((y_true_hc == 1) & (y_pred_hc == 0)))
            prec_hc = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_hc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_hc = (2 * prec_hc * rec_hc) / (prec_hc + rec_hc) if (prec_hc + rec_hc) > 0 else 0.0

            res_entry = {
                "t_med": t_med, "t_high": t_high,
                "macro_f1": macro_f1,
                "hc_prec": prec_hc, "hc_rec": rec_hc, "hc_f1": f1_hc,
                "tp": tp, "fp": fp, "fn": fn
            }
            dev_sweep_results.append(res_entry)
            log(f"{t_med:<7} {t_high:<8} {macro_f1:<14.4f} {prec_hc:<12.2%} {rec_hc:<11.2%} {f1_hc:<10.4f} {tp:<4} {fp:<4}")

    best_setting = max(dev_sweep_results, key=lambda x: x["macro_f1"])
    log(f"\nOptimal Setting Chosen by Dev Macro-F1: T_med={best_setting['t_med']}, T_high={best_setting['t_high']} (Dev Macro-F1: {best_setting['macro_f1']:.4f})\n")

    # Evaluate on TEST (65 issues) exactly once
    test_records = []
    for _, grow in test_df.iterrows():
        iid = grow["issue_id"]
        info = issue_map.get(iid, {})
        title = info.get("title", grow.get("title", ""))
        desc = info.get("description", "")
        labels = info.get("labels", "")
        n_c = info.get("comments", 0)
        n_u = comm_user_map.get(iid, 1 if n_c > 0 else 0)
        lvl, score, reasons, hard_crit = rule_based_priority(title, desc, labels, n_c, n_u)
        test_records.append({
            "iid": iid, "gold": grow["gold_priority"],
            "score": score, "hard_crit": hard_crit
        })

    def eval_test_thresholds(tm, th):
        y_true_t = [d["gold"] for d in test_records]
        y_pred_t = []
        for d in test_records:
            if d["hard_crit"] or d["score"] >= 7:
                p = "Critical"
            elif d["score"] >= th:
                p = "High"
            elif d["score"] >= tm:
                p = "Medium"
            else:
                p = "Low"
            y_pred_t.append(p)
        acc = float(accuracy_score(y_true_t, y_pred_t))
        f1_per = f1_score(y_true_t, y_pred_t, labels=["Low", "Medium", "High"], average=None, zero_division=0)
        macro_f1 = float(np.mean(f1_per))
        y_true_hc = np.array([1 if y in ("High", "Critical") else 0 for y in y_true_t])
        y_pred_hc = np.array([1 if y in ("High", "Critical") else 0 for y in y_pred_t])
        tp = int(np.sum((y_true_hc == 1) & (y_pred_hc == 1)))
        fp = int(np.sum((y_true_hc == 0) & (y_pred_hc == 1)))
        fn = int(np.sum((y_true_hc == 1) & (y_pred_hc == 0)))
        prec_hc = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_hc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_hc = (2 * prec_hc * rec_hc) / (prec_hc + rec_hc) if (prec_hc + rec_hc) > 0 else 0.0
        return {
            "acc": acc, "macro_f1": macro_f1,
            "hc_prec": prec_hc, "hc_rec": rec_hc, "hc_f1": f1_hc,
            "tp": tp, "fp": fp, "fn": fn,
            "f1_low": float(f1_per[0]), "f1_med": float(f1_per[1]), "f1_high": float(f1_per[2])
        }

    opt_test = eval_test_thresholds(best_setting["t_med"], best_setting["t_high"])
    orig_test = eval_test_thresholds(1, 4)

    log("--- SINGLE UNBIASED EVALUATION ON TEST SET (65 ISSUES) ---")
    log(f"{'Setting':<38} {'Accuracy':<10} {'Macro-F1':<10} {'High+ Prec':<12} {'High+ Rec':<11} {'High+ F1':<10} {'TP':<4} {'FP':<4}")
    log("-" * 99)
    log(f"{'Dev-Tuned Pick (T_med=' + str(best_setting['t_med']) + ', T_high=' + str(best_setting['t_high']) + ')':<38} {opt_test['acc']:<10.2%} {opt_test['macro_f1']:<10.4f} {opt_test['hc_prec']:<12.2%} {opt_test['hc_rec']:<11.2%} {opt_test['hc_f1']:<10.4f} {opt_test['tp']:<4} {opt_test['fp']:<4}")
    log(f"{'Original Thresholds (T_med=1, T_high=4)':<38} {orig_test['acc']:<10.2%} {orig_test['macro_f1']:<10.4f} {orig_test['hc_prec']:<12.2%} {orig_test['hc_rec']:<11.2%} {orig_test['hc_f1']:<10.4f} {orig_test['tp']:<4} {orig_test['fp']:<4}\n")

    # ============================================================
    # 1000-RESAMPLE BOOTSTRAP 95% CONFIDENCE INTERVALS
    # ============================================================
    log("=" * 78)
    log("1000-RESAMPLE BOOTSTRAP 95% CONFIDENCE INTERVALS")
    log("=" * 78)
    log("Empirical 2.5% and 97.5% percentiles computed over 1000 bootstrap resamples (seed=42).")
    log("Demonstrates estimation variance, especially for rare classes like High (n=5 overall, n=3 test).\n")

    def compute_bootstrap_cis(df_sub, n_boot=1000, seed=42):
        rng = np.random.RandomState(seed)
        n = len(df_sub)
        out = {}
        for m_col in ["pred_base", "pred_rules", "pred_model", "pred_hybrid"]:
            accs, f1s, recs, precs = [], [], [], []
            y_t_arr = df_sub["gold_priority"].values
            y_p_arr = df_sub[m_col].values
            for _ in range(n_boot):
                idx = rng.randint(0, n, size=n)
                yt = y_t_arr[idx]
                yp = y_p_arr[idx]
                accs.append(accuracy_score(yt, yp))
                f1_per = f1_score(yt, yp, labels=["Low", "Medium", "High"], average=None, zero_division=0)
                f1s.append(np.mean(f1_per))
                yt_hc = np.array([1 if y in ("High", "Critical") else 0 for y in yt])
                yp_hc = np.array([1 if y in ("High", "Critical") else 0 for y in yp])
                tp = np.sum((yt_hc == 1) & (yp_hc == 1))
                fp = np.sum((yt_hc == 0) & (yp_hc == 1))
                fn = np.sum((yt_hc == 1) & (yp_hc == 0))
                precs.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
                recs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            out[m_col] = {
                "acc": (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))),
                "macro_f1": (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))),
                "hc_rec": (float(np.percentile(recs, 2.5)), float(np.percentile(recs, 97.5))),
                "hc_prec": (float(np.percentile(precs, 2.5)), float(np.percentile(precs, 97.5)))
            }
        return out

    for scope_title, s_frame in [("ALL 130 GOLD ISSUES", gold_df), ("TEST SET ONLY (N=65)", test_df)]:
        log(f"--- Scope: {scope_title} ---")
        cis = compute_bootstrap_cis(s_frame)
        for m_key, m_lbl in [("pred_base", "Always-Low Baseline"),
                             ("pred_rules", "Rule-Based Scorer"),
                             ("pred_model", "ML Model (LogisticReg)"),
                             ("pred_hybrid", "Hybrid Consensus")]:
            c = cis[m_key]
            log(f"  {m_lbl:<24}: Acc 95% CI=[{c['acc'][0]:.1%}, {c['acc'][1]:.1%}]  "
                f"Macro-F1 95% CI=[{c['macro_f1'][0]:.4f}, {c['macro_f1'][1]:.4f}]  "
                f"High+ Rec 95% CI=[{c['hc_rec'][0]:.1%}, {c['hc_rec'][1]:.1%}]  "
                f"High+ Prec 95% CI=[{c['hc_prec'][0]:.1%}, {c['hc_prec'][1]:.1%}]")
        log()
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        
    print(f"\nEvaluation report written to: {output_file}")
    return subsets


if __name__ == "__main__":
    # Run training, batch predictions, and gold-set evaluation
    bundle = train_priority_model()
    batch_predict_all_issues()
    evaluate_gold_set()
