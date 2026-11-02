"""
GitHub Issue Classification (Hybrid: rules + TF-IDF Logistic Regression)
 
Categories: Bug, Feature, Documentation, Security, Performance, Question
 
Pipeline:
    Title + Description -> Text cleaning -> Strong rule matching
        match    -> category (rule-based)
        no match -> TF-IDF + Logistic Regression -> category
    category -> LABEL_MAP -> suggested GitHub label
 
Run (from the classification folder):
    python issue_classification.py
"""
 
import os
import re
import pickle
import warnings
 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
 
warnings.filterwarnings("ignore")
 
# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "..", "cleaned_issues.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
MODEL_FILE = os.path.join(MODEL_DIR, "issue_classifier.pkl")
 
MAX_ROWS = 60000
RANDOM_STATE = 42
 
CATEGORIES = ["Bug", "Feature", "Documentation", "Security", "Performance", "Question"]
 
LABEL_MAP = {
    "Bug": "bug",
    "Feature": "enhancement",
    "Documentation": "documentation",
    "Security": "security",
    "Performance": "performance",
    "Question": "question",
}
 
# ============================================================
# RULES  (checked in this order; first match wins)
# (category, patterns, title_only)
#   title_only=True -> rule is applied to the issue title only
# Order matters: Security / Performance / Bug are the most specific signals.
# Question comes before Documentation / Feature so that
# "How can I add support for X?" is treated as a question.
# ============================================================
RULES = [
    ("Security", [
        r"\bsecurity\b", r"\bvulnerab\w*", r"\bcve[- ]?\d*", r"\bxss\b", r"\bcsrf\b", r"\bssrf\b",
        r"\bsql injection\b", r"\b(code|command|header) injection\b", r"\bexploit\w*",
        r"\bauthenticat\w*\s+bypass", r"\bbypass\w*\s+(auth\w*|security)",
        r"\bprivilege escalation\b", r"\bpath traversal\b", r"\bremote code execution\b",
        r"\brce\b", r"\bdenial of service\b", r"\binsecure\b", r"\bunauthori[sz]ed\b",
        r"\bsanitiz\w*", r"\bsensitive (data|information)\b", r"\bopen redirect\b",
        r"\b(token|password|secret|credential|api key)s?\s+(is |are )?(leak(s|ed)?|exposed)\b",
        r"\bleak(s|ed|ing)?\s+(a\s+|the\s+)?(token|password|secret|credential|api key)s?\b",
        r"\bpasswords?\s+(stored|saved|logged)\s+in\s+(plain ?text|logs?)\b",
    ], False),
    ("Performance", [
        r"\bslow\w*", r"\bperformance\b", r"\blatency\b", r"\bmemory (leak|usage|consumption)\b",
        r"\bout of memory\b", r"\boom\b", r"\bhigh (cpu|memory|load)\b", r"\bcpu usage\b",
        r"\bbottleneck\b", r"\bspeed( up)?\b", r"\boptimi[sz]\w*", r"\btakes (too |very )?long\b",
        r"\blag(s|gy|ging)?\b", r"\bfaster\b", r"\bperformant\b", r"\bload time\b",
    ], False),
    ("Bug", [
        r"\bcrash\w*", r"\bbug\b", r"\berror\w*", r"\bexception\w*", r"\btraceback\b",
        r"\bfail(s|ed|ing|ure|ures)?\b", r"\bbroken\b", r"\bnot working\b",
        r"\b(does not|doesn'?t|do not|don'?t|won'?t|cannot|can'?t) work\b",
        r"\b(can'?t|cannot|unable to)\s+(install|import|run|start|build|connect|login|log in|open|load)\b",
        r"\bstops? working\b", r"\bthrows?\b", r"\braises?\b", r"\bregression\b",
        r"\bsegfault\b", r"\bunexpected\b", r"\bincorrect\w*", r"\bwrong\b",
        r"\bhang(s|ing)?\b", r"\bfreez\w+", r"\bpanic\b",
    ], False),
    ("Question", [
        r"\?",
        r"^(how|what|why|when|where|which|who|should|could|is there|are there|is it)\b",
        r"^(can|does|do|is|are)\b(?!['\u2019]t)",
        r"\bhow (to|do|can|should)\b", r"\bis it possible\b", r"\bany way to\b",
        r"\bneed help\b", r"\bhelp with\b",
    ], True),
    ("Documentation", [
        r"\bdocs?\b", r"\bdocumentation\b", r"\breadme\b", r"\btypos?\b", r"\btutorials?\b",
        r"\bdocstrings?\b", r"\bchangelog\b", r"\bapi reference\b",
        r"\bexamples? (is|are) (missing|outdated|wrong)\b",
    ], False),
    ("Feature", [
        r"\badd (support|an? option|option|ability|a new|new)\b", r"\bsupport for\b",
        r"\bfeature( request)?\b", r"\benhancement\b", r"\bimplement\w*",
        r"\bwould be (nice|great|useful)\b", r"\bit would be\b", r"\bpropos(e|al)\b",
        r"\brequest for\b", r"\ballow (users|to|the|for)\b", r"\bintroduce\b",
        r"\bsuggestion\b", r"\bnew api\b", r"\bimprove\b", r"\bmake it possible\b",
    ], False),
]
 
_COMPILED_RULES = [
    (cat, [re.compile(p, re.IGNORECASE) for p in pats], title_only)
    for cat, pats, title_only in RULES
]
 
 
# ============================================================
# TEXT CLEANING
# ============================================================
def clean_text(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        text = ""
    text = str(text).lower()
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
 
 
# ============================================================
# RULE-BASED CLASSIFIER
# ============================================================
def rule_match(title, description="", title_only_scope=False):
    """Return (category, matched_text) for the first strong rule, else (None, None).
    The title is checked first, then title + description."""
    title = (title or "").strip()
    description = (description or "").strip()
 
    scopes = [("title", title)]
    if not title_only_scope and description:
        scopes.append(("full", f"{title} {description}"))
 
    for scope_name, text in scopes:
        if not text:
            continue
        for cat, patterns, title_only in _COMPILED_RULES:
            if title_only and scope_name != "title":
                continue
            for p in patterns:
                m = p.search(text)
                if m:
                    return cat, m.group(0).strip()
    return None, None
 
 
def rule_based_category(title, description="", title_only_scope=False):
    return rule_match(title, description, title_only_scope)[0]
 
 
# ============================================================
# MODEL LOADING + HYBRID PREDICTION (used by label_recommender.py)
# ============================================================
def load_bundle(path=MODEL_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\nRun 'python issue_classification.py' first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)
 
 
def classify_issue(title, description="", bundle=None):
    """Hybrid classification.
 
    Returns None for empty input, otherwise a dict with:
    category, label, method, confidence (None for rules), level, matched, top3
    """
    title = (title or "").strip()
    description = (description or "").strip()
    if not title and not description:
        return None
 
    cat, matched = rule_match(title, description)
    if cat:
        return {
            "category": cat,
            "label": LABEL_MAP[cat],
            "method": "Rule-based",
            "confidence": None,
            "level": "High",
            "matched": matched,
            "top3": [],
        }
 
    if bundle is None:
        bundle = load_bundle()
    vec, model = bundle["vectorizer"], bundle["model"]
 
    text = clean_text(f"{title} {title} {description}")  # title counted twice = more weight
    X = vec.transform([text])
    if X.nnz == 0:
        return {
            "category": "Unknown",
            "label": "needs-triage",
            "method": "None (no known words)",
            "confidence": 0.0,
            "level": "Low",
            "matched": None,
            "top3": [],
        }
 
    proba = model.predict_proba(X)[0]
    order = np.argsort(proba)[::-1]
    best = int(order[0])
    conf = float(proba[best])
    level = "High" if conf >= 0.60 else "Medium" if conf >= 0.40 else "Low"
    cat = str(model.classes_[best])
    return {
        "category": cat,
        "label": LABEL_MAP.get(cat, "needs-triage"),
        "method": "ML (Logistic Regression)",
        "confidence": conf,
        "level": level,
        "matched": None,
        "top3": [(str(model.classes_[i]), float(proba[i])) for i in order[:3]],
    }
 
 
# ============================================================
# TRAINING DATA
# ============================================================
# Small hand-written seed set so every category is always represented
# (Security and Performance are rare in the dataset).
SEED_EXAMPLES = [
    ("Bug", "application crashes on startup"), ("Bug", "unexpected exception thrown during request"),
    ("Bug", "function returns wrong value"), ("Bug", "page fails to load with traceback"),
    ("Bug", "button does not work after update"), ("Bug", "server hangs when file is uploaded"),
    ("Feature", "add support for new file format"), ("Feature", "feature request add dark mode"),
    ("Feature", "would be nice to have an export option"), ("Feature", "implement caching layer"),
    ("Feature", "allow users to customize settings"), ("Feature", "proposal new plugin system"),
    ("Documentation", "readme is missing installation steps"), ("Documentation", "typo in the docs"),
    ("Documentation", "update tutorial for latest version"), ("Documentation", "improve api documentation"),
    ("Documentation", "missing docstrings in module"), ("Documentation", "changelog not updated"),
    ("Security", "security vulnerability in login"), ("Security", "xss injection in form input"),
    ("Security", "cve found in dependency"), ("Security", "token leaked in logs"),
    ("Security", "authentication bypass using header"), ("Security", "sql injection in search endpoint"),
    ("Security", "password stored in plain text"), ("Security", "unauthorized access to admin page"),
    ("Security", "csrf protection missing on form"), ("Security", "path traversal when serving files"),
    ("Performance", "application is very slow"), ("Performance", "high memory usage under load"),
    ("Performance", "optimize database queries"), ("Performance", "latency increased after update"),
    ("Performance", "page takes too long to load"), ("Performance", "reduce cpu usage in worker"),
    ("Performance", "memory leak in long running process"), ("Performance", "slow response for large requests"),
    ("Performance", "speed up startup time"), ("Performance", "bottleneck in request handling"),
    ("Question", "how can i configure this"), ("Question", "what is the best way to deploy"),
    ("Question", "is it possible to use this with docker"), ("Question", "why does this behave like this"),
    ("Question", "need help with environment variables"), ("Question", "does this support python 3"),
]
 
# Hand-labelled test set, NOT derived from the rules or the dataset.
# Small, so treat the result as a sanity check rather than a benchmark.
GOLD_TEST = [
    ("App crashes on startup", "Bug"),
    ("Login button throws exception when clicked", "Bug"),
    ("Returns wrong value for empty list", "Bug"),
    ("Cannot install package on Windows", "Bug"),
    ("Server hangs after second request", "Bug"),
    ("Unexpected output when parsing dates", "Bug"),
    ("Add support for PostgreSQL", "Feature"),
    ("Feature request: dark mode", "Feature"),
    ("It would be great to export reports as PDF", "Feature"),
    ("Implement user roles", "Feature"),
    ("Allow users to customize the dashboard", "Feature"),
    ("Proposal: plugin system", "Feature"),
    ("Typo in README", "Documentation"),
    ("Update installation docs for Python 3.12", "Documentation"),
    ("Improve API documentation", "Documentation"),
    ("Missing docstrings in utils module", "Documentation"),
    ("Tutorial is outdated", "Documentation"),
    ("Changelog not updated for v2.0", "Documentation"),
    ("SQL injection in search form", "Security"),
    ("XSS vulnerability in comments", "Security"),
    ("Password stored in plain text", "Security"),
    ("CVE-2024-1234 affects dependency", "Security"),
    ("Authentication bypass possible via header", "Security"),
    ("Token leaked in logs", "Security"),
    ("Application is very slow", "Performance"),
    ("High memory usage under load", "Performance"),
    ("Optimize database queries", "Performance"),
    ("Latency increased after upgrade", "Performance"),
    ("Page takes too long to load", "Performance"),
    ("Reduce CPU usage in worker", "Performance"),
    ("How can I configure Flask?", "Question"),
    ("What is the best way to deploy?", "Question"),
    ("Is it possible to use this with Docker?", "Question"),
    ("Why does the default port differ?", "Question"),
    ("Does this work with Python 3.8?", "Question"),
    ("Need help with environment variables", "Question"),
]
 
 
# ============================================================
# DATA LOADING
# ============================================================
def find_column(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None
 
 
def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df):,} rows. Columns: {list(df.columns)}")
 
    title_col = find_column(df, ["title", "issue_title", "clean_title", "cleaned_title"])
    body_col = find_column(df, ["body", "description", "issue_body", "clean_body",
                                "cleaned_body", "text", "clean_text", "cleaned_text"])
 
    if title_col is None and body_col is None:
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if not text_cols:
            raise ValueError("No text columns found in the dataset.")
        df["title_"] = df[text_cols[0]].fillna("").astype(str)
        df["body_"] = ""
    else:
        df["title_"] = df[title_col].fillna("").astype(str) if title_col else ""
        df["body_"] = df[body_col].fillna("").astype(str) if body_col and body_col != title_col else ""
 
    print(f"Using title column: {title_col}, body column: {body_col}")
    df = df[(df["title_"].str.strip() != "") | (df["body_"].str.strip() != "")]
    return df.reset_index(drop=True)
 
 
# ============================================================
# EVALUATION
# ============================================================
def evaluate_gold(bundle):
    """Evaluate the full hybrid system on the hand-labelled gold set."""
    rows, correct = [], 0
    by_method = {"Rule-based": [0, 0], "ML": [0, 0]}
    for title, expected in GOLD_TEST:
        r = classify_issue(title, "", bundle)
        ok = r["category"] == expected
        correct += ok
        key = "Rule-based" if r["method"] == "Rule-based" else "ML"
        by_method[key][0] += ok
        by_method[key][1] += 1
        rows.append((title, expected, r["category"], r["method"], ok))
 
    print("\n" + "=" * 70)
    print("HYBRID EVALUATION ON HAND-LABELLED GOLD SET")
    print("=" * 70)
    print(f"Examples: {len(GOLD_TEST)}  |  Correct: {correct}  |  Accuracy: {correct / len(GOLD_TEST):.2%}")
    for k, (c, n) in by_method.items():
        if n:
            print(f"  {k:<11}: {c}/{n} correct")
    wrong = [r for r in rows if not r[4]]
    if wrong:
        print("\nMisclassified:")
        for title, exp, got, method, _ in wrong:
            print(f"  '{title}' -> expected {exp}, got {got} ({method})")
    return correct / len(GOLD_TEST), rows
 
 
# ============================================================
# MAIN (TRAINING)
# ============================================================
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
 
    print("=" * 70)
    print("GITHUB ISSUE CLASSIFICATION (HYBRID)")
    print("=" * 70)
 
    df = load_data()
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
 
    # ---- Weak supervision: label by TITLE rules first (less noisy than body rules)
    df["category"] = [rule_based_category(t, "", title_only_scope=True) for t in df["title_"]]
    labelled = df[df["category"].notna()]
    if len(labelled) < 300:  # not enough title matches -> also use body text
        df["category"] = [rule_based_category(t, b) for t, b in zip(df["title_"], df["body_"])]
        labelled = df[df["category"].notna()]
 
    print(f"\nWeakly labelled rows: {len(labelled):,} of {len(df):,}")
    print(labelled["category"].value_counts().to_string())
 
    texts = [clean_text(f"{t} {t} {b}") for t, b in zip(labelled["title_"], labelled["body_"])]
    labels = labelled["category"].tolist()
 
    for cat, txt in SEED_EXAMPLES:
        texts.append(clean_text(txt))
        labels.append(cat)
 
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
 
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                 stop_words="english", min_df=2, sublinear_tf=True)
    X_train = vectorizer.fit_transform(X_train_txt)
    X_test = vectorizer.transform(X_test_txt)
 
    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=list(model.classes_))
 
    print("\n" + "=" * 70)
    print("ML MODEL EVALUATION (on weakly labelled test split)")
    print("=" * 70)
    print(f"Accuracy: {acc:.4f}   Macro-F1: {macro_f1:.4f}")
    print("NOTE: labels are auto-generated from rules, so this is NOT real-world accuracy.\n")
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print([str(c) for c in model.classes_])
    print(cm)
 
    bundle = {
        "vectorizer": vectorizer,
        "model": model,
        "categories": CATEGORIES,
        "weak_label_accuracy": acc,
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nModel saved to: {MODEL_FILE}")
 
    gold_acc, _ = evaluate_gold(bundle)
 
    with open(os.path.join(RESULT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Weak-label test accuracy: {acc:.4f}\nMacro-F1: {macro_f1:.4f}\n\n{report}\n")
        f.write(f"Confusion matrix:\n{[str(c) for c in model.classes_]}\n{cm}\n\n")
        f.write(f"Hybrid accuracy on hand-labelled gold set ({len(GOLD_TEST)} examples): {gold_acc:.4f}\n")
 
    demos = [
        "Application crashes when running request",
        "Add support for new API",
        "Improve Flask documentation",
        "Security vulnerability in authentication",
        "Application is very slow",
        "How can I configure Flask?",
    ]
    print("\n" + "=" * 70)
    print("HYBRID PREDICTION DEMO")
    print("=" * 70)
    for title in demos:
        r = classify_issue(title, "", bundle)
        conf = "N/A (strong rule matched)" if r["confidence"] is None else f"{r['confidence'] * 100:.2f}%"
        print(f"Title     : {title}")
        print(f"Category  : {r['category']}")
        print(f"Method    : {r['method']}")
        print(f"Confidence: {conf}\n")
 
    print("=" * 70)
    print("ISSUE CLASSIFICATION COMPLETED")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()