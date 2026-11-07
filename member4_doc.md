# GitHub Issue Intelligence
## Member 4 – Issue Summarization and Priority Prediction Documentation

### 1. Introduction

GitHub Issue Intelligence is a collaborative platform designed to analyse open-source GitHub issues using NLP and machine learning. As Member 4 of the team, my responsibility encompasses two core analytical capabilities:

1. **Extractive Issue Summarization & Context Extraction**: Automatically distilling long issue descriptions and community discussions into concise, actionable summaries (~70 words max), while extracting diagnostic metadata including traceback exceptions (`key_error`), execution runtime environments (`environment`), and recent discussion highlights (`discussion_summary`).
2. **Four-Level Priority Prediction**: Categorising issues into `Low`, `Medium`, `High`, or `Critical` using a weak-supervision framework that combines an interpretable rule scoring engine with a machine learning classifier into a hybrid consensus predictor.

All code and artifacts for this module are located in `summarization/`. The module is fully decoupled from other team members' code while providing a clean integration interface for Member 5 (FastAPI backend and Streamlit dashboard).

---

### 2. Objectives

The primary objectives of the Member 4 module are:

- **Markdown Cleaning without Information Loss**: Strip noisy GitHub markdown elements (HTML comments, template boilerplate, image tags, code fences, greetings, closings) while strictly preserving technical identifiers such as snake_case tokens (`before_request`, `app.route`, `test_client`).
- **Extractive Summarization without Heavy Neural Dependencies**: Segment sentences without NLTK, construct sentence graphs using TF-IDF cosine similarity, and rank sentences using PageRank (TextRank) boosted by title similarity, early position, and diagnostic cue words, penalising run-on sentences.
- **Diagnostic Context Extraction**: Extract the root cause exception line from Python tracebacks (e.g. `AttributeError: ...`), detect runtime versions (Python, Flask, Werkzeug), and summarize comment discussions with a recency bias.
- **Weak Supervision for Priority**: Tackle the absence of native priority labels in `pallets/flask` by designing transparent, weighted heuristic rules, training an ML classifier on weakly labelled data (strictly excluding the evaluation gold set), and merging predictions via a hybrid consensus model with hard-critical overrides.
- **Human-in-the-Loop Gold Evaluation**: Build a 130-issue curated gold set (60 random + 30 severity-enriched + 40 metadata-only sampled) to rigorously measure accuracy, precision, recall, macro-F1, within-one-level accuracy, MAE, and merged High+Critical detection across rules, ML, and hybrid methods compared against the always-Low baseline.
- **Unified Integration**: Expose `analyze_issue(title, description, labels, comments)` combining summarization, priority prediction, and Member 2's category classification into a single payload with explicit agreement reporting.

---

### 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Core development language |
| Pandas | Dataset manipulation and CSV processing |
| NumPy & SciPy | Mathematical operations, graph PageRank iteration, and sparse matrix handling |
| Scikit-learn | TF-IDF vectorization, feature scaling, and multi-class Logistic Regression |
| Regular Expressions (`re`) | Markdown cleaning, sentence splitting, pattern-based rule scoring, and exception parsing |
| Git | Version control |

> **Constraint Adherence**: No paid APIs, deep learning frameworks (PyTorch/TensorFlow), or external model downloads were used. Only standard Python libraries, NumPy, SciPy, Pandas, and Scikit-learn power the entire pipeline.

---

### 4. Dataset Context & The Weak Supervision Problem

The dataset collected by Member 1 contains **2,769 issues** from `pallets/flask` and **3,668 associated comments** from `raw_comments.csv`.

#### The Ground Truth Reality
A critical observation across the dataset is that **GitHub issues do not come with explicit priority labels**. The pre-existing label column consists of tags like `docs` (137 issues), `cli` (38), `typing` (23), `blueprints` (23), and `good first issue` (8), with only 16 issues tagged with `bug` and zero issues tagged with standard priority levels (`Low`, `Medium`, `High`, `Critical`).

To overcome this lack of ground truth, we adopted a **Weak Supervision** paradigm:
1. **Rule-Based Heuristic Labeling**: Construct transparent, domain-driven heuristic rules to score issues and assign initial weak labels across the dataset.
2. **ML Classifier Training on Weak Labels**: Train a statistical classifier to learn generalized patterns from the weak labels and surface linguistic and numeric interactions beyond hand-crafted rules.
3. **Gold Set Isolation**: Hand-annotate a representative 130-issue Gold Set and **strictly exclude all 130 gold issues** from the ML training pool (leaving 2,649 issues for training).
4. **Honest Reporting**: In all evaluations, clearly separate *agreement with weak labels* on held-out data (which measures how well the ML model learned the rules) from *human evaluation on the Gold Set* (which measures real-world accuracy).

---

### 5. Issue Summarization Engine (`summarizer.py`)

The summarizer pipeline distills multi-paragraph issues and technical logs into a structured 4-part analytical summary.

```
Raw Issue Description + Comments
   │
   ├─► Markdown Cleaner ──► Regex Sentence Splitter ──► TextRank Graph ──► Top 3 Extractive Sentences
   ├─► Traceback Parser ──► Tightened Exception Extractor ─────────────► key_error
   ├─► Version Pattern Matcher ─────────────────────────────────────────► environment (Python/Flask/Werkzeug)
   └─► Comment Recency Filter ──────────────────────────────────────────► discussion_summary (2 sentences)
```

#### 5.1 Text Cleaning & Preservation of Code Identifiers
GitHub issue descriptions contain noise that degrades summarization if left uncleaned. `clean_markdown()` performs targeted cleaning while protecting technical identifiers:
- Strips HTML comments (`<!-- ... -->`), GitHub issue template headers (e.g. `### Environment`, `### Steps to reproduce`), URLs, image embeds, blockquotes, and conversational greetings/closings (`Hi`, `Thanks in advance`, `Best regards`).
- Code fences (```` ```...``` ````) are stripped from the prose summarization input to prevent large stack traces from dominating extractive ranking.
- **Protection of snake_case Identifiers**: Standard cleaners often replace underscores with spaces, corrupting identifiers like `before_request` into `before request` or `request_ctx` into `request ctx`. Our cleaner explicitly preserves snake_case tokens using regex boundary lookaheads.

#### 5.2 Sentence Splitting without NLTK
To avoid external dependencies or neural tokenizers, `split_into_sentences()` implements a regex-based sentence boundary disambiguation system:
- Protects common Python and technical abbreviations: `e.g.`, `i.e.`, `etc.`, `vs.`, `fig.`, `app.`, `cls.`.
- Protects version numbers (`v1.1.2`, `2.0.0`) and decimal floats (`0.05`).
- Splits strictly on punctuation (`.`, `!`, `?`) followed by whitespace and an uppercase letter or end of string.
- Filters out fragments shorter than 15 characters or fewer than 3 words.

#### 5.3 Extractive TextRank with Diagnostic Cue Bonuses
Sentences are ranked using an enhanced graph-based PageRank algorithm:
1. **TF-IDF Cosine Graph**: Sentences are vectorized using sublinear TF-IDF word n-grams (1–2), and an undirected weighted adjacency graph is built from pairwise cosine similarities.
2. **PageRank Iteration**: Power iteration computes sentence centrality with a damping factor of $d = 0.85$ and convergence tolerance of $10^{-5}$.
3. **Domain Diagnostic Cue Bonuses**:
   - **Title Keyword Similarity**: Sentences containing words from the issue title receive a bonus ($+0.25 \times \text{overlap}$).
   - **Position Decay**: Early sentences receive an exponential position bonus ($e^{-0.2 \times \text{index}}$), prioritising the user's opening problem statement.
   - **Cue Word Boosting**: Sentences containing diagnostic indicator words (`expected`, `instead`, `fails`, `error`, `crash`, `regression`, `broke`, `exception`, `reproduce`) receive a $+0.25$ score bonus.
   - **Run-On Penalty**: Sentences exceeding 45 words are penalised by $0.70\times$ to avoid selecting unreadable run-on sentences.
4. **Top-3 Sentence Selection**: The top 3 ranked sentences are returned **in their original document order**, capped at approximately 70 words, ensuring each sentence terminates with clean punctuation.
5. **Fallback Rules**: Descriptions with $\le 40$ words of prose are preserved as-is (`short-prose` method); empty descriptions fall back directly to the title.

#### 5.4 Diagnostic Context Extraction
- **Tightened `key_error` Extraction**:
  - Target: Exact exception class and message from Python tracebacks.
  - Tightened logic: Accepts only lines where a valid Python exception class (e.g. `TypeError`, `ValueError`, `ModuleNotFoundError`, `mod.SomeError`, `werkzeug.routing.WebsocketMismatch`) appears at the START of the line, optionally followed by `: message`.
  - Rejection filters: Strictly rejects lines starting with a timestamp (e.g. `2021-10-08T09:59:35Z`), dates, or containing dictionary/set braces `{...}` (preventing gevent/server logs from being falsely identified as exceptions).
  - Traverses backwards from the final traceback block, looking for the exception line immediately following the stack frames (`File "..."`).
- **Runtime `environment` Extraction**: Uses regex to extract Python versions (`Python 3.9`), Flask versions (`Flask 2.0.2`), and Werkzeug versions (`Werkzeug 2.0.1`) from issue descriptions.
- **Discussion Summary (`discussion_summary`)**: For issues with comment threads, extracts the top 2 sentences from the discussion using an exponential recency weight ($w_i = e^{0.3 \times i}$), prioritising recent maintainer triage outcomes, workarounds, and pull request references over initial "me too" comments.

---

### 6. Priority Prediction Architecture (`priority_prediction.py`)

Priority is classified into four operational levels:
- **Critical (Level 3)**: Breaks installs or startup for many users, exploitable vulnerabilities in Flask itself, or silent data loss.
- **High (Level 2)**: Core functionality broken, severe regressions affecting many users without workaround, or unhandled crashes in core pipelines.
- **Medium (Level 1)**: Non-critical bugs, edge cases, minor regressions with workarounds, or deprecations.
- **Low (Level 0)**: Documentation, tutorials, typos, user questions, feature requests, unsupported setups, external extensions.

#### 6.1 Rule Scorer & Thresholds Audit
The heuristic engine evaluates text, labels, comment velocity, and community engagement to produce a transparent integer score and human-readable explanation:

| Signal Category | Condition / Keyword Patterns | Points |
|---|---|:---:|
| **Hard-Critical Override** | `rce`, `cve-\d+`, `remote code execution`, `auth bypass`, `sql injection`, `data loss`, `corrupt data`, `crashes on startup`, `breaks fresh install` | **Critical (Override)** |
| **Crash & Startup** | `traceback (most recent call last)`, `crash`, `fails to start`, `segfault`, `segmentation fault` | $+3$ |
| **Regressions & Breaking Changes** | `regression`, `broke in`, `broken in`, `breaking change`, `worked in`, `after update`, `after upgrading` | $+3$ |
| **Security & Production Impact** | `security`, `vulnerability`, `denial of service`, `dos`, `production`, `memory leak` | $+3$ |
| **Installation & Import Failures** | `cannot import`, `modulenotfounderror`, `importerror`, `fails to install`, `pip install fail` | $+2$ |
| **Error Signals in Title** | Title contains `error`, `exception`, `fail`, `crash`, `bug`, `broken`, `cannot` | $+2$ |
| **Python Traceback in Description** | Contains `traceback (most recent call last):` | $+2$ |
| **High Discussion Activity** | $\ge 8$ comments or $\ge 5$ commenters ($+2$), $\ge 4$ comments or $\ge 3$ commenters ($+1$) | $+1$ to $+2$ |
| **Blocker Label** | Issue labels contain `blocker` or `critical` | $+4$ |
| **Bug Label** | Issue labels contain `bug` | $+2$ |
| **Docs & Typos** | `docs`, `typo`, `documentation`, `spelling`, `readme`, `tutorial`, label contains `docs` | $-3$ |
| **Questions & Support** | `how to`, `how do i`, `question`, `help needed`, `stackoverflow`, `best practice` | $-3$ |
| **Feature Requests** | `feature request`, `proposal`, `enhancement`, `suggestion`, `would be nice`, label contains `enhancement`/`feature` | $-2$ |
| **Beginner Tasks** | Label contains `good first issue` or `easy` | $-2$ |
| **Cosmetic & Minor** | `minor`, `cosmetic`, `nit`, `cleanup`, `style`, `pep8` | $-2$ |
| **Spam** | Contains `<spam>` or `spam` | $-5$ |

**Actual Code Score-to-Level Thresholds**:
- $\text{Hard-Critical} \text{ or } \text{Score} \ge 7 \rightarrow \textbf{Critical}$
- $\text{Score} \ge 4 \rightarrow \textbf{High}$
- $\text{Score} \ge 1 \rightarrow \textbf{Medium}$
- $\text{Score} < 1 \rightarrow \textbf{Low}$

> **Code Diff & Tuning Verification**:
> We performed a line-by-line diff of `rule_based_priority()` in `priority_prediction.py` against its very first version in the project history. The diff confirmed that **no rules, point weights, triggers, or score thresholds were adjusted after viewing gold results** (the sole modification was defensive null-handling: `str(x) if pd.notna(x) else ""`). All numbers reported reflect zero post-hoc rule tuning.

#### 6.2 Machine Learning Model
- **Training Data**: 2,649 issues (strictly excluding all 130 gold issues). Weak labels generated by the rule scorer: Low: 1,702 (64.25%), Medium: 707 (26.69%), High: 201 (7.59%), Critical: 39 (1.47%).
- **Feature Pipeline**:
  - **Text Features**: `TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)` applied to title-boosted issue text (`f"{title} {title} {description}"`).
  - **Numeric & Structural Features (10 features, StandardScaler)**:
    1. $\log(1 + \text{comments})$
    2. $\log(1 + \text{distinct commenters})$
    3. $\log(1 + \text{description length})$
    4. Traceback presence indicator ($0/1$)
    5. Code block density (count of ```` ``` ```` fences / 2)
    6. `has_bug_label` ($0/1$)
    7. `has_docs_label` ($0/1$)
    8. `has_good_first_issue_label` ($0/1$)
    9. `has_feature_label` ($0/1$)
    10. `has_blocker_label` ($0/1$)
- **Classifier**: `LogisticRegression(class_weight="balanced", C=2.0, max_iter=1000, random_state=42)`.
- **Validation Agreement**: Evaluated on an 80/20 stratified validation split (530 issues) against the weak labels:
  - **Agreement Rate**: **74.72%**
  - **Macro-F1 (Weak Agreement)**: **0.5783**

#### 6.3 Hybrid Consensus Predictor & Agreement Reporting
The final prediction synthesizes the interpretable rules and the statistical model:
1. **Hard-Critical Override**: If a Hard-Critical trigger fires (e.g. CVE, RCE, startup crash), the prediction is immediately locked to `Critical`.
2. **Level Averaging**: Maps the rule level and model probabilities to numeric indices ($\text{Low}=0, \text{Medium}=1, \text{High}=2, \text{Critical}=3$):
   $$\text{Hybrid Value} = \frac{\text{Rule Level} + \sum_{c} \text{Level}(c) \cdot P(c)}{2}$$
3. **Thresholding**:
   - $\text{Hybrid Value} \ge 2.5 \rightarrow \textbf{Critical}$
   - $1.5 \le \text{Hybrid Value} < 2.5 \rightarrow \textbf{High}$
   - $0.5 \le \text{Hybrid Value} < 1.5 \rightarrow \textbf{Medium}$
   - $\text{Hybrid Value} < 0.5 \rightarrow \textbf{Low}$
4. **Transparent Agreement Display**: Rather than presenting raw softmax probabilities as misleading "confidence" percentages, the system reports explicit rule/model consensus:
   - When rule level equals the model's top class: `"rules and model agree"`
   - When they diverge: `"rules and model disagree (rule: X, model: Y)"`
5. **Dataset-Wide Prediction Distribution (2,769 issues)**:
   - Low: 1,681 (60.71%)
   - Medium: 822 (29.69%)
   - High: 228 (8.23%)
   - Critical: 38 (1.37%)

---

### 7. Human-in-the-Loop Gold Set Construction

To evaluate real-world performance honestly, we built a curated Gold Set of **130 issues** with audited labels.

#### 7.1 Sampling Strategy (Metadata-Only Sampling)
The 130 issues were drawn through two complementary sampling rounds:
1. **Original 90 Issues**:
   - **60 Random Issues**: Sampled uniformly at random (`random_state=42`) to establish unbiased majority distribution.
   - **30 Severity-Enriched Issues**: Sampled using keyword triggers to ensure candidate high-severity issues were present for review.
2. **40 Additional Metadata-Sampled Issues**:
   - Added strictly via repository metadata (no rule keywords used):
     - **20 issues with $\ge 8$ comments** (high discussion engagement).
     - **10 issues with labels containing `bug` or `blocker`**.
     - **10 issues created in 2021 or later** (modern Flask 2.x/3.x codebase).

#### 7.2 Strict Priority Audit & Label Drift Log
Each issue was audited by reading its full title, description, and maintainer closing remarks. Across drafts, labels evolved as strict definitions were applied:

- **Draft 1 (90 issues, rule/heuristic scan)**: Low: 44, Medium: 27, High: 15, Critical: 4.
- **Draft 2 (90 issues, strict definition audit)**: Low: 74, Medium: 13, High: 3, Critical: 0 (**34 labels shifted**: 31 downgrades to Low, 3 adjustments).
  - *Why Critical dropped to 0*: Issues like #5021 were dependency bumps for an upstream Werkzeug CVE (already patched in Flask); #953 was tutorial example XSS (not Flask core); #2479 was a single-user Windows Anaconda binary launcher mistake.
- **Draft 3 (130 issues, metadata expansion)**:
  - Re-checked #5448: Maintainer `davidism` confirmed the traceback originated in external `Flask-Session`/`cachelib` under FIPS, not Flask core itself $\rightarrow$ updated from High to Low (**1 label change** on existing 90).
  - Re-checked #3318 (HMAC-SHA1 feature proposal) and #2030 (SameSite RFC proposal) $\rightarrow$ confirmed Low.
  - Added 40 metadata issues: Low: 26, Medium: 11, High: 3, Critical: 0.
- **User Review**: Confirmed without modification (**0 labels changed** by user).
- **Final Gold Distribution (130 issues)**:
  - **Low**: 101 issues (77.69%)
  - **Medium**: 24 issues (18.46%)
  - **High**: 5 issues (3.85%) [#367, #941, #1011, #1427, #2188]
  - **Critical**: 0 issues (0.00%)

---

### 8. Evaluation Results (Real Numbers from Final Run)

All metrics below are copied verbatim from `summarization/results/priority_evaluation.txt` and `summarization/results/summary_evaluation.txt`. Macro-F1 is computed over classes with $n \ge 5$ (dropping Critical from all subsets, as Critical has $n=0$).

#### 8.1 Priority Prediction Evaluation on All 130 Gold Issues

| Method | Overall Accuracy | Within-1-Level Accuracy | MAE (Levels) | Macro-F1 ($n \ge 5$) | Merged High+Critical F1 | Merged High+Critical Raw Counts |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Always-Low Baseline** | **77.69%** | **96.15%** | **0.262** | 0.2915 | 0.0000 | $\text{TP}=0, \text{FP}=0, \text{FN}=5, \text{TN}=125$ |
| **Rule-Based** | 56.15% | 86.15% | 0.623 | **0.4246** | 0.1935 | $\text{TP}=3, \text{FP}=23, \text{FN}=2, \text{TN}=102$ |
| **ML Model (LogReg)** | 52.31% | 84.62% | 0.662 | 0.3777 | 0.1290 | $\text{TP}=2, \text{FP}=24, \text{FN}=3, \text{TN}=101$ |
| **Hybrid Consensus** | 52.31% | 88.46% | 0.623 | 0.4085 | **0.2069** | $\text{TP}=3, \text{FP}=21, \text{FN}=2, \text{TN}=104$ |

*Note: On All Gold Issues ($N=130$), evaluable classes with $n \ge 5$ are Low ($n=101$), Medium ($n=24$), and High ($n=5$). Critical has $n=0$ and is dropped.*

##### Per-Class Performance on All Gold Issues ($N=130$)

| Priority Class | Support ($n$) | Rule-Based Precision / Recall / F1 | ML Model Precision / Recall / F1 | Hybrid Precision / Recall / F1 |
|---|:---:|:---:|:---:|:---:|
| **Low** | 101 | 87.14% / 60.40% / 0.7135 | 90.00% / 53.47% / 0.6708 | 88.52% / 53.47% / 0.6667 |
| **Medium** | 24 | 26.47% / 37.50% / 0.3103 | 29.55% / 54.17% / 0.3824 | 24.44% / 45.83% / 0.3188 |
| **High** | 5 | 15.79% / 60.00% / 0.2500 | 5.00% / 20.00% / 0.0800 | 15.00% / 60.00% / 0.2400 |
| **Critical** | 0 | *not evaluable ($n=0$)* | *not evaluable ($n=0$)* | *not evaluable ($n=0$)* |
| **High+Critical** | 5 | 11.54% / 60.00% / 0.1935 | 7.69% / 40.00% / 0.1290 | 12.50% / 60.00% / 0.2069 |

##### Confusion Matrices ($N=130$)

```
Labels Order: ['Low', 'Medium', 'High', 'Critical']

RULE-BASED CONFUSION MATRIX:
  Low (True)      :   61   24   10    6
  Medium (True)   :    8    9    6    1
  High (True)     :    1    1    3    0
  Critical (True) :    0    0    0    0

ML MODEL CONFUSION MATRIX:
  Low (True)      :   54   28   15    4
  Medium (True)   :    6   13    4    1
  High (True)     :    0    3    1    1
  Critical (True) :    0    0    0    0

HYBRID CONSENSUS CONFUSION MATRIX:
  Low (True)      :   54   32   11    4
  Medium (True)   :    7   11    6    0
  High (True)     :    0    2    3    0
  Critical (True) :    0    0    0    0
```

---

#### 8.2 Priority Prediction Evaluation on Random Subset Only ($N=60$)

The 60 random issues represent the unbiased distribution of real-world GitHub issues.

| Method | Overall Accuracy | Within-1-Level Accuracy | MAE (Levels) | Macro-F1 ($n \ge 5$) | Macro-F1 (3-Class) | Merged High+Critical F1 | Merged High+Critical Raw Counts |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Always-Low Baseline** | **80.00%** | **96.67%** | **0.233** | 0.4444 | 0.2963 | 0.0000 | $\text{TP}=0, \text{FP}=0, \text{FN}=2, \text{TN}=58$ |
| **Rule-Based** | 61.67% | 91.67% | 0.500 | 0.5158 | 0.3439 | 0.0000 | $\text{TP}=0, \text{FP}=7, \text{FN}=2, \text{TN}=51$ |
| **ML Model (LogReg)** | 61.67% | 91.67% | 0.483 | 0.5421 | 0.3614 | 0.0000 | $\text{TP}=0, \text{FP}=7, \text{FN}=2, \text{TN}=51$ |
| **Hybrid Consensus** | 63.33% | 95.00% | 0.433 | **0.5481** | **0.3654** | 0.0000 | $\text{TP}=0, \text{FP}=6, \text{FN}=2, \text{TN}=52$ |

*Note: On Random Subset ($N=60$), evaluable classes with $n \ge 5$ are Low ($n=48$) and Medium ($n=10$). High ($n=2$) and Critical ($n=0$) have $n < 5$. Macro-F1 ($n \ge 5$) averages Low and Medium; Macro-F1 (3-Class) averages Low, Medium, High with Critical dropped.*

##### Per-Class Performance on Random Subset ($N=60$)

| Priority Class | Support ($n$) | Rule-Based Precision / Recall / F1 | ML Model Precision / Recall / F1 | Hybrid Precision / Recall / F1 |
|---|:---:|:---:|:---:|:---:|
| **Low** | 48 | 87.18% / 70.83% / 0.7816 | 89.19% / 68.75% / 0.7765 | 91.89% / 70.83% / 0.8000 |
| **Medium** | 10 | 21.43% / 30.00% / 0.2500 | 25.00% / 40.00% / 0.3077 | 23.53% / 40.00% / 0.2963 |
| **High** | 2 | *not evaluable ($n=2$)* | *not evaluable ($n=2$)* | *not evaluable ($n=2$)* |
| **Critical** | 0 | *not evaluable ($n=0$)* | *not evaluable ($n=0$)* | *not evaluable ($n=0$)* |
| **Merged High+Critical** | 2 | 0.00% / 0.00% / 0.0000 | 0.00% / 0.00% / 0.0000 | 0.00% / 0.00% / 0.0000 |

##### Confusion Matrices ($N=60$)

```
Labels Order: ['Low', 'Medium', 'High', 'Critical']

RULE-BASED CONFUSION MATRIX (RANDOM):
  Low (True)      :   34   10    2    2
  Medium (True)   :    4    3    3    0
  High (True)     :    1    1    0    0
  Critical (True) :    0    0    0    0

ML MODEL CONFUSION MATRIX (RANDOM):
  Low (True)      :   33   10    4    1
  Medium (True)   :    4    4    2    0
  High (True)     :    0    2    0    0
  Critical (True) :    0    0    0    0

HYBRID CONSENSUS CONFUSION MATRIX (RANDOM):
  Low (True)      :   34   11    2    1
  Medium (True)   :    3    4    3    0
  High (True)     :    0    2    0    0
  Critical (True) :    0    0    0    0
```

---

#### 8.3 Issue Summarization Evaluation Results

Evaluated over all **1,679 substantive issues** with $> 40$ words of cleaned prose:

| Metric | Result | Interpretation |
|---|---:|---|
| **Total Issues in Dataset** | 2,769 | Total repository dataset |
| **Substantive Prose Issues ($> 40$ words)** | 1,679 | Eligible for proxy evaluation |
| **Mean Summary Length** | 44.9 words (std: 15.5) | Well within the ~70 word target constraint |
| **Mean Compression Ratio** | 45.1% reduction | Concise distillation without losing context |
| **Title-Keyword Coverage** | 51.7% | Core issue subject matter retained in summary |
| **Tracebacks Encountered** | 312 | Issues with `Traceback (most recent call last)` |
| **Tracebacks with `key_error` Extracted** | 296 (94.9%) | Tightened exception extraction rate |
| **Key Error Correctness (30 Random Audit)** | 30 / 30 (100.0%) | 100% verified exception lines |

##### 30-Traceback Correctness Sample Audit
Across 30 randomly sampled tracebacks (fixed seed 42), the tightened extractor achieved **100% precision**:
- #4299: `werkzeug.routing.WebsocketMismatch: 400 Bad Request: The browser (or proxy) sent a request...`
- #5391: `TypeError: super(type, obj): obj must be an instance or subtype of type`
- #2961: `flask.cli.NoAppException: Could not import "C".`
- #3144: `AssertionError: View function mapping is overwriting an existing endpoint function: register`
- #3294: `TypeError: The view function did not return a valid response...`
- #4023: `ModuleNotFoundError: No module named 'flask._compat'`
- #4408: `FileNotFoundError: [Errno 2] No such file or directory: ...`
- #3340: `TypeError: '<' not supported between instances of 'NoneType' and 'str'`
- #1699: `KeyError: 'WERKZEUG_SERVER_FD'`
- #5774: `ValueError: <Token var=<ContextVar name='flask.request_ctx' at ...> was created in a different Context`

##### ROUGE Metric Implementation
`evaluate_summaries.py` implements pure-Python tokenization, n-gram overlap, and Longest Common Subsequence (LCS) algorithms for **ROUGE-1, ROUGE-2, and ROUGE-L** precision, recall, and F1. The reference evaluation file `data/summary_references.csv` contains 15 sampled issues with an empty `reference_summary` column ready for hand-written reference summaries.

---

### 9. Limitations and Planned Improvements

A transparent and scientifically honest engineering report must document the constraints, statistical variance, and concrete improvement plans of the approach.

#### 9.1 Core System Limitations

1. **Weak Supervision Constraint**:
   - The dataset possesses no native priority ground truth. The ML classifier is trained on weak labels generated by the heuristic rule engine.
   - The ML validation agreement of **74.72%** (macro-F1 0.5783) measures *fidelity to the heuristic rules*, not true human classification accuracy.
2. **AI-Drafted then Human-Audited Gold Set**:
   - Initial gold set labels were drafted using heuristic pattern scanning and subsequently audited by human review of full issue bodies and maintainer comments.
   - Applying strict criteria caused a **34-label drift** between Draft 1 and Draft 2 (31 downgrades from High/Critical to Low), demonstrating how easily keyword-based heuristics over-estimate severity.
   - Re-checking maintainer remarks in Draft 3 changed #5448 from High to Low upon discovering the traceback stemmed from an external extension (`Flask-Session`/`cachelib`).
   - The user review resulted in **0 label changes**, confirming the audited 130-issue distribution.
3. **Information Asymmetry (Maintainer Triage vs Raw Issue Text)**:
   - Gold priority reflects real-world, maintainer-informed outcomes (e.g. maintainers explaining that an apparent crash was user misconfiguration or an external extension bug).
   - In contrast, the rule engine and ML model evaluate *only the initial text submitted by the user*, where users frequently use alarming words (`critical`, `crash`, `broken`) for minor configuration mistakes.
4. **Extreme Class Imbalance & Baseline Reality**:
   - In open-source libraries like Flask, true `Critical` bugs (catastrophic startup failures or zero-day vulnerabilities in core) are exceptionally rare.
   - The 130-issue Gold Set contains **0 Critical issues** and only **5 High issues** (3.85%), with **101 Low issues** (77.69%).
   - Consequently, per-class metrics for Critical are statistically invalid and marked as `not evaluable (n=0)`. On the 60 random issues, High ($n=2$) is also `not evaluable`.
5. **Model Does NOT Beat the Always-Low Baseline**:
   - The model does **NOT** beat the always-Low baseline on overall accuracy (**52.31% vs 77.69%** on all gold issues, and **63.33% vs 80.00%** on random issues).
   - The model also does **NOT** beat the baseline on within-one-level accuracy (**88.46% vs 96.15%** on all gold issues, and **95.00% vs 96.67%** on random issues). Within-one-level accuracy is therefore **not a model strength**; it is simply an artifact of massive Low-class dominance where predicting Low is always within one level of Medium.
   - **The Real Value**: The model's value lies strictly in **High recall: 3/5 (60.0%) at precision 12.50%** on the full gold set (roughly a **3x enrichment** over the 3.85% base rate), whereas the always-Low baseline catches 0/5 (0.0% recall). However, on the 60 random issues, the model detected **0 of 2 High issues** ($TP=0, FP=6$).
6. **Extractive vs Abstractive Summarization**:
   - Extractive TextRank selects verbatim sentences. While this guarantees factual accuracy and preserves technical symbols (`before_request`) without hallucination, it cannot synthesize multi-sentence explanations or rephrase complex technical arguments as fluidly as generative LLMs.

---

#### 9.2 Stratified Dev/Test Split & Score Threshold Optimization

To analyze the sensitivity of the heuristic cutoffs, the 130 gold issues were partitioned 50/50 into a **Dev split (65 issues)** and a **Held-Out Test split (65 issues)**, stratified by gold label (`random_state=42`):
- **Dev Set (N=65)**: 51 Low, 12 Medium, 2 High, 0 Critical.
- **Test Set (N=65)**: 50 Low, 12 Medium, 3 High, 0 Critical.

> **Methodological Disclosure**:
> The full 130-issue gold evaluation results were **seen and analyzed before this retrospective split was conducted**. Therefore, this partitioning serves as an exploratory sensitivity analysis rather than a strictly blind prospective trial. Furthermore, the production pipeline and primary system reports **strictly maintain the original thresholds ($T_{\text{med}}=1, T_{\text{high}}=4$)** as the primary reported baseline.

Score thresholds for Medium ($T_{\text{med}} \in \{1, 2, 3\}$) and High ($T_{\text{high}} \in \{3, 4, 5, 6\}$) were swept on the Dev split:

##### Threshold Sweep Table on Dev Set (65 Issues)

| $T_{\text{med}}$ | $T_{\text{high}}$ | Dev Macro-F1 | High+ Precision | High+ Recall | High+ F1 | TP | FP |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 0.4232 | 7.69% | 50.00% | 0.1333 | 1 | 12 |
| **1** | **4 *(Original Primary)*** | **0.4519** | **11.11%** | **50.00%** | **0.1818** | **1** | **8** |
| 1 | 5 *(Exploratory Pick)* | 0.4973 | 16.67% | 50.00% | 0.2500 | 1 | 5 |
| 1 | 6 | 0.3932 | 0.00% | 0.00% | 0.0000 | 0 | 3 |
| 2 | 3 | 0.3991 | 7.69% | 50.00% | 0.1333 | 1 | 12 |
| 2 | 4 | 0.4337 | 11.11% | 50.00% | 0.1818 | 1 | 8 |
| 2 | 5 | 0.4831 | 16.67% | 50.00% | 0.2500 | 1 | 5 |
| 2 | 6 | 0.3821 | 0.00% | 0.00% | 0.0000 | 0 | 3 |
| 3 | 4 | 0.3941 | 11.11% | 50.00% | 0.1818 | 1 | 8 |
| 3 | 5 | 0.4596 | 16.67% | 50.00% | 0.2500 | 1 | 5 |
| 3 | 6 | 0.3692 | 0.00% | 0.00% | 0.0000 | 0 | 3 |

*Exploratory adjustment chosen on dev*: **$T_{\text{med}}=1, T_{\text{high}}=5$** (Macro-F1: 0.4973). Note that this adjustment was picked on a tiny dev sample containing **only 2 High issues**, making it inherently fragile.

##### Held-Out Test Split Evaluation (65 Issues)

Evaluating the primary original thresholds alongside the exploratory dev-adjusted thresholds on the held-out test split:

| Threshold Setting | Accuracy | Macro-F1 | High+ Precision | High+ Recall | High+ F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| **Original Primary ($T_{\text{med}}=1, T_{\text{high}}=4$)** | **50.77%** | **0.3880** | **11.76%** | **66.67%** | **0.2000** | **2** | **15** | **1** |
| Exploratory Dev Pick ($T_{\text{med}}=1, T_{\text{high}}=5$) | 52.31% | 0.4416 | 16.67% | 66.67% | 0.2667 | 2 | 10 | 1 |

**Analysis**:
- Under the primary production thresholds ($T_{\text{med}}=1, T_{\text{high}}=4$), the rules detect **66.67% of High issues (2 of 3)** on the held-out test split at **11.76% precision** (a 2.5x enrichment over the 4.62% test base rate).
- The exploratory adjustment ($T_{\text{high}}=5$) trimmed 5 false positives (10 vs 15 FP) while retaining the same 2 true positives. However, because this was selected on a dev partition with only 2 High examples, and because 130-issue numbers were seen prior to the split, the original thresholds remain our formal reported baseline.

---

#### 9.3 1000-Resample Bootstrap 95% Confidence Intervals

To quantify estimation uncertainty on small support classes, we executed **1,000 bootstrap resamples** with replacement (`seed=42`) across both the full gold set and the held-out test set:

##### Scope: All 130 Gold Issues

| Method | Accuracy 95% CI | Macro-F1 95% CI | High+ Recall 95% CI | High+ Precision 95% CI |
|---|---|---|---|---|
| Always-Low Baseline | [70.8%, 84.6%] | [0.2763, 0.3056] | [0.0%, 0.0%] | [0.0%, 0.0%] |
| Rule-Based Scorer | [47.7%, 64.6%] | [0.3279, 0.5198] | [0.0%, 100.0%] | [0.0%, 25.0%] |
| ML Model (LogisticReg) | [43.8%, 60.8%] | [0.3004, 0.4664] | [0.0%, 100.0%] | [0.0%, 19.4%] |
| Hybrid Consensus | [43.8%, 61.5%] | [0.3079, 0.5149] | [0.0%, 100.0%] | [0.0%, 26.3%] |

##### Scope: Test Set Only (65 Issues)

| Method | Accuracy 95% CI | Macro-F1 95% CI | High+ Recall 95% CI | High+ Precision 95% CI |
|---|---|---|---|---|
| Always-Low Baseline | [66.2%, 87.7%] | [0.2654, 0.3115] | [0.0%, 0.0%] | [0.0%, 0.0%] |
| Rule-Based Scorer | [38.5%, 63.1%] | [0.2569, 0.5187] | [0.0%, 100.0%] | [0.0%, 29.4%] |
| ML Model (LogisticReg) | [35.4%, 58.5%] | [0.2507, 0.4943] | [0.0%, 100.0%] | [0.0%, 33.3%] |
| Hybrid Consensus | [35.4%, 58.5%] | [0.2392, 0.4925] | [0.0%, 100.0%] | [0.0%, 27.8%] |

**Statistical Insight**:
Because the ground truth contains only 5 High issues overall (and only 3 in the test set), resampling draws samples containing 0, 1, 2, or 3 High instances. This creates an extremely wide recall CI spanning `[0.0%, 100.0%]`. This empirically proves that point estimates for rare classes carry high variance, validating the requirement to state confidence intervals rather than relying on point accuracy alone.

---

#### 9.4 Planned Improvements & Human Evaluation Infrastructure

To address the limitations discovered during development, two concrete human evaluation frameworks and data artifacts were created:

1. **Second-Annotator Submission-Time Agreement Framework (`compute_kappa.py`)**:
   - **Data File**: `data/gold_second_annotator.csv` contains 40 sampled gold issues providing **only the issue title and raw description**. All gold labels and maintainer discussion comments have been strictly removed.
   - **Submission-Time Measurement**: The second-annotator Cohen's kappa specifically measures **submission-time agreement**. Annotators evaluate raw user descriptions without the benefit of maintainer comments, reproduction reports, or closed commit fixes. This directly tests whether two independent human readers interpret the raw problem text identically at the moment of issue creation.
   - **Annotator Instructions**: Provided in `data/annotator_instructions.md`, documenting the exact four priority definitions (Critical, High, Medium, Low) and inclusion/exclusion criteria.
   - **Evaluation Script**: `summarization/compute_kappa.py` computes raw percent agreement, unweighted Cohen's Kappa ($\kappa$), and quadratic-weighted Cohen's Kappa ($\kappa_w$, accounting for ordinal penalties). Whenever simulated test data is evaluated (`--demo`), it explicitly emits a `*** WARNING: DEMO DATA - DO NOT REPORT ***` banner.
2. **Blinded A/B Summary Evaluation Rubric (`summary_ratings.csv`)**:
   - **Nature of Preliminary Ratings (Single-Rater AI Judge)**: Any preliminary evaluation scores reported during iterative development represent automated **"AI judge" results (single rater)**, not definitive human consensus. While automated judges provide rapid relative heuristics, formal validation requires independent human raters.
   - **AI Judge Recognition & Positional Artifacts**: A specific limitation of the AI judge evaluation is that the model may have recognized Lead-3 outputs from characteristic positional cues—such as introductory greetings (`"Hi all,"`, `"Hello,"`), issue template boilerplate, or conversational openers that appear in the first three sentences of raw issue descriptions. This potential recognition introduces a risk of heuristic evaluation bias against Lead-3. Furthermore, because these preliminary ratings are strictly single-rater, inter-rater consensus cannot be confirmed until human annotators score the file.
   - **Differing Pairs Subset & Sampling Pool**: The 30 evaluation pairs cover **only issues where TextRank and Lead-3 strictly differ**. Across the full dataset of 1,679 substantive prose issues (>40 words of cleaned prose), **24.2% (406 of 1,679)** are completely identical between TextRank and Lead-3, while **75.8% (1,273 issues)** differ. In the broader candidate pool of issues with >40 raw description words (2,120 issues), 1,407 differed, which formed the exact candidate pool from which the 30 evaluation issues were randomly sampled. The evaluation dataset was drawn exclusively from differing pairs to ensure evaluators assess genuine algorithmic trade-offs rather than redundant identical outputs.
   - **Truncated Source Context**: Each row includes `source_description_truncated` (the first 800 characters of the raw issue body) so evaluators can verify context directly without inspecting external files.
   - **Blinded A/B Ordering**: For each issue, TextRank and Lead-3 are randomly assigned to **Summary A** and **Summary B** (`seed=42`). The blinding key is stored separately in `data/summary_ratings_key.csv` and never exposed to raters.
   - **Scoring Rubric (1–5 Likert Scale)**:
     - **Faithfulness (1–5)**: Does the summary contain only facts directly supported by the issue body? *Note: Faithfulness is near-trivial for extractive methods* (since candidate sentences are copied verbatim directly from the original issue text, hallucination is virtually 0% unless sentence clipping severs a crucial conditional clause).
     - **Coverage (1–5)**: Does the summary capture the core technical problem, error symptom, and reproduction conditions?
     - **Readability (1–5)**: Is the summary grammatical, cohesive, and free of dangling fragments?
   - **Unblinding Script**: `summarization/evaluate_ratings.py` merges ratings with the secret key, calculating mean criteria scores per method, head-to-head win/tie/loss counts with two-sided sign test p-values, and measuring inter-rater agreement strictly on dual-rated issues.
   - **Summary Length & Length-Adjusted Analysis (OLS)**: Across the 30 rated issues in `summary_ratings.csv`, TextRank summaries average **41.77 words** (std: 15.15) and Lead-3 summaries average **38.40 words** (std: 15.99), a difference of **+3.37 words**. Length difference is positively associated with rating differences (Coverage: Pearson $r = 0.388, p = 0.0342$, Spearman $\rho = 0.441, p = 0.0146$; Readability: Pearson $r = 0.457, p = 0.0111$, Spearman $\rho = 0.521, p = 0.0032$). In an ordinary least squares (OLS) regression of rating difference $(\text{TR} - \text{L3})$ on word difference $(\text{TR} - \text{L3})$ across $n = 30$ pairs:
     - **Coverage**: Slope $\beta = +0.0429$ per word ($p = 0.0342$); Intercept $\alpha$ (**length-adjusted advantage**) = **$+0.5554$ points** (95% CI: $[+0.0324, +1.0785]$, $p = 0.0382$).
     - **Readability**: Slope $\beta = +0.0316$ per word ($p = 0.0111$); Intercept $\alpha$ (**length-adjusted advantage**) = **$+0.3936$ points** (95% CI: $[+0.0786, +0.7087]$, $p = 0.0162$).
     *Methodological note*: The positive intercepts indicate that TextRank holds a statistically significant score advantage even when holding length constant at zero difference; however, this reflects an empirical association within single-rater AI evaluations, not a proven causal mechanism.
3. **Generative LLM Distillation**: In production environments where lightweight CPU inference is not mandated, fine-tuning an open-weight small language model (e.g. Qwen2.5-Coder-1.5B or Phi-3.5) on developer issue summaries could bridge the gap between extractive sentence selection and fluent multi-sentence technical synthesis.

---

### 10. Integration Architecture for Member 5 (`analyze_issue.py`)

Member 5 builds the FastAPI backend and Streamlit interface. `analyze_issue.py` provides a unified, single-call integration entry point that bundles summarization, metadata extraction, priority prediction, and Member 2's category classification:

```python
from summarization.analyze_issue import analyze_issue

result = analyze_issue(
    title="AttributeError: 'NoneType' object has no attribute 'route'",
    description="When initializing the blueprint in app.py, it crashes on startup with Traceback ...",
    labels="bug",
    comments=None
)
```

#### Output Payload Schema

```json
{
  "title": "AttributeError: 'NoneType' object has no attribute 'route'",
  "summary": "When initializing the blueprint in app.py, it crashes on startup with Traceback...",
  "key_error": "AttributeError: 'NoneType' object has no attribute 'route'",
  "environment": {"python": "3.10", "flask": "2.2.2"},
  "discussion_summary": null,
  "summary_method": "textrank",
  "compression": 55.3,
  "priority": "Medium",
  "priority_agreement": "rules and model agree",
  "priority_reasons": [
    "Crash or fatal error signal (+3)",
    "Bug label present (+2)",
    "Hybrid consensus: rule level 'Medium' (1) + model expected (0.85) -> Medium"
  ],
  "priority_method": "Hybrid (Rules + Model)",
  "priority_score": 0.92,
  "priority_probabilities": {"Low": 0.32, "Medium": 0.54, "High": 0.12, "Critical": 0.02},
  "category": "Bug",
  "suggested_label": "bug",
  "classification_method": "TF-IDF + LogisticRegression"
}
```

- **Transparent Agreement Output**: Downstream UI components display `priority_agreement` (`"rules and model agree"` or `"rules and model disagree (rule: X, model: Y)"`). The raw softmax probabilities (`priority_probabilities`) are retained in the payload for custom thresholding, but are not displayed as "confidence" percentages.
- **Decoupled Team Integration**: Member 2's `classification.issue_classification.classify_issue` is imported inside a `try/except` block. If Member 2's model is absent, `category` and `suggested_label` gracefully default to `None` without failing.
- **CLI Utility**: Supports command-line testing via `python analyze_issue.py --title "..." --desc "..." [--json]`.

---

### 11. Project File Manifest

All Member 4 code and evaluation artifacts are stored in `summarization/`:

| File Path | Description |
|---|---|
| `summarization/summarizer.py` | Markdown cleaning, sentence splitting, TextRank extractive summarizer, `key_error`, `environment`, and `discussion_summary` extraction |
| `summarization/priority_prediction.py` | Heuristic rule engine, ML training pipeline, hybrid predictor, batch prediction runner, and gold evaluation runner with dev/test threshold tuning and bootstrap CIs |
| `summarization/evaluate_summaries.py` | Proxy quality metrics evaluator (>40 words), tightened traceback extractor evaluator, and pure-Python ROUGE-1/2/L engine |
| `summarization/evaluate_ratings.py` | Unblinded evaluator for human A/B summary ratings (TextRank vs lead-3) with inter-rater correlation |
| `summarization/compute_kappa.py` | Standalone script computing Cohen's Kappa ($\kappa$) inter-annotator agreement and confusion matrix |
| `summarization/analyze_issue.py` | Unified integration module and CLI for Member 5 combining summarization, priority, and category |
| `summarization/README.md` | Architecture guide, file list, run commands, and Member 5 integration instructions |
| `summarization/review_gold.md` | Full 130-issue gold audit table with drafts, rationales, and confidence levels (LOW-confidence first) |
| `summarization/models/priority_model.pkl` | Pickled model bundle (TF-IDF vectorizer, StandardScaler, LogisticRegression, metadata) |
| `summarization/data/priority_gold.csv` | Ground truth evaluation dataset with 130 hand-annotated issues |
| `summarization/data/gold_second_annotator.csv` | 40 gold issues stripped of labels and comments for double-blind second teammate review |
| `summarization/data/annotator_instructions.md` | Detailed annotation guidelines and definitions (Critical, High, Medium, Low) for second annotator |
| `summarization/data/summary_references.csv` | 15 sampled issues formatted with empty reference columns for manual ROUGE benchmarking |
| `summarization/data/summary_ratings.csv` | Blinded A/B summary ratings file (30 issues with differing summaries and 800-char source descriptions) with empty 1–5 Likert columns for two raters |
| `summarization/data/summary_ratings_key.csv` | Secret blinding key mapping Summary A and Summary B to TextRank / lead-3 |
| `summarization/results/issue_priorities.csv` | Priority predictions for all 2,769 dataset issues |
| `summarization/results/priority_evaluation.txt` | Complete evaluation report comparing Rules, ML, and Hybrid against baseline, with dev/test threshold sweep and bootstrap CIs |
| `summarization/results/summary_evaluation.txt` | Proxy metrics and traceback extraction evaluation report |
| `member4_doc.md` | Comprehensive repo-root technical report matching Member 1's documentation style |

---

### 12. Conclusion

As Member 4, I designed and implemented the **Issue Summarization and Priority Prediction** subsystem for the GitHub Issue Intelligence platform.

The module deliverables and findings are:
1. **Summarization & Context Extraction**: Distills complex, multi-paragraph GitHub issues into clean, ~45-word summaries with 45.1% mean compression, while isolating traceback exceptions with **94.9% dataset coverage** and **100% verified precision** on sample audit.
2. **Honest Priority Prediction Assessment**:
   - The model does **not** outperform an always-Low majority baseline on overall accuracy (52.31% vs 77.69%) or within-one-level accuracy (88.46% vs 96.15%).
   - The operational utility is concentrated in **High-priority recall (3/5 detected on the full gold set at 12.50% precision, approximately 3x base rate enrichment)**, whereas the baseline catches 0/5.
   - On the random 60-issue sample, 0 of 2 High issues were detected. Critical priority has 0 examples in the gold set and is mathematically un-evaluable.
3. **Decoupled Architecture**: Exposes `analyze_issue.py` with transparent rule/model agreement reporting, ready for immediate consumption by Member 5's FastAPI backend and Streamlit dashboard.
