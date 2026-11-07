"""
GitHub Issue Summarizer (Extractive TextRank + Metadata Extraction)
Member 4: Summarization + Priority Prediction

Provides extractive summarization using TF-IDF cosine graph + PageRank,
sentence weighting bonuses (title similarity, early position, cue words),
run-on sentence penalties, and cleans GitHub markdown while preserving
snake_case identifiers (e.g., before_request).

Also extracts:
- key_error: last exception line from Python tracebacks
- environment: Python, Flask, Werkzeug versions
- discussion_summary: 2-sentence summary of comments with recency bias
"""

import os
import re
import math
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cue words indicating important issue content
CUE_WORDS = {
    "expected", "actual", "instead", "regression", "crash", "crashes", "crashing",
    "fail", "fails", "failed", "failing", "failure", "error", "errors",
    "exception", "exceptions", "traceback", "broken", "cannot", "can't",
    "problem", "bug", "issue", "reproduce", "reproduced", "behavior", "cause",
    "raises", "raised", "unhandled", "segfault", "freeze", "hang", "hangs",
    "steps", "reproduction", "unexpected", "wrong", "fix", "fixed", "patch"
}

DISCUSSION_CUE_WORDS = {
    "fixed", "resolved", "merged", "closing", "closed", "workaround", "solution",
    "pr", "pull", "commit", "released", "version", "cause", "issue is", "tested",
    "confirm", "reproduced", "update", "patch", "works", "reason"
}

DROP_COLON_PATTERNS = [
    r"^(?:here\s+is|here\'s|below\s+is|see|the\s+following|this\s+is)(?:\s+(?:the|some|a|an))?.*$",
    r"^(?:steps?\s+to\s+reproduce|to\s+reproduce|minimal\s+(?:repro|example|code)|sample\s+code|test\s+code|reproduction(?:\s+code)?)$",
    r"^(?:replication|reproduce\s+with|run|output|traceback|error|exception|result|results|expected|actual|behavior).*$",
    r"^(?:code|example|sample|snippet|log|logs|terminal|console)$",
    r"^(?:got|like|should\s+be|fix\s+branch|comparison|for\s+example|note|call|the\s+call\s+raises|it\s+raises|raises|fails\s+with)$",
    r"^[a-zA-Z0-9_./\-]+\.(?:py|rst|md|html|js|json|cfg|txt|sh)$",
    r"^(?:def\s+[a-zA-Z0-9_]+\(.*\)|class\s+[a-zA-Z0-9_]+.*)$",
    r"^(?:in|from)\s+`?[a-zA-Z0-9_]+`?\s*(?:\(.*\))?$",
    r"^[-*]?\s*(?:python|flask|werkzeug|os|environment|commit)(?:\s+version)?.*$"
]

def should_drop_colon_sentence(s):
    """Check if a sentence ending in a colon adds no substantive information."""
    clean_s = re.sub(r"[*_`#]", "", s).strip().lower()
    clean_s = clean_s.rstrip(":").strip()
    words = clean_s.split()
    if len(words) <= 2:
        return True
    for pat in DROP_COLON_PATTERNS:
        if re.search(pat, clean_s, re.IGNORECASE):
            return True
    return False


# ============================================================
# MARKDOWN & TEXT CLEANING
# ============================================================
def clean_markdown(text):
    """Clean GitHub issue markdown for summarization.
    
    Removes HTML comments, code fences, images, template headings,
    environment blocks, greetings, and closings.
    Preserves snake_case identifiers like before_request.
    Sentences ending in ':' that precede a removed code block have
    ' (code omitted)' appended or are dropped if they add nothing.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Standardize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # 1. Remove HTML comments <!-- ... -->
    text = re.sub(r"<!--[\s\S]*?-->", "", text)
    
    # 2. Remove standard issue template preambles
    text = re.sub(
        r"\*\*This issue tracker is a tool to address bugs in Flask itself[\s\S]*?(?:fill out the template below[^\n]*\n|\Z)",
        "", text, flags=re.IGNORECASE
    )
    
    # 3. Remove markdown images: ![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    
    # 4. Handle sentences ending in ":" that precede code blocks or tracebacks
    def _sub_fenced_colon(m):
        pre = m.group(1).strip()
        if should_drop_colon_sentence(pre):
            return "\n"
        return pre.rstrip(":").strip() + " (code omitted).\n"
        
    text = re.sub(
        r"(?m)(^[^\n]+:)[ \t]*\n+[ \t]*(?:```[^\n]*\n[\s\S]*?```|~~~[^\n]*\n[\s\S]*?~~~)",
        _sub_fenced_colon,
        text
    )
    def _sub_tb_colon(m):
        pre = m.group(1).strip()
        if should_drop_colon_sentence(pre):
            return "\n" + m.group(2)
        return pre.rstrip(":").strip() + " (code omitted).\n" + m.group(2)
        
    text = re.sub(
        r"(?m)(^[^\n]+:)[ \t]*\n+[ \t]*(Traceback \(most recent call last\):?)",
        _sub_tb_colon,
        text
    )

    # 5. Remove remaining fenced code blocks (```...``` and ~~~...~~~) with any language tag / spacing
    text = re.sub(r"```[^\n]*\n[\s\S]*?```", "", text)
    text = re.sub(r"~~~[^\n]*\n[\s\S]*?~~~", "", text)
    
    # 6. Remove unfenced tracebacks from prose (including indented tracebacks)
    text = re.sub(
        r"(?m)^\s*Traceback \(most recent call last\):?[\s\S]*?(?=(?:^\s*[A-Z][a-zA-Z0-9_]*(?:Error|Exception|Warning|Interrupt|StopIteration):[^\n]*|\Z))",
        "", text
    )
    text = re.sub(
        r"(?m)^\s*[A-Z][a-zA-Z0-9_]*(?:Error|Exception|Warning|Interrupt|StopIteration):[^\n]*$",
        "", text
    )
    text = re.sub(r'(?m)^\s*File "[^"]+", line \d+.*$', "", text)
    text = re.sub(r"(?m)^\s*Traceback \(most recent call last\):?\s*$", "", text)
    
    # 7. Remove environment lines from prose
    text = re.sub(
        r"(?m)^\s*[-*]?\s*(?:Python|Flask|Werkzeug|OS|Environment(?:\s+details)?|Commit).*?:\s*.*$",
        "", text, flags=re.IGNORECASE
    )
    
    # 8. Remove issue template headings
    heading_pattern = (
        r"(?m)^\s*(?:#{1,6}|\*{1,2})?\s*"
        r"(?:Expected [Bb]ehavior|Actual [Bb]ehavior|Steps [Tt]o [Rr]eproduce|"
        r"Environment(?:\s+details)?|Description|Context|Problem|Summary|"
        r"Reproduction|Minimal Reproduction|Versions?|Control cases|Additional context)[:\s*]*$"
    )
    text = re.sub(heading_pattern, "", text, flags=re.IGNORECASE)
    
    # 8. Markdown links [text](url) -> text, and strip standalone URLs
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    
    # 9. Strip inline code backticks `code` -> code (keeps snake_case intact)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    
    # 10. Remove HTML tags like <details>, </summary>, <p>, etc.
    text = re.sub(r"<[^>]+>", " ", text)
    
    # 11. Remove greetings and pleasantries
    text = re.sub(
        r"(?m)^\s*(?:Hi|Hello|Hey|Greetings|Dears?)(?:\s+(?:all|there|team|everyone|guys))?[,\.!]?\s*$",
        "", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"^\s*(?:Hi|Hello|Hey|Greetings|Dears?)(?:\s+(?:all|there|team|everyone|guys))?[,\.!]?\s*",
        "", text, flags=re.IGNORECASE
    )
    text = re.sub(r"^\s*Hope all is well[,\.!]?\s*", "", text, flags=re.IGNORECASE)
    
    # 12. Remove common closings
    text = re.sub(
        r"\n\s*(?:Thanks(?: in advance| a lot)?|Thank you|Cheers|Best regards|Regards|Sincerely)[,\.!]?\s*(?:\n[A-Za-z0-9_\- ]{1,30})?\s*$",
        "", text, flags=re.IGNORECASE
    )
    
    # 13. Normalize whitespace while preserving line structure
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


# ============================================================
# SENTENCE SPLITTING WITHOUT NLTK
# ============================================================
def split_sentences(text):
    """Split text into sentences without external libraries.
    
    Protects:
    - Abbreviations: e.g., i.e., etc., vs.
    - Version numbers: 3.10.4, 1.0
    - Dotted identifiers: app.route, request.args
    - Underscored identifiers: before_request
    """
    if not text:
        return []
    
    # Protect patterns containing dots
    text_sub = re.sub(
        r"\b(e\.g\.|i\.e\.|etc\.|vs\.|fig\.|ref\.)",
        lambda m: m.group(0).replace(".", "<DOT>"),
        text,
        flags=re.IGNORECASE
    )
    # Protect versions (e.g., 3.11.2 or 2.0)
    text_sub = re.sub(r"(\d+)\.(\d+)", r"\1<DOT>\2", text_sub)
    # Protect dotted identifiers (e.g. app.route)
    text_sub = re.sub(r"([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)", r"\1<DOT>\2", text_sub)
    
    # Split on sentence terminators followed by whitespace and capital/quote, or newlines
    raw_sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])|\n+', text_sub)
    
    sentences = []
    for s in raw_sents:
        # Restore dots
        s = s.replace("<DOT>", ".").strip()
        # Clean stray list bullets or numbers
        s = re.sub(r"^[-*•\d\.\)]+\s*", "", s).strip()
        # Guard against dangling colons: if a sentence ends in ":" or ":."
        if s.endswith(":."):
            s = s[:-2] + ":"
        if s.endswith(":"):
            if should_drop_colon_sentence(s):
                continue
            s = s.rstrip(":").strip() + " (code omitted)."

        # Only keep substantive sentences (exclude pure traceback headers or code markers)
        if len(s) > 10 and re.search(r"[a-zA-Z]", s):
            if re.match(r"^Traceback \(most recent call last\)", s, re.IGNORECASE):
                continue
            # Ensure ending punctuation
            if not s.endswith((".", "!", "?")):
                s += "."
            sentences.append(s)
            
    return sentences


# ============================================================
# METADATA EXTRACTION: KEY ERROR & ENVIRONMENT
# ============================================================
_EXC_REGEX = re.compile(
    r"^((?:[a-zA-Z_][a-zA-Z0-9_.]*\.)?(?:[a-zA-Z0-9_]*(?:[Ee]rror|[Ee]xception|Warning|Interrupt|StopIteration|SystemExit|NotFound|BadRequest|Forbidden|MethodNotAllowed|InternalServerError|Unauthorized|Mismatch|Fault)))\b(?::\s*.*)?$"
)
_PASCAL_EXC_REGEX = re.compile(
    r"^([a-zA-Z_][a-zA-Z0-9_.]*(?:[A-Z][a-z0-9]+)+(?:[A-Z][a-z0-9]*)*)\b(?::\s*.*)$"
)
_NON_EXCEPTIONS = {
    "During", "The", "File", "Traceback", "Expected", "Actual", "Steps", "Step",
    "Environment", "Version", "Note", "Output", "Description", "Python", "Flask"
}

def _parse_exception_line(raw_line):
    line = raw_line.strip()
    while line.startswith('>'):
        line = line[1:].strip()
    line = line.strip("`* ")
    # Optional docker container prefix
    line = re.sub(r"^[a-zA-Z0-9_\-\.]+\s+\|\s+", "", line).strip("`* ")
    if not line:
        return None
    # Reject lines containing braces
    if "{" in line or "}" in line:
        return None
    # Reject lines starting with timestamp, dates, or numbers
    if re.match(r"^\[?\d{1,4}[-/:.]\d{1,4}", line):
        return None
    if re.match(r"^\d+\.", line) or line.startswith(('-', '*', '+', '#')):
        return None
    if line.startswith(("During handling of the above exception", "The above exception was the direct cause", "Traceback", "File ", "line ", "~~~", "^^^")):
        return None

    m = _EXC_REGEX.match(line)
    if m:
        cls_name = m.group(1).split('.')[-1]
        if cls_name in ("The", "During", "File", "Traceback"):
            return None
        if cls_name in ("Error", "Exception") and not line.startswith(f"{cls_name}:"):
            return None
        return line
        
    m2 = _PASCAL_EXC_REGEX.match(line)
    if m2:
        cls_name = m2.group(1).split('.')[-1]
        if cls_name in _NON_EXCEPTIONS:
            return None
        return line
        
    return None


def extract_key_error(text):
    """Extract the final exception line from a Python traceback.
    
    Accepts only lines where an exception class appears at the start of the line,
    optionally followed by ': message'. Rejects lines starting with a timestamp
    or containing braces.
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # 1. Search after traceback headers
    tb_matches = list(re.finditer(r"Traceback \(most recent call last\):", text, re.IGNORECASE))
    if tb_matches:
        tb_text = text[tb_matches[-1].start():]
        lines = tb_text.split('\n')
        
        seen_file = False
        candidates = []
        for l in lines:
            stripped = l.strip()
            if 'File "' in stripped or "File '" in stripped or 'File `' in stripped:
                seen_file = True
                continue
            if seen_file:
                if stripped.startswith(('```', '##', '---', '>>>', '$')):
                    if candidates:
                        break
                    continue
                cand = _parse_exception_line(l)
                if cand:
                    candidates.append(cand)
        if candidates:
            return candidates[-1]
            
        # Scan tb_text lines in reverse
        for l in reversed(lines):
            cand = _parse_exception_line(l)
            if cand:
                return cand
                
    # 2. Standalone fallback across full text
    for l in reversed(text.split('\n')):
        cand = _parse_exception_line(l)
        if cand:
            return cand
            
    return None


def extract_environment(text):
    """Extract Python, Flask, and Werkzeug versions from issue text."""
    if not text or not isinstance(text, str):
        return {}
    
    env = {}
    
    # Python version
    py_m = re.search(
        r"(?:Python|Python version)[:\s]+`?([0-9]+\.[0-9]+(?:\.[0-9a-zA-Z]+)?)`?",
        text, re.IGNORECASE
    )
    if py_m:
        env["python"] = py_m.group(1).strip(" .`*")
        
    # Flask version
    flask_m = re.search(
        r"(?:Flask|Flask version)[:\s]+`?([0-9]+\.[0-9]+(?:\.[0-9a-zA-Z._]+)?)`?",
        text, re.IGNORECASE
    )
    if flask_m:
        env["flask"] = flask_m.group(1).strip(" .`*")
        
    # Werkzeug version
    wz_m = re.search(
        r"(?:Werkzeug|Werkzeug version)[:\s]+`?([0-9]+\.[0-9]+(?:\.[0-9a-zA-Z._]+)?)`?",
        text, re.IGNORECASE
    )
    if wz_m:
        env["werkzeug"] = wz_m.group(1).strip(" .`*")
        
    return env


# ============================================================
# DISCUSSION SUMMARY FROM COMMENTS
# ============================================================
def extract_discussion_summary(comments):
    """Generate a 2-sentence summary of discussion comments with recency bias."""
    if not comments:
        return None
    
    # Convert input comments into a list of cleaned strings
    comment_list = []
    if isinstance(comments, list):
        for c in comments:
            if isinstance(c, dict):
                body = c.get("comment_body", "") or c.get("body", "")
            else:
                body = str(c)
            cleaned = clean_markdown(body)
            if cleaned:
                comment_list.append(cleaned)
    elif isinstance(comments, str):
        cleaned = clean_markdown(comments)
        if cleaned:
            comment_list.append(cleaned)
            
    if not comment_list:
        return None
    
    # Collect sentences with recency weighting
    n_comments = len(comment_list)
    candidate_sentences = []
    sentence_weights = []
    
    for i, comm in enumerate(comment_list):
        recency_weight = 1.0 + 0.6 * (i / max(1, n_comments - 1))
        sents = split_sentences(comm)
        for s in sents:
            words = set(s.lower().split())
            cue_bonus = 1.0
            if words.intersection(DISCUSSION_CUE_WORDS):
                cue_bonus = 1.4
            candidate_sentences.append(s)
            sentence_weights.append(recency_weight * cue_bonus)
            
    if not candidate_sentences:
        return None
    
    if len(candidate_sentences) <= 2:
        return " ".join(candidate_sentences)
    
    # TF-IDF scoring across comment sentences
    try:
        vec = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z0-9_]+\b")
        tfidf = vec.fit_transform(candidate_sentences)
        sim = cosine_similarity(tfidf, tfidf)
        np.fill_diagonal(sim, 0.0)
        pr = pagerank(sim)
        
        scores = []
        for idx, (p, w) in enumerate(zip(pr, sentence_weights)):
            # Penalize run-ons
            word_len = len(candidate_sentences[idx].split())
            run_on = 1.0 - min(0.5, 0.02 * max(0, word_len - 30))
            scores.append(p * w * run_on)
            
        top_indices = sorted(range(len(candidate_sentences)), key=lambda k: scores[k], reverse=True)[:2]
        selected = [candidate_sentences[idx] for idx in sorted(top_indices)]
        return " ".join(selected)
    except Exception:
        # Fallback to the latest two sentences
        return " ".join(candidate_sentences[-2:])


# ============================================================
# PAGERANK & SENTENCE RANKING
# ============================================================
def pagerank(similarity_matrix, d=0.85, max_iter=50, tol=1e-4):
    """Power iteration PageRank algorithm."""
    n = similarity_matrix.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    
    row_sums = similarity_matrix.sum(axis=1)
    M = np.zeros_like(similarity_matrix)
    for i in range(n):
        if row_sums[i] > 0:
            M[i] = similarity_matrix[i] / row_sums[i]
        else:
            M[i] = np.ones(n) / n
            
    p = np.ones(n) / n
    for _ in range(max_iter):
        next_p = (1 - d) / n + d * M.T.dot(p)
        if np.linalg.norm(next_p - p, 1) < tol:
            break
        p = next_p
    return p


def rank_sentences(sentences, title=""):
    """Rank sentences using TextRank, title similarity, position, and cue words."""
    n = len(sentences)
    if n <= 3:
        return list(range(n))
    
    docs = [title] + sentences
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z0-9_]+\b"
        )
        tfidf = vectorizer.fit_transform(docs)
    except ValueError:
        return list(range(min(3, n)))
        
    title_vec = tfidf[0]
    sent_vecs = tfidf[1:]
    
    # Sentence similarity graph
    sim_matrix = cosine_similarity(sent_vecs, sent_vecs)
    np.fill_diagonal(sim_matrix, 0.0)
    
    pr = pagerank(sim_matrix)
    max_pr = np.max(pr) if len(pr) > 0 and np.max(pr) > 0 else 1.0
    
    title_sims = cosine_similarity(sent_vecs, title_vec).flatten()
    
    scores = []
    for i, s in enumerate(sentences):
        words = s.split()
        word_count = len(words)
        
        # Base PageRank
        score = pr[i]
        
        # Title similarity bonus
        score += 0.35 * max_pr * float(title_sims[i])
        
        # Early position bonus (diminishing with position)
        pos_factor = 1.0 / (1.0 + 0.25 * i)
        score += 0.25 * max_pr * pos_factor
        
        # Cue word bonus
        lower_words = {w.lower().strip('.,!?":;()[]{}') for w in words}
        num_cues = len(lower_words.intersection(CUE_WORDS))
        if num_cues > 0:
            score += 0.20 * max_pr * min(3, num_cues)
            
        # Run-on penalty: penalize sentences over 35 words
        if word_count > 35:
            penalty = 1.0 - min(0.6, 0.02 * (word_count - 35))
            score *= penalty
            
        scores.append(score)
        
    # Top 3 sentences
    top_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)[:3]
    return sorted(top_indices)


# ============================================================
# PUBLIC API: SUMMARIZE ISSUE
# ============================================================
def summarize_issue(title, description, comments=None):
    """Summarize a GitHub issue and extract relevant technical context.
    
    Parameters:
    -----------
    title : str
        The issue title.
    description : str
        The issue body/description markdown text.
    comments : list or str, optional
        Associated issue comments or discussion.
        
    Returns:
    --------
    dict with keys:
        summary : str
            The extractive summary (up to 3 sentences, ~70 words max).
        key_error : str or None
            Extracted exception line from tracebacks.
        environment : dict
            Versions found (python, flask, werkzeug).
        discussion_summary : str or None
            2-sentence summary of discussion from comments.
        original_words : int
            Word count of cleaned prose.
        summary_words : int
            Word count of summary.
        compression : float
            Proportion of words reduced (0.0 to 1.0).
        method : str
            Summarization method ('extractive-textrank', 'short-prose', 'title-fallback').
    """
    title = (title or "").strip()
    raw_desc = (description or "").strip()
    
    # 1. Extract metadata before stripping code blocks
    key_error = extract_key_error(raw_desc)
    environment = extract_environment(raw_desc)
    discussion_summary = extract_discussion_summary(comments)
    
    # 2. Handle empty description -> fallback to title
    if not raw_desc:
        summary = title if title.endswith((".", "!", "?")) else (title + "." if title else "")
        s_words = len(summary.split())
        return {
            "summary": summary,
            "key_error": key_error,
            "environment": environment,
            "discussion_summary": discussion_summary,
            "original_words": len(title.split()),
            "summary_words": s_words,
            "compression": 0.0,
            "method": "title-fallback",
        }
        
    # 3. Clean markdown prose
    cleaned = clean_markdown(raw_desc)
    prose_words = cleaned.split()
    orig_words = len(prose_words)
    
    if orig_words == 0:
        summary = title if title.endswith((".", "!", "?")) else (title + "." if title else "")
        s_words = len(summary.split())
        return {
            "summary": summary,
            "key_error": key_error,
            "environment": environment,
            "discussion_summary": discussion_summary,
            "original_words": len(title.split()),
            "summary_words": s_words,
            "compression": 0.0,
            "method": "title-fallback",
        }
        
    # 4. Short prose (<= 40 words) is kept as-is
    if orig_words <= 40:
        sents = split_sentences(cleaned)
        summary = " ".join(sents)
        if not summary.endswith((".", "!", "?")):
            summary += "."
        s_words = len(summary.split())
        return {
            "summary": summary,
            "key_error": key_error,
            "environment": environment,
            "discussion_summary": discussion_summary,
            "original_words": orig_words,
            "summary_words": s_words,
            "compression": 0.0,
            "method": "short-prose",
        }
        
    # 5. TextRank summarization for longer prose
    sents = split_sentences(cleaned)
    if not sents:
        summary = title if title.endswith((".", "!", "?")) else (title + "." if title else "")
        s_words = len(summary.split())
        return {
            "summary": summary,
            "key_error": key_error,
            "environment": environment,
            "discussion_summary": discussion_summary,
            "original_words": orig_words,
            "summary_words": s_words,
            "compression": 0.0,
            "method": "title-fallback",
        }
        
    top_indices = rank_sentences(sents, title)
    
    # Word budget ~70 words max
    selected = []
    curr_words = 0
    for idx in top_indices:
        s = sents[idx]
        w_len = len(s.split())
        if selected and (curr_words + w_len > 75):
            continue
        selected.append(s)
        curr_words += w_len
        
    if not selected:
        selected = [sents[top_indices[0]]]
        
    summary = " ".join(selected)
    s_words = len(summary.split())
    
    compression = round(1.0 - (s_words / max(1, orig_words)), 3)
    if compression < 0:
        compression = 0.0
        
    return {
        "summary": summary,
        "key_error": key_error,
        "environment": environment,
        "discussion_summary": discussion_summary,
        "original_words": orig_words,
        "summary_words": s_words,
        "compression": compression,
        "method": "extractive-textrank",
    }


if __name__ == "__main__":
    # Test on a representative issue
    demo_title = "Flask test_client retaining lock on files, preventing os.remove() on Windows"
    demo_desc = """
    ### Expected Behavior
    Flask app test_client should release file handle after request finishes.
    
    ```python
    import os
    from app import app
    test_client = app.test_client()
    ```
    
    ### Actual Behavior
    When calling `before_request` in `app.route`, `os.remove()` fails with `PermissionError`.
    
    Traceback (most recent call last):
      File "test.py", line 12, in test_file
        os.remove(filename)
    PermissionError: [WinError 32] The process cannot access the file because it is being used by another process.
    
    Environment:
    - Python version: 3.10.2
    - Flask version: 2.2.0
    """
    demo_comments = [
        "I was able to reproduce this on Windows 10 with Python 3.10.",
        "Fixed in Werkzeug by properly closing the response file descriptor in teardown."
    ]
    
    result = summarize_issue(demo_title, demo_desc, demo_comments)
    print("DEMO SUMMARIZATION RESULT:")
    for k, v in result.items():
        print(f"  {k}: {v}")
