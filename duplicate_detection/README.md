# Duplicate Detection

This module ranks likely duplicate GitHub issues within the same repository.
It preserves the original issue fields in the candidate CSV so reviewers can
compare the evidence.

## Member 3 work summary

Member 3 owns duplicate detection and semantic search. The work in this module
covers:

- Preparing issue text and filtering unusable candidates, including placeholder
  descriptions, spam-only titles, generated coverage reports, and rows without
  issue or repository IDs.
- Generating 384-dimensional embeddings with the existing
  `all-MiniLM-L6-v2` Sentence Transformer and comparing them with cosine
  similarity.
- Generating same-repository, distinct-issue duplicate candidates with a
  configurable similarity threshold, deterministic ordering, and CSV output
  that preserves the source issue details.
- Providing `search_related_issues()` for query-to-issue semantic retrieval.
  It ranks eligible issues, returns up to Top-K results with similarity
  scores, and can exclude a query issue by ID.
- Providing a manual validation queue that leaves labels blank until a person
  reviews them, and threshold and Precision@k evaluators that use explicitly
  verified labels only.
- Updating Precision@k evaluation so an unreviewed result remains in its
  actual ranked position. A query is not scored if any result in its Top-K
  lacks a verified judgment; the evaluator does not skip unreviewed candidates
  or invent labels.

The current `results/duplicate_candidates.csv` contains **483 logical
candidate records**. This is a candidate count, not a count of confirmed
duplicates. The manual review queue currently has no verified relevance
labels, so model-quality evaluation is not complete and no evaluation score
or threshold recommendation is available.

### Example semantic search run

The existing semantic search was run using issue #6175 as its query, with
`exclude_issue_id` set to the query issue ID. Its title is
`[rejected AI] Error handling docs examples use request.arg instead of request.args`.
Its description reports errors in Flask error-handling documentation examples:
an unnecessary `user_id` view argument and use of the nonexistent
`request.arg` attribute instead of `request.args`.

The five retrieved results were:

| Issue | Title | Cosine similarity |
|---:|---|---:|
| #338 | TypeError in "Method Views for APIs" | 0.737631 |
| #353 | MethodViews for API example is missing parameter | 0.716796 |
| #45 | Mistake in docstring of app.errorhandler | 0.690175 |
| #4409 | Typo/Bug in make_response()? | 0.684443 |
| #644 | user could customs non-OK entry body | 0.672413 |

These are retrieval similarity scores only. They are not human relevance
judgments, Precision@k, or evidence that a result is a duplicate.

## Algorithm

1. Read `cleaned_issues.csv` with the required fields `issue_id`,
   `issue_number`, `title`, `description`, and `repository`.
2. Clean markup and whitespace while retaining readable text, links/URLs, and
   code content. Newlines are joined with spaces, and the known `TheFlask`
   concatenation is normalized to `The Flask`.
3. Exclude placeholder descriptions, titles that are only spam, clearly
   generated test-coverage reports, and generic titles with no description.
   Missing repository names and issue IDs are not eligible for comparison.
   A specific title may still be considered when its description is empty.
4. Embed the remaining title/description text with the existing
   `all-MiniLM-L6-v2` Sentence Transformer (384-dimensional vectors).
5. Calculate cosine similarity, keep each unordered pair once, require the
   same normalized non-empty repository and different issue IDs, and apply a
   default threshold of `0.80`.
6. Write every original output field to the existing results folder. CSV
   fields are quoted and multiline values remain within one logical record.

The function `detect_duplicates()` returns a pandas DataFrame, making it
callable from another Python application without coupling this module to
FastAPI or Streamlit:

```python
from duplicate_detection.duplicate_detector import detect_duplicates

candidates = detect_duplicates()
```

Callers can override `data_file`, `results_file`, and
`similarity_threshold`, or inject an object implementing
`generate_embeddings(texts)` for application-managed model lifecycle/testing.
The default model and threshold are unchanged.

## Semantic retrieval

Use `search_related_issues()` to rank eligible existing issues by cosine
similarity to a text query. It returns up to `top_k` rows with their source
issue fields and a `similarity_score`; `exclude_issue_id` omits the query issue
when searching from an existing issue:

```python
from duplicate_detection.duplicate_detector import search_related_issues

related = search_related_issues(
    "Login fails after password reset",
    top_k=5,
)
```

The search uses the same Sentence Transformer and preprocessing/filtering as
duplicate detection, and can accept an injected embedder for tests or managed
model lifecycle.

## Dependencies and commands

The module uses Python, pandas, NumPy, scikit-learn, and
sentence-transformers. The existing project environment already imports these
packages; no project-wide dependency manifest is added or changed here.

Run commands from the `github_intelligence_platform` project root:

```powershell
# Run duplicate detection at the default 0.80 threshold
python -m duplicate_detection.duplicate_detector

# Run the module tests (no model download is needed by the tests)
python -m pytest duplicate_detection/tests

# Create/update a manual review queue from the existing 0.80 candidate CSV
python -m duplicate_detection.validation --initialize

# Generate a 0.75 candidate pool for comparing all four thresholds, then queue it
python -m duplicate_detection.validation --build-pool --initialize

# After reviewing and verifying labels, calculate metrics
python -m duplicate_detection.validation --evaluate

# Evaluate semantic retrieval using verified pair labels
python -m duplicate_detection.validation --evaluate-precision-at-k
```

`--build-pool` writes
`duplicate_detection/results/threshold_validation_pool_0_75.csv`; it does not
overwrite `duplicate_candidates.csv`. It runs the model again to ensure the
review pool covers candidates from 0.75 through 0.90. `--initialize` preserves
prior annotations for pairs that remain in the selected pool, and labels for
which no explicit verification status exists are treated as preliminary.

## Manual review and evaluation

The queue is
`duplicate_detection/validation/manual_validation.csv`. Every generated
`ground_truth_label` is blank. Review each pair against the issue details and
enter `1` for duplicate or `0` for not duplicate. Set `label_status` to
`verified` only after a human has checked the pair; use `review_notes` for
context. `review_flags` point out blank descriptions, likely spam, generated
reports, or identical issue content that merits attention. They do not assign
labels.

Evaluation uses only rows explicitly marked `verified`, and requires at least
10 verified examples of each class by default. Change the minimum only when
you have a reason to do so:

```powershell
python -m duplicate_detection.validation --evaluate --min-per-class 20
```

The report compares thresholds `0.75`, `0.80`, `0.85`, and `0.90`, and
contains precision, recall, F1, and TN/FP/FN/TP counts only when the label
minimum is met and the candidate pool covers the threshold. It recommends a
threshold only when that pool covers all four thresholds and both classes
meet the minimum; the recommendation maximizes F1, then precision, then
threshold. Otherwise it reports why no recommendation is available.

Precision@k is reported in
`duplicate_detection/results/precision_at_k_evaluation.csv` for k values 1, 3,
5, and 10 by default (override with `--precision-k`). It ranks all candidate
pairs in both query directions, including unreviewed and preliminary results.
A query contributes only when it has at least k results and every result in
its actual top-k has a verified relevance judgment; otherwise that query is
excluded and reported as incomplete. This is a reviewed-candidate evaluation,
not a measurement over every issue in the corpus.

**Evaluation limitation:** these metrics describe only verified pairs in the
manually reviewed candidate pool. Recall is not recall over every possible
pair of issues; the detector did not label or enumerate all negative issue
pairs. A 0.75 pool is needed to compare all listed thresholds fairly. The
current `results/duplicate_candidates.csv` contains 483 logical candidate
records, and the manual validation file currently has no verified labels, so
performance evaluation is not complete and no Precision@k score or
data-based threshold recommendation is available yet. The module does not
fabricate labels or metrics.

## Reproducibility and limitations

Pair ordering is deterministic for a fixed input/model output. Similarities
are bounded to the mathematical cosine range before writing, and CSV export
uses a stable column order and quoting. Reproducibility across machines still
depends on the installed Sentence Transformer, its model weights, hardware,
and package versions.

The detector is a candidate generator, not a proof of duplication. Semantic
similarity can miss duplicates below threshold and can produce false
positives, especially for templated reports or short issue text. Conservative
filters cover the observed spam/generated-report cases, but may need adjustment
for other repositories after human review. Comments, labels, issue status, and
cross-repository duplicates are not used by this algorithm.
