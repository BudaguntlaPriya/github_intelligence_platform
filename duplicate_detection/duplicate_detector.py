import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .embeddings import IssueEmbedder
from .preprocess import build_issue_text, clean_text
from .similarity import SimilarityCalculator


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "cleaned_issues.csv"
RESULT_DIR = BASE_DIR / "duplicate_detection" / "results"
DEFAULT_OUTPUT_FILE = RESULT_DIR / "duplicate_candidates.csv"

SIMILARITY_THRESHOLD = 0.80
PLACEHOLDER_DESCRIPTIONS = {"llm junk removed"}
LOW_INFORMATION_TITLES = {
    "bug",
    "error",
    "flask",
    "help",
    "issue",
    "problem",
    "question",
    "test",
    "testing",
}
SPAM_TITLES = {"spam", "ai spam"}

REQUIRED_COLUMNS = (
    "issue_id",
    "issue_number",
    "title",
    "description",
    "repository",
)
OUTPUT_COLUMNS = (
    "issue_id_1",
    "issue_id_2",
    "issue_number_1",
    "issue_number_2",
    "repository_1",
    "repository_2",
    "title_1",
    "title_2",
    "description_1",
    "description_2",
    "similarity_score",
    "confidence",
)

_GENERATED_TEST_REPORT_TITLE = re.compile(
    r"^missing tests?:\s*\d+\s+source files? have no test coverage$",
    re.IGNORECASE,
)


def _normalized_phrase(value):
    return re.sub(r"[\W_]+", " ", str(value or "").casefold()).strip()


def _is_generated_test_report(title, description):
    if not _GENERATED_TEST_REPORT_TITLE.fullmatch(str(title or "").strip()):
        return False

    description_text = str(description or "").casefold()
    return (
        "affected paths:" in description_text
        and "test files were found" in description_text
    )


def prepare_issues(issues):
    """Validate issue columns and return eligible rows with embedding text."""
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in issues.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    prepared = issues.copy()
    for column in ("title", "description", "issue_id", "repository"):
        prepared[column] = prepared[column].fillna("").astype(str)
    prepared["issue_id"] = prepared["issue_id"].str.strip()

    descriptions_normalized = (
        prepared["description"]
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    placeholder_mask = descriptions_normalized.isin(
        PLACEHOLDER_DESCRIPTIONS
    )
    generated_report_mask = pd.Series(
        [
            _is_generated_test_report(title, description)
            for title, description in zip(
                prepared["title"],
                prepared["description"],
            )
        ],
        index=prepared.index,
    )
    spam_mask = prepared["title"].map(
        lambda value: _normalized_phrase(value) in SPAM_TITLES
    )
    low_information_mask = pd.Series(
        [
            not description.strip()
            and _normalized_phrase(title) in LOW_INFORMATION_TITLES
            for title, description in zip(
                prepared["title"],
                prepared["description"],
            )
        ],
        index=prepared.index,
    )
    texts = [
        build_issue_text(title, description)
        for title, description in zip(
            prepared["title"],
            prepared["description"],
        )
    ]
    usable_text_mask = pd.Series(
        [bool(text.strip()) for text in texts],
        index=prepared.index,
    )
    identity_mask = (
        prepared["issue_id"].ne("")
        & prepared["repository"].str.strip().ne("")
    )
    eligible_mask = (
        ~placeholder_mask
        & ~generated_report_mask
        & ~spam_mask
        & ~low_information_mask
        & usable_text_mask
        & identity_mask
    )

    eligible = prepared.loc[eligible_mask].copy()
    unique_issue_mask = ~eligible["issue_id"].duplicated(keep="first")
    eligible = eligible.loc[unique_issue_mask].copy()
    eligible_texts = [
        text
        for text, unique in zip(
            [
                text
                for text, include in zip(texts, eligible_mask.to_numpy())
                if include
            ],
            unique_issue_mask.to_numpy(),
        )
        if unique
    ]
    return eligible, eligible_texts


def generate_duplicate_pairs(
    issues,
    similarity_matrix,
    threshold=SIMILARITY_THRESHOLD,
):
    """Build unique, same-repository candidate pairs from a similarity matrix."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")

    issue_count = len(issues)
    matrix = np.asarray(similarity_matrix)
    if matrix.shape != (issue_count, issue_count):
        raise ValueError(
            "similarity_matrix shape must match the number of issue rows"
        )

    repositories = (
        issues["repository"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .to_numpy()
    )
    issue_ids = (
        issues["issue_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .to_numpy()
    )
    same_repository = (
        (repositories[:, None] == repositories[None, :])
        & (repositories[:, None] != "")
    )
    different_issue = (
        (issue_ids[:, None] != issue_ids[None, :])
        & (issue_ids[:, None] != "")
        & (issue_ids[None, :] != "")
    )
    upper_triangle = np.triu(
        np.ones((issue_count, issue_count), dtype=bool),
        k=1,
    )
    row_indices, column_indices = np.where(
        (matrix >= threshold)
        & same_repository
        & different_issue
        & upper_triangle
    )

    pairs_by_id = {}
    for row_index, column_index in zip(row_indices, column_indices):
        pair_key = frozenset((issue_ids[row_index], issue_ids[column_index]))
        score = float(np.clip(matrix[row_index, column_index], -1.0, 1.0))
        first = issues.iloc[row_index]
        second = issues.iloc[column_index]
        pair = {
            "issue_id_1": first["issue_id"],
            "issue_id_2": second["issue_id"],
            "issue_number_1": first["issue_number"],
            "issue_number_2": second["issue_number"],
            "repository_1": first["repository"],
            "repository_2": second["repository"],
            "title_1": first["title"],
            "title_2": second["title"],
            "description_1": first["description"],
            "description_2": second["description"],
            "similarity_score": score,
            "confidence": (
                "Very High"
                if score >= 0.90
                else "High"
                if score >= SIMILARITY_THRESHOLD
                else "Review"
            ),
        }
        previous = pairs_by_id.get(pair_key)
        if previous is None or score > previous["similarity_score"]:
            pairs_by_id[pair_key] = pair

    return pd.DataFrame(list(pairs_by_id.values()), columns=OUTPUT_COLUMNS).sort_values(
        by=["similarity_score", "issue_id_1", "issue_id_2"],
        ascending=[False, True, True],
        kind="mergesort",
        ignore_index=True,
    )


def write_candidates(results, output_file=DEFAULT_OUTPUT_FILE):
    """Write candidates with a stable schema and CSV-safe multiline quoting."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.reindex(columns=OUTPUT_COLUMNS).to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )
    return output_path


def search_related_issues(
    query,
    data_file=DATA_FILE,
    top_k=5,
    embedder=None,
    exclude_issue_id=None,
):
    """Rank existing eligible issues by semantic similarity to query text."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    query_text = clean_text(query)
    if not query_text:
        raise ValueError("query must contain non-empty text")

    source_path = Path(data_file)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    issues = pd.read_csv(
        source_path,
        keep_default_na=False,
        dtype=str,
    )
    eligible, texts = prepare_issues(issues)
    if exclude_issue_id is not None:
        exclude_mask = eligible["issue_id"].ne(str(exclude_issue_id))
        eligible = eligible.loc[exclude_mask].reset_index(drop=True)
        texts = [
            text
            for text, include in zip(texts, exclude_mask.to_numpy())
            if include
        ]

    if eligible.empty:
        return eligible.assign(similarity_score=pd.Series(dtype=float))

    active_embedder = embedder if embedder is not None else IssueEmbedder()
    embeddings = np.asarray(
        active_embedder.generate_embeddings([query_text, *texts]),
        dtype=float,
    )
    expected_rows = len(texts) + 1
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise ValueError(
            "embedder must return one vector for the query and each eligible issue"
        )

    scores = SimilarityCalculator.calculate_query_similarity(
        embeddings[0],
        embeddings[1:],
    )
    results = eligible.copy()
    results["similarity_score"] = np.clip(scores, -1.0, 1.0)
    return results.sort_values(
        by=["similarity_score", "issue_id"],
        ascending=[False, True],
        kind="mergesort",
    ).head(top_k).reset_index(drop=True)


def detect_duplicates(
    data_file=DATA_FILE,
    results_file=DEFAULT_OUTPUT_FILE,
    similarity_threshold=SIMILARITY_THRESHOLD,
    embedder=None,
):
    """Detect same-repository duplicate candidates and return the result frame.

    An embedder implementing ``generate_embeddings(texts)`` may be injected by
    callers or tests; otherwise the configured Sentence Transformer is used.
    """
    source_path = Path(data_file)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0.0 and 1.0")

    issues = pd.read_csv(
        source_path,
        keep_default_na=False,
        dtype=str,
    )
    eligible, texts = prepare_issues(issues)

    if len(eligible) < 2:
        results = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        active_embedder = embedder if embedder is not None else IssueEmbedder()
        embeddings = active_embedder.generate_embeddings(texts)
        similarity_matrix = SimilarityCalculator.calculate_similarity(
            embeddings
        )
        results = generate_duplicate_pairs(
            eligible,
            similarity_matrix,
            threshold=similarity_threshold,
        )

    output_path = write_candidates(results, results_file)
    print(f"Issues available for analysis: {len(eligible)}")
    print(f"Similarity threshold: {similarity_threshold:.2f}")
    print(f"Potential duplicate pairs: {len(results)}")
    print(f"Results saved: {output_path}")
    return results


if __name__ == "__main__":
    detect_duplicates()
