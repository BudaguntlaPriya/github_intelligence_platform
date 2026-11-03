import csv

import numpy as np
import pandas as pd

from duplicate_detection.duplicate_detector import (
    OUTPUT_COLUMNS,
    detect_duplicates,
    generate_duplicate_pairs,
    prepare_issues,
    search_related_issues,
    write_candidates,
)
from duplicate_detection.preprocess import build_issue_text, clean_text
from duplicate_detection.similarity import SimilarityCalculator


def _issue(issue_id, repository="org/repo", title="A useful title", description=""):
    return {
        "issue_id": issue_id,
        "issue_number": int(issue_id),
        "title": title,
        "description": description,
        "repository": repository,
    }


def test_preprocessing_fixes_theflask_without_losing_issue_content():
    text = clean_text(
        "TheFlask crashes.\n"
        "See [the traceback](https://example.test/trace).\n"
        "```python\nraise ValueError('bad')\n```"
    )

    assert "The Flask crashes." in text
    assert "https://example.test/trace" in text
    assert "raise ValueError('bad')" in text
    assert "TheFlask" not in text


def test_preprocessing_joins_lines_with_spaces_and_omits_empty_sections():
    assert clean_text("The\nFlask application\ncrashes") == (
        "The Flask application crashes"
    )
    assert build_issue_text("", "") == ""
    assert build_issue_text("TheFlask fails", "") == "Title: The Flask fails"


def test_similarity_calculator_returns_cosine_similarity():
    matrix = SimilarityCalculator.calculate_similarity(
        np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    )

    assert matrix.shape == (3, 3)
    assert matrix[0, 1] == 1.0
    assert matrix[0, 2] == 0.0


def test_query_similarity_calculates_scores_against_corpus():
    scores = SimilarityCalculator.calculate_query_similarity(
        np.array([1.0, 0.0]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    )

    np.testing.assert_allclose(scores, [1.0, 0.0])


def test_prepare_issues_excludes_spam_generated_reports_and_weak_empty_titles():
    issues = pd.DataFrame(
        [
            _issue("1", title="spam"),
            _issue(
                "2",
                title="Missing tests: 9 source files have no test coverage",
                description=(
                    "Affected paths: src/a.py. 9 test files were found."
                ),
            ),
            _issue("3", title="Flask", description=""),
            _issue("4", title="A specific failure", description=""),
        ]
    )

    eligible, texts = prepare_issues(issues)

    assert eligible["issue_id"].tolist() == ["4"]
    assert len(texts) == 1
    assert "A specific failure" in texts[0]


def test_pair_generation_enforces_repository_identity_threshold_and_unique_ids():
    issues = pd.DataFrame(
        [
            _issue("1", title="first"),
            _issue("2", title="second"),
            _issue("3", repository="org/other", title="third"),
            _issue("1", title="same issue id"),
        ]
    )
    similarities = np.array(
        [
            [1.0, 0.95, 0.99, 0.99],
            [0.95, 1.0, 0.99, 1.0000002],
            [0.99, 0.99, 1.0, 0.99],
            [0.99, 1.0000002, 0.99, 1.0],
        ]
    )

    pairs = generate_duplicate_pairs(issues, similarities, threshold=0.90)

    assert len(pairs) == 1
    assert {
        frozenset(pair)
        for pair in zip(pairs["issue_id_1"], pairs["issue_id_2"])
    } == {frozenset(("1", "2"))}
    assert pairs["similarity_score"].max() == 1.0
    assert all(pairs["repository_1"] == pairs["repository_2"])


def test_csv_export_preserves_multiline_values_and_schema(tmp_path):
    results = pd.DataFrame(
        [
            {
                "issue_id_1": "1",
                "issue_id_2": "2",
                "title_1": "Title line one\nTitle line two",
                "description_1": "First line\nSecond line",
            }
        ]
    )
    output = tmp_path / "nested" / "candidates.csv"

    write_candidates(results, output)

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["title_1"] == "Title line one\nTitle line two"
    assert rows[0]["description_1"] == "First line\nSecond line"
    assert list(rows[0]) == list(OUTPUT_COLUMNS)


def test_detector_exposes_injectable_embedder_and_returns_candidates(tmp_path):
    source = tmp_path / "issues.csv"
    output = tmp_path / "results" / "candidates.csv"
    pd.DataFrame(
        [
            _issue("1", title="Broken login", description="Login fails"),
            _issue("2", title="Login broken", description="Cannot log in"),
            _issue(
                "3",
                repository="org/other",
                title="Broken login",
                description="Login fails",
            ),
        ]
    ).to_csv(source, index=False)

    class FakeEmbedder:
        def generate_embeddings(self, texts):
            assert len(texts) == 3
            return np.array([[1.0, 0.0], [0.95, 0.05], [1.0, 0.0]])

    result = detect_duplicates(
        data_file=source,
        results_file=output,
        embedder=FakeEmbedder(),
    )

    assert len(result) == 1
    assert result.iloc[0]["repository_1"] == result.iloc[0]["repository_2"]
    assert output.exists()


def test_semantic_search_ranks_related_issues_and_can_exclude_query_issue(
    tmp_path,
):
    source = tmp_path / "issues.csv"
    pd.DataFrame(
        [
            _issue("1", title="Unrelated change"),
            _issue("2", title="Login does not work"),
            _issue("3", title="Login stopped working"),
        ]
    ).to_csv(source, index=False)

    class FakeEmbedder:
        def generate_embeddings(self, texts):
            assert texts[0] == "login failure"
            assert len(texts) == 3
            return np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.8, 0.6],
                ]
            )

    results = search_related_issues(
        "login failure",
        data_file=source,
        top_k=2,
        embedder=FakeEmbedder(),
        exclude_issue_id="2",
    )

    assert results["issue_id"].tolist() == ["3", "1"]
    assert results["similarity_score"].is_monotonic_decreasing
