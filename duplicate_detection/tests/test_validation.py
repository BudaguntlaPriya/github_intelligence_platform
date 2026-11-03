import pandas as pd

from duplicate_detection.duplicate_detector import OUTPUT_COLUMNS
from duplicate_detection.validation import (
    evaluate_thresholds,
    evaluate_precision_at_k,
    initialize_manual_validation,
)


def _candidate(issue_id_1, issue_id_2, score, title="Same title", description="Same"):
    row = {column: "" for column in OUTPUT_COLUMNS}
    row.update(
        {
            "issue_id_1": issue_id_1,
            "issue_id_2": issue_id_2,
            "issue_number_1": issue_id_1,
            "issue_number_2": issue_id_2,
            "repository_1": "org/repo",
            "repository_2": "org/repo",
            "title_1": title,
            "title_2": title,
            "description_1": description,
            "description_2": description,
            "similarity_score": score,
            "confidence": "High",
        }
    )
    return row


def _write_labels(path, rows):
    frame = pd.DataFrame(rows)
    frame["label_status"] = "verified"
    frame["ground_truth_label"] = frame["ground_truth_label"].astype(str)
    frame["candidate_pool_threshold"] = frame[
        "candidate_pool_threshold"
    ].astype(str)
    frame.to_csv(path, index=False)


def test_initialize_validation_creates_blank_labels_and_review_flags(tmp_path):
    candidate_file = tmp_path / "candidates.csv"
    validation_file = tmp_path / "validation" / "manual_validation.csv"
    candidates = pd.DataFrame(
        [
            _candidate("1", "2", 0.95),
            _candidate("3", "4", 0.82, description=""),
        ]
    )
    candidates.to_csv(candidate_file, index=False)

    review = initialize_manual_validation(
        candidate_file,
        validation_file,
        pool_threshold=0.75,
    )

    assert len(review) == 2
    assert review["ground_truth_label"].tolist() == ["", ""]
    assert review["label_status"].tolist() == ["unreviewed", "unreviewed"]
    assert "Identical title and description" in review.iloc[0]["review_flags"]
    assert "descriptions are blank" in review.iloc[1]["review_flags"]


def test_initialize_preserves_existing_labels_as_preliminary(tmp_path):
    candidate_file = tmp_path / "candidates.csv"
    validation_file = tmp_path / "manual_validation.csv"
    pd.DataFrame([_candidate("1", "2", 0.95)]).to_csv(
        candidate_file,
        index=False,
    )
    pd.DataFrame(
        [
            {
                "issue_id_1": "1",
                "issue_id_2": "2",
                "ground_truth_label": "1",
                "review_notes": "Existing, not yet verified",
            }
        ]
    ).to_csv(validation_file, index=False)

    review = initialize_manual_validation(candidate_file, validation_file)

    assert review.loc[0, "ground_truth_label"] == "1"
    assert review.loc[0, "label_status"] == "preliminary"
    assert review.loc[0, "review_notes"] == "Existing, not yet verified"


def test_evaluate_thresholds_requires_both_verified_classes(tmp_path):
    validation_file = tmp_path / "manual_validation.csv"
    rows = [
        {
            "similarity_score": 0.95,
            "ground_truth_label": 1,
            "candidate_pool_threshold": 0.75,
        }
    ]
    _write_labels(validation_file, rows)

    results, recommendation = evaluate_thresholds(
        validation_file,
        thresholds=(0.75, 0.80),
        min_per_class=2,
    )

    assert recommendation is None
    assert set(results["status"]) == {"insufficient_verified_labels"}
    assert results["precision"].eq("").all()
    assert results["tp"].eq("").all()


def test_evaluate_thresholds_calculates_metrics_and_recommends_best_f1(
    tmp_path,
):
    validation_file = tmp_path / "manual_validation.csv"
    rows = []
    for score, label in ((0.88, 1), (0.95, 1), (0.82, 0), (0.78, 0)):
        rows.append(
            {
                "similarity_score": score,
                "ground_truth_label": label,
                "candidate_pool_threshold": 0.75,
            }
        )
    _write_labels(validation_file, rows)

    results, recommendation = evaluate_thresholds(
        validation_file,
        min_per_class=2,
    )

    best = results.loc[results["threshold"] == 0.85].iloc[0]
    assert recommendation == 0.85
    assert best["precision"] == 1.0
    assert best["recall"] == 1.0
    assert best["f1"] == 1.0
    assert (best["tn"], best["fp"], best["fn"], best["tp"]) == (2, 0, 0, 2)
    assert "not all issue pairs" in best["evaluation_scope"]


def test_threshold_below_candidate_pool_is_not_reported_as_evaluated(tmp_path):
    validation_file = tmp_path / "manual_validation.csv"
    rows = []
    for score, label in ((0.90, 1), (0.82, 0)):
        rows.append(
            {
                "similarity_score": score,
                "ground_truth_label": label,
                "candidate_pool_threshold": 0.80,
            }
        )
    _write_labels(validation_file, rows)

    results, recommendation = evaluate_thresholds(
        validation_file,
        thresholds=(0.75, 0.80),
        min_per_class=1,
    )

    assert recommendation is None
    assert results.loc[results["threshold"] == 0.75, "status"].item() == (
        "candidate_pool_does_not_cover_threshold"
    )
    assert results.loc[results["threshold"] == 0.80, "status"].item() == "evaluated"


def test_precision_at_k_does_not_skip_unreviewed_results(
    tmp_path,
):
    validation_file = tmp_path / "manual_validation.csv"
    pairs = []
    for issue_id_2, score, label in (
        ("B", 0.95, "1"),
        ("C", 0.90, "0"),
        ("D", 0.85, "1"),
    ):
        row = _candidate("A", issue_id_2, score)
        row.update(
            {
                "ground_truth_label": label,
                "label_status": "verified",
                "candidate_pool_threshold": "0.75",
            }
        )
        pairs.append(row)
    unreviewed = _candidate("A", "E", 0.99)
    unreviewed.update(
        {
            "ground_truth_label": "",
            "label_status": "unreviewed",
            "candidate_pool_threshold": "0.75",
        }
    )
    pairs.append(unreviewed)
    pd.DataFrame(pairs).to_csv(validation_file, index=False)

    results = evaluate_precision_at_k(validation_file, k_values=(1, 2))

    at_one = results.loc[results["k"] == 1].iloc[0]
    assert at_one["status"] == "evaluated"
    assert at_one["evaluated_queries"] == 3
    assert at_one["incomplete_queries"] == 2
    assert at_one["precision_at_k"] == 2 / 3

    at_two = results.loc[results["k"] == 2].iloc[0]
    assert at_two["status"] == "incomplete_relevance_judgments"
    assert at_two["evaluated_queries"] == 0
    assert at_two["incomplete_queries"] == 1
    assert at_two["precision_at_k"] == ""
    assert "not a full-corpus evaluation" in at_two["evaluation_scope"]


def test_precision_at_k_reports_incomplete_top_k_without_a_score(tmp_path):
    validation_file = tmp_path / "manual_validation.csv"
    verified = _candidate("A", "B", 0.95)
    verified.update(
        {
            "ground_truth_label": "1",
            "label_status": "verified",
            "candidate_pool_threshold": "0.75",
        }
    )
    unreviewed = _candidate("A", "C", 0.99)
    unreviewed.update(
        {
            "ground_truth_label": "",
            "label_status": "unreviewed",
            "candidate_pool_threshold": "0.75",
        }
    )
    unreviewed_for_reverse_query = _candidate("B", "D", 0.98)
    unreviewed_for_reverse_query.update(
        {
            "ground_truth_label": "",
            "label_status": "unreviewed",
            "candidate_pool_threshold": "0.75",
        }
    )
    pd.DataFrame(
        [verified, unreviewed, unreviewed_for_reverse_query]
    ).to_csv(validation_file, index=False)

    result = evaluate_precision_at_k(validation_file, k_values=(1,)).iloc[0]

    assert result["status"] == "incomplete_relevance_judgments"
    assert result["evaluated_queries"] == 0
    assert result["incomplete_queries"] == 4
    assert result["precision_at_k"] == ""


def test_precision_at_k_reports_insufficient_verified_labels(tmp_path):
    validation_file = tmp_path / "manual_validation.csv"
    row = _candidate("A", "B", 0.9)
    row.update(
        {
            "ground_truth_label": "1",
            "label_status": "preliminary",
            "candidate_pool_threshold": "0.75",
        }
    )
    pd.DataFrame([row]).to_csv(validation_file, index=False)

    results = evaluate_precision_at_k(validation_file, k_values=(1,))

    assert results.loc[0, "status"] == "insufficient_verified_labels"
    assert results.loc[0, "precision_at_k"] == ""
