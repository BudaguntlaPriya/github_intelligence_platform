import argparse
import csv
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from .duplicate_detector import (
    DATA_FILE,
    DEFAULT_OUTPUT_FILE,
    OUTPUT_COLUMNS,
    RESULT_DIR,
    SIMILARITY_THRESHOLD,
    detect_duplicates,
    _is_generated_test_report,
    _normalized_phrase,
)


VALIDATION_DIR = Path(__file__).resolve().parent / "validation"
MANUAL_VALIDATION_FILE = VALIDATION_DIR / "manual_validation.csv"
THRESHOLD_EVALUATION_FILE = RESULT_DIR / "threshold_evaluation.csv"
PRECISION_AT_K_FILE = RESULT_DIR / "precision_at_k_evaluation.csv"
THRESHOLD_POOL_FILE = RESULT_DIR / "threshold_validation_pool_0_75.csv"
THRESHOLDS = (0.75, 0.80, 0.85, 0.90)
DEFAULT_PRECISION_K = (1, 3, 5, 10)
MIN_VERIFIED_PER_CLASS = 10
LABEL_COLUMN = "ground_truth_label"
STATUS_COLUMN = "label_status"
VALIDATION_COLUMNS = OUTPUT_COLUMNS + (
    LABEL_COLUMN,
    STATUS_COLUMN,
    "review_notes",
    "review_flags",
    "candidate_pool_threshold",
)


def _pair_key(row):
    return str(row["issue_id_1"]), str(row["issue_id_2"])


def _review_flags(row):
    flags = []
    title_1 = str(row.get("title_1", ""))
    title_2 = str(row.get("title_2", ""))
    description_1 = str(row.get("description_1", ""))
    description_2 = str(row.get("description_2", ""))

    if not description_1.strip() or not description_2.strip():
        flags.append("One or both descriptions are blank; review title evidence.")
    if _normalized_phrase(title_1) in {"", "spam", "ai spam"} or _normalized_phrase(
        title_2
    ) in {"", "spam", "ai spam"}:
        flags.append("A title is empty or appears to be spam.")
    if _is_generated_test_report(title_1, description_1) or _is_generated_test_report(
        title_2, description_2
    ):
        flags.append("Generated test-coverage report; verify it is a real issue.")
    if (
        _normalized_phrase(title_1) == _normalized_phrase(title_2)
        and description_1.strip() == description_2.strip()
    ):
        flags.append("Identical title and description; check for reposted reports.")
    return " ".join(flags)


def initialize_manual_validation(
    candidates_file=DEFAULT_OUTPUT_FILE,
    validation_file=MANUAL_VALIDATION_FILE,
    pool_threshold=SIMILARITY_THRESHOLD,
):
    """Create/update a review queue without assigning human labels."""
    candidate_path = Path(candidates_file)
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {candidate_path}")
    if not 0.0 <= pool_threshold <= 1.0:
        raise ValueError("pool_threshold must be between 0.0 and 1.0")

    candidates = pd.read_csv(
        candidate_path,
        keep_default_na=False,
        dtype=str,
    )
    missing_columns = [
        column for column in OUTPUT_COLUMNS if column not in candidates.columns
    ]
    if missing_columns:
        raise ValueError(f"Candidate file is missing columns: {missing_columns}")

    validation_path = Path(validation_file)
    existing_rows = {}
    if validation_path.exists():
        existing = pd.read_csv(
            validation_path,
            keep_default_na=False,
            dtype=str,
        )
        for _, row in existing.iterrows():
            key = _pair_key(row)
            if key in existing_rows:
                raise ValueError(f"Duplicate pair in manual validation file: {key}")
            existing_rows[key] = row.to_dict()

    output_rows = []
    for _, candidate in candidates.iterrows():
        row = {column: candidate[column] for column in OUTPUT_COLUMNS}
        prior = existing_rows.get(_pair_key(row), {})
        label = prior.get(LABEL_COLUMN, "").strip()
        if label not in {"", "0", "1"}:
            raise ValueError(
                f"Invalid {LABEL_COLUMN} value for pair {_pair_key(row)}: {label}"
            )

        prior_status = prior.get(STATUS_COLUMN, "").strip()
        if prior_status in {"unreviewed", "preliminary", "verified"}:
            label_status = prior_status
        elif label:
            label_status = "preliminary"
        else:
            label_status = "unreviewed"

        row.update(
            {
                LABEL_COLUMN: label,
                STATUS_COLUMN: label_status,
                "review_notes": prior.get("review_notes", ""),
                "review_flags": _review_flags(row),
                "candidate_pool_threshold": f"{pool_threshold:.2f}",
            }
        )
        output_rows.append(row)

    validation_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows, columns=VALIDATION_COLUMNS).to_csv(
        validation_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )
    return pd.DataFrame(output_rows, columns=VALIDATION_COLUMNS)


def evaluate_thresholds(
    validation_file=MANUAL_VALIDATION_FILE,
    thresholds=THRESHOLDS,
    min_per_class=MIN_VERIFIED_PER_CLASS,
):
    """Evaluate only verified labels and return results plus a gated recommendation."""
    validation_path = Path(validation_file)
    if not validation_path.exists():
        raise FileNotFoundError(
            f"Manual validation file not found: {validation_path}"
        )
    if min_per_class < 1:
        raise ValueError("min_per_class must be at least 1")
    if not thresholds:
        raise ValueError("at least one threshold is required")
    if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("all thresholds must be between 0.0 and 1.0")

    labels = pd.read_csv(
        validation_path,
        keep_default_na=False,
        dtype=str,
    )
    required = {LABEL_COLUMN, STATUS_COLUMN, "similarity_score"}
    missing_columns = sorted(required.difference(labels.columns))
    if missing_columns:
        raise ValueError(
            f"Manual validation file is missing columns: {missing_columns}"
        )

    verified_mask = labels[STATUS_COLUMN].str.strip().eq("verified")
    verified = labels.loc[verified_mask].copy()
    invalid_labels = ~verified[LABEL_COLUMN].isin({"0", "1"})
    if invalid_labels.any():
        raise ValueError(
            "Every verified row must have ground_truth_label set to 0 or 1"
        )

    scores = pd.to_numeric(verified["similarity_score"], errors="raise")
    candidate_scores = pd.to_numeric(labels["similarity_score"], errors="raise")
    y_true = verified[LABEL_COLUMN].astype(int).to_numpy()
    positive_count = int((y_true == 1).sum())
    negative_count = int((y_true == 0).sum())
    pool_threshold_values = (
        pd.to_numeric(labels["candidate_pool_threshold"], errors="raise")
        if "candidate_pool_threshold" in labels.columns
        else pd.Series([SIMILARITY_THRESHOLD])
    )
    if pool_threshold_values.empty:
        pool_threshold = SIMILARITY_THRESHOLD
    else:
        unique_pool_thresholds = sorted(pool_threshold_values.unique())
        if len(unique_pool_thresholds) != 1:
            raise ValueError(
                "Manual validation rows must use one candidate-pool threshold"
            )
        pool_threshold = float(unique_pool_thresholds[0])
    if not 0.0 <= pool_threshold <= 1.0:
        raise ValueError("candidate-pool threshold must be between 0.0 and 1.0")

    class_gate_passed = (
        positive_count >= min_per_class and negative_count >= min_per_class
    )
    evaluation_scope = (
        f"Verified candidate pairs generated at >= {pool_threshold:.2f}; "
        "recall is limited to this reviewed candidate pool, not all issue pairs."
    )
    results = []
    for threshold in thresholds:
        covered_by_pool = threshold >= pool_threshold
        status = "evaluated"
        metric_values = {
            "precision": "",
            "recall": "",
            "f1": "",
            "tn": "",
            "fp": "",
            "fn": "",
            "tp": "",
        }

        if not covered_by_pool:
            status = "candidate_pool_does_not_cover_threshold"
        elif not class_gate_passed:
            status = "insufficient_verified_labels"
        else:
            y_pred = (scores.to_numpy() >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1],
            ).ravel()
            metric_values = {
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }

        results.append(
            {
                "threshold": threshold,
                "status": status,
                "verified_pairs": len(verified),
                "verified_duplicates": positive_count,
                "verified_non_duplicates": negative_count,
                "candidate_pool_threshold": pool_threshold,
                "pool_pairs_above_threshold": int(
                    (candidate_scores >= threshold).sum()
                ),
                "evaluation_scope": evaluation_scope,
                **metric_values,
            }
        )

    result_frame = pd.DataFrame(results)
    evaluated = result_frame[result_frame["status"] == "evaluated"]
    recommendation = None
    if (
        pool_threshold <= min(thresholds)
        and class_gate_passed
        and not evaluated.empty
    ):
        recommendation = (
            evaluated.sort_values(
                ["f1", "precision", "threshold"],
                ascending=[False, False, False],
                kind="mergesort",
            )
            .iloc[0]["threshold"]
        )
        recommendation = float(recommendation)
    return result_frame, recommendation


def evaluate_precision_at_k(
    validation_file=MANUAL_VALIDATION_FILE,
    k_values=DEFAULT_PRECISION_K,
):
    """Calculate macro Precision@k from explicitly verified issue-pair labels.

    Each candidate pair is evaluated in both query directions. A query
    contributes only when its actual top-k candidates all have verified
    judgments; unjudged candidates remain in the ranking and cannot be skipped.
    """
    validation_path = Path(validation_file)
    if not validation_path.exists():
        raise FileNotFoundError(
            f"Manual validation file not found: {validation_path}"
        )
    if not k_values:
        raise ValueError("at least one k value is required")
    if any(
        not isinstance(k, int) or isinstance(k, bool) or k < 1
        for k in k_values
    ):
        raise ValueError("all k values must be positive integers")

    labels = pd.read_csv(
        validation_path,
        keep_default_na=False,
        dtype=str,
    )
    required = {
        "issue_id_1",
        "issue_id_2",
        LABEL_COLUMN,
        STATUS_COLUMN,
        "similarity_score",
    }
    missing_columns = sorted(required.difference(labels.columns))
    if missing_columns:
        raise ValueError(
            f"Manual validation file is missing columns: {missing_columns}"
        )

    verified = labels.loc[labels[STATUS_COLUMN].str.strip().eq("verified")].copy()
    invalid_labels = ~verified[LABEL_COLUMN].isin({"0", "1"})
    if invalid_labels.any():
        raise ValueError(
            "Every verified row must have ground_truth_label set to 0 or 1"
        )

    labels["similarity_score"] = pd.to_numeric(
        labels["similarity_score"],
        errors="raise",
    )
    ranked_by_query = {}
    for _, row in labels.iterrows():
        issue_id_1 = str(row["issue_id_1"]).strip()
        issue_id_2 = str(row["issue_id_2"]).strip()
        if not issue_id_1 or not issue_id_2 or issue_id_1 == issue_id_2:
            raise ValueError("Candidate pairs must contain two distinct issue IDs")

        is_verified = row[STATUS_COLUMN].strip() == "verified"
        relevance = int(row[LABEL_COLUMN]) if is_verified else None
        score = float(row["similarity_score"])
        for query_id, related_id in (
            (issue_id_1, issue_id_2),
            (issue_id_2, issue_id_1),
        ):
            query_results = ranked_by_query.setdefault(query_id, {})
            previous = query_results.get(related_id)
            if (
                previous is not None
                and previous[2]
                and is_verified
                and previous[1] != relevance
            ):
                raise ValueError(
                    "A verified issue pair has conflicting relevance labels"
                )
            if previous is None:
                query_results[related_id] = (score, relevance, is_verified)
            else:
                query_results[related_id] = (
                    max(score, previous[0]),
                    previous[1] if previous[2] else relevance,
                    previous[2] or is_verified,
                )

    verified_query_count = sum(
        any(result[2] for result in query_results.values())
        for query_results in ranked_by_query.values()
    )
    scope = (
        "Macro Precision@k over candidate rankings; a query is included only "
        "when every result in its actual top-k has a verified relevance "
        "judgment. This is not a full-corpus evaluation."
    )
    results = []
    for k in k_values:
        top_k_by_query = [
            sorted(
                query_results.items(),
                key=lambda item: (-item[1][0], item[0]),
            )[:k]
            for query_results in ranked_by_query.values()
            if len(query_results) >= k
        ]
        eligible_queries = [
            top_k
            for top_k in top_k_by_query
            if all(result[2] for _, result in top_k)
        ]
        incomplete_queries = len(top_k_by_query) - len(eligible_queries)
        if not verified.empty and not eligible_queries:
            status = (
                "incomplete_relevance_judgments"
                if incomplete_queries
                else "insufficient_verified_candidates"
            )
            precision = ""
        elif verified.empty:
            status = "insufficient_verified_labels"
            precision = ""
        else:
            status = "evaluated"
            query_precisions = []
            for top_k in eligible_queries:
                relevant_in_top_k = sum(result[1] for _, result in top_k)
                query_precisions.append(relevant_in_top_k / k)
            precision = sum(query_precisions) / len(query_precisions)

        results.append(
            {
                "k": k,
                "status": status,
                "precision_at_k": precision,
                "evaluated_queries": len(eligible_queries),
                "incomplete_queries": incomplete_queries,
                "verified_queries": verified_query_count,
                "verified_pairs": len(verified),
                "evaluation_scope": scope,
            }
        )
    return pd.DataFrame(results)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a manual duplicate-review queue or evaluate verified labels."
    )
    parser.add_argument(
        "--build-pool",
        action="store_true",
        help="Generate the candidate pool at the lowest comparison threshold (0.75).",
    )
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="Create/update the manual validation CSV with blank labels.",
    )
    parser.add_argument(
        "--candidate-file",
        type=Path,
        default=None,
        help="Candidate CSV to use when initializing the review queue.",
    )
    parser.add_argument(
        "--pool-threshold",
        type=float,
        default=None,
        help="Threshold used to generate --candidate-file.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate verified labels and write the threshold report.",
    )
    parser.add_argument(
        "--evaluate-precision-at-k",
        action="store_true",
        help="Evaluate semantic retrieval using verified relevance labels.",
    )
    parser.add_argument(
        "--precision-k",
        nargs="+",
        type=int,
        default=DEFAULT_PRECISION_K,
        help="k values for Precision@k (default: 1 3 5 10).",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=MIN_VERIFIED_PER_CLASS,
        help="Minimum verified examples of each class required for metrics.",
    )
    args = parser.parse_args(argv)

    candidate_file = args.candidate_file or DEFAULT_OUTPUT_FILE
    pool_threshold = (
        args.pool_threshold
        if args.pool_threshold is not None
        else SIMILARITY_THRESHOLD
    )

    if args.build_pool:
        candidate_file = THRESHOLD_POOL_FILE
        pool_threshold = min(THRESHOLDS)
        detect_duplicates(
            data_file=DATA_FILE,
            results_file=candidate_file,
            similarity_threshold=pool_threshold,
        )
        args.initialize = True

    if args.initialize:
        review_rows = initialize_manual_validation(
            candidates_file=candidate_file,
            validation_file=MANUAL_VALIDATION_FILE,
            pool_threshold=pool_threshold,
        )
        print(
            f"Manual review rows: {len(review_rows)} "
            f"({MANUAL_VALIDATION_FILE})"
        )
        print("Labels are blank; no human decisions were generated.")

    if args.evaluate:
        report, recommendation = evaluate_thresholds(
            validation_file=MANUAL_VALIDATION_FILE,
            min_per_class=args.min_per_class,
        )
        THRESHOLD_EVALUATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(
            THRESHOLD_EVALUATION_FILE,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        print(report.to_string(index=False))
        if recommendation is None:
            print("Threshold recommendation: unavailable.")
        else:
            print(f"Recommended threshold by verified-label F1: {recommendation:.2f}")
        print(f"Evaluation report: {THRESHOLD_EVALUATION_FILE}")

    if args.evaluate_precision_at_k:
        report = evaluate_precision_at_k(
            validation_file=MANUAL_VALIDATION_FILE,
            k_values=args.precision_k,
        )
        PRECISION_AT_K_FILE.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(
            PRECISION_AT_K_FILE,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        print(report.to_string(index=False))
        print(f"Precision@k report: {PRECISION_AT_K_FILE}")

    if not (
        args.build_pool
        or args.initialize
        or args.evaluate
        or args.evaluate_precision_at_k
    ):
        parser.print_help()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
