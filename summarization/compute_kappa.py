"""
Inter-Annotator Agreement (Cohen's Kappa) Evaluation
Member 4: Summarization + Priority Prediction

Computes agreement and Cohen's Kappa score between primary gold annotations
and a teammate's independent annotations from data/gold_second_annotator.csv.

Usage:
    python compute_kappa.py [--file path/to/gold_second_annotator.csv] [--demo]
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np

# Safe compatibility patch
if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = int

_orig_np_array = np.array
def _compat_np_array(*args, **kwargs):
    if kwargs.get("copy") is None and "copy" in kwargs:
        kwargs["copy"] = False
    return _orig_np_array(*args, **kwargs)
np.array = _compat_np_array

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SECOND_FILE = os.path.join(BASE_DIR, "data", "gold_second_annotator.csv")
DEFAULT_GOLD_FILE = os.path.join(BASE_DIR, "data", "priority_gold.csv")
PRIORITY_LEVELS = ["Low", "Medium", "High", "Critical"]


def interpret_kappa(kappa):
    """Interpret Cohen's kappa using Landis & Koch (1977) scale."""
    if kappa < 0.0:
        return "Poor agreement (less than chance)"
    elif kappa <= 0.20:
        return "Slight agreement"
    elif kappa <= 0.40:
        return "Fair agreement"
    elif kappa <= 0.60:
        return "Moderate agreement"
    elif kappa <= 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


def compute_inter_annotator_agreement(second_file=DEFAULT_SECOND_FILE, gold_file=DEFAULT_GOLD_FILE, demo=False):
    """Compute agreement metrics between primary gold and second annotator."""
    if not os.path.exists(second_file):
        # Check root data/ as fallback
        alt_second = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "gold_second_annotator.csv"))
        if os.path.exists(alt_second):
            second_file = alt_second
        else:
            raise FileNotFoundError(f"Second annotator file not found: {second_file}")

    if not os.path.exists(gold_file):
        alt_gold = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "priority_gold.csv"))
        if os.path.exists(alt_gold):
            gold_file = alt_gold
        else:
            raise FileNotFoundError(f"Primary gold file not found: {gold_file}")

    df_sec = pd.read_csv(second_file)
    df_gold = pd.read_csv(gold_file).set_index("issue_id")["gold_priority"].to_dict()

    print("=" * 70)
    print("INTER-ANNOTATOR AGREEMENT EVALUATION (COHEN'S KAPPA)")
    print("=" * 70)
    print(f"Second Annotator File: {second_file}")
    print(f"Primary Gold File    : {gold_file}")
    print(f"Total Sampled Issues : {len(df_sec)}")

    if demo:
        print("\n" + "*" * 70)
        print("*** WARNING: DEMO DATA - DO NOT REPORT IN FINAL DOCUMENTATION OR PAPERS ***")
        print("*" * 70 + "\n")
        print("[DEMO MODE ACTIVATED] Populating mock teammate annotations with ~75% agreement...")
        demo_labels = []
        rng = np.random.RandomState(42)
        for _, row in df_sec.iterrows():
            iid = row["issue_id"]
            true_lbl = df_gold.get(iid, "Low")
            if rng.rand() < 0.75:
                demo_labels.append(true_lbl)
            else:
                # Random slight shift
                demo_labels.append("Medium" if true_lbl == "Low" else "Low")
        df_sec["teammate_priority"] = demo_labels

    # Check how many are annotated
    filled_mask = df_sec["teammate_priority"].fillna("").astype(str).str.strip() != ""
    n_filled = filled_mask.sum()
    n_total = len(df_sec)

    print(f"Annotated Issues     : {n_filled} / {n_total} ({(n_filled / n_total):.1%})")

    if n_filled == 0:
        print("\n" + "!" * 70)
        print("ACTION REQUIRED: Second annotator labels are currently pending.")
        print("A teammate must open 'data/gold_second_annotator.csv' and populate")
        print("the 'teammate_priority' column with one of:")
        print("    Low, Medium, High, Critical")
        print("Once filled, re-run this script to compute Cohen's kappa.")
        print("Tip: Run 'python compute_kappa.py --demo' to preview sample calculation.")
        print("!" * 70 + "\n")
        return None

    if n_filled < n_total:
        print(f"\n[NOTE] Evaluating on partially completed subset ({n_filled} issues).")

    df_eval = df_sec[filled_mask].copy()
    y_gold = []
    y_team = []

    for _, row in df_eval.iterrows():
        iid = row["issue_id"]
        if iid not in df_gold:
            continue
        g_lbl = df_gold[iid].strip()
        t_lbl = str(row["teammate_priority"]).strip()
        y_gold.append(g_lbl)
        y_team.append(t_lbl)

    if not y_gold:
        print("Error: No matching issue IDs found between gold and second annotator files.")
        return None

    # Calculate metrics
    raw_agreement = accuracy_score(y_gold, y_team)
    kappa_unweighted = cohen_kappa_score(y_gold, y_team, labels=PRIORITY_LEVELS)
    
    # Quadratic weighted kappa for ordinal levels
    try:
        kappa_weighted = cohen_kappa_score(y_gold, y_team, labels=PRIORITY_LEVELS, weights="quadratic")
    except Exception:
        kappa_weighted = None

    cm = confusion_matrix(y_gold, y_team, labels=PRIORITY_LEVELS)

    print("\n" + "-" * 70)
    print("AGREEMENT RESULTS")
    print("-" * 70)
    print(f"Raw Percent Agreement     : {raw_agreement:.2%}")
    print(f"Cohen's Kappa (Unweighted): {kappa_unweighted:.4f}")
    print(f"Interpretation            : {interpret_kappa(kappa_unweighted)}")
    if kappa_weighted is not None:
        print(f"Cohen's Kappa (Quadratic) : {kappa_weighted:.4f} (accounts for ordinal penalty)")

    print("\nConfusion Matrix (Rows = Primary Gold, Columns = Teammate):")
    print(f"{'':<12} " + "  ".join(f"{lvl:>8}" for lvl in PRIORITY_LEVELS))
    for idx, lvl in enumerate(PRIORITY_LEVELS):
        row_str = "  ".join(f"{cm[idx, c]:>8}" for c in range(len(PRIORITY_LEVELS)))
        print(f"{lvl:<12} {row_str}")

    if demo:
        print("\n" + "*" * 70)
        print("*** WARNING: DEMO DATA - DO NOT REPORT IN FINAL DOCUMENTATION OR PAPERS ***")
        print("*" * 70)

    print("=" * 70 + "\n")
    return {
        "n_evaluated": len(y_gold),
        "raw_agreement": raw_agreement,
        "kappa_unweighted": kappa_unweighted,
        "kappa_weighted": kappa_weighted,
        "confusion_matrix": cm
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Cohen's Kappa Inter-Annotator Agreement")
    parser.add_argument("--file", type=str, default=DEFAULT_SECOND_FILE, help="Path to second annotator CSV")
    parser.add_argument("--gold", type=str, default=DEFAULT_GOLD_FILE, help="Path to primary gold CSV")
    parser.add_argument("--demo", action="store_true", help="Simulate second annotator inputs to preview metrics")
    args = parser.parse_args()

    compute_inter_annotator_agreement(second_file=args.file, gold_file=args.gold, demo=args.demo)


if __name__ == "__main__":
    main()
