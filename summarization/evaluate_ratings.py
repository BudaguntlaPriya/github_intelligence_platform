"""
Blinded Summary Evaluation Unblinder and Analysis Script
Member 4: Summarization + Priority Prediction

Unblinds A/B summary ratings from summary_ratings.csv using
summary_ratings_key.csv, computing mean criteria scores per method,
head-to-head win/tie/loss counts, and inter-rater agreement statistics
strictly on rows rated by both raters.

Usage:
    python evaluate_ratings.py [--ratings path/to/summary_ratings.csv]
                               [--key path/to/summary_ratings_key.csv]
                               [--demo]
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
from scipy.stats import pearsonr, spearmanr, binomtest
from sklearn.metrics import cohen_kappa_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RATINGS_FILE = os.path.join(BASE_DIR, "data", "summary_ratings.csv")
DEFAULT_KEY_FILE = os.path.join(BASE_DIR, "data", "summary_ratings_key.csv")
CRITERIA = ["faithfulness", "coverage", "readability"]


def evaluate_summary_ratings(ratings_file=DEFAULT_RATINGS_FILE, key_file=DEFAULT_KEY_FILE, demo=False):
    """Unblind and analyze summary ratings."""
    if not os.path.exists(ratings_file):
        alt = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "summary_ratings.csv"))
        if os.path.exists(alt):
            ratings_file = alt
        else:
            raise FileNotFoundError(f"Ratings file not found: {ratings_file}")

    if not os.path.exists(key_file):
        alt_key = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "summary_ratings_key.csv"))
        if os.path.exists(alt_key):
            key_file = alt_key
        else:
            raise FileNotFoundError(f"Key file not found: {key_file}")

    df_ratings = pd.read_csv(ratings_file)
    df_key = pd.read_csv(key_file).set_index("issue_id")

    print("=" * 75)
    print("BLINDED SUMMARY EVALUATION (rater1 = AI judge, rater2 = human if provided)")
    print("=" * 75)
    print(f"Ratings file: {ratings_file}")
    print(f"Key file    : {key_file}")
    print(f"Total Issues: {len(df_ratings)}")

    if demo:
        print("\n" + "*" * 75)
        print("*** WARNING: DEMO DATA - DO NOT REPORT IN FINAL DOCUMENTATION OR PAPERS ***")
        print("*" * 75 + "\n")
        print("[DEMO MODE ACTIVATED] Simulating Rater 2 (human) ratings to preview report...")
        rng = np.random.RandomState(42)
        demo_rows = []
        for _, row in df_ratings.iterrows():
            iid = row["issue_id"]
            krow = df_key.loc[iid]
            tr_is_a = (krow["summary_a_method"] == "textrank")
            row = row.copy()
            
            # Check if rater 1 is already populated
            r1_has_scores = pd.notna(pd.to_numeric(row.get("rater1_a_faithfulness"), errors="coerce"))
            
            if not r1_has_scores:
                # Base distribution for Rater 1 (AI judge)
                r1_tr_f = int(np.clip(rng.choice([4, 5, 5, 5, 5]), 1, 5))
                r1_tr_c = int(np.clip(rng.choice([4, 4, 5, 5, 4]), 1, 5))
                r1_tr_r = int(np.clip(rng.choice([4, 4, 5, 4, 4]), 1, 5))

                r1_l3_f = int(np.clip(rng.choice([4, 5, 5, 4, 5]), 1, 5))
                r1_l3_c = int(np.clip(rng.choice([3, 4, 3, 4, 5]), 1, 5))
                r1_l3_r = int(np.clip(rng.choice([3, 4, 4, 5, 3]), 1, 5))
                
                if tr_is_a:
                    row["rater1_a_faithfulness"], row["rater1_a_coverage"], row["rater1_a_readability"] = r1_tr_f, r1_tr_c, r1_tr_r
                    row["rater1_b_faithfulness"], row["rater1_b_coverage"], row["rater1_b_readability"] = r1_l3_f, r1_l3_c, r1_l3_r
                else:
                    row["rater1_b_faithfulness"], row["rater1_b_coverage"], row["rater1_b_readability"] = r1_tr_f, r1_tr_c, r1_tr_r
                    row["rater1_a_faithfulness"], row["rater1_a_coverage"], row["rater1_a_readability"] = r1_l3_f, r1_l3_c, r1_l3_r
            
            # Rater 2 correlates with Rater 1 with slight noise (±1 jitter)
            jitter = lambda x: int(np.clip(x + rng.choice([-1, 0, 0, 0, 1]), 1, 5))
            v1_a_f = int(float(row["rater1_a_faithfulness"]))
            v1_a_c = int(float(row["rater1_a_coverage"]))
            v1_a_r = int(float(row["rater1_a_readability"]))
            v1_b_f = int(float(row["rater1_b_faithfulness"]))
            v1_b_c = int(float(row["rater1_b_coverage"]))
            v1_b_r = int(float(row["rater1_b_readability"]))

            row["rater2_a_faithfulness"] = jitter(v1_a_f)
            row["rater2_a_coverage"]     = jitter(v1_a_c)
            row["rater2_a_readability"]  = jitter(v1_a_r)
            row["rater2_b_faithfulness"] = jitter(v1_b_f)
            row["rater2_b_coverage"]     = jitter(v1_b_c)
            row["rater2_b_readability"]  = jitter(v1_b_r)

            demo_rows.append(row)
        df_ratings = pd.DataFrame(demo_rows)

    # Check if ratings are filled
    rating_cols = [
        "rater1_a_faithfulness", "rater1_a_coverage", "rater1_a_readability",
        "rater1_b_faithfulness", "rater1_b_coverage", "rater1_b_readability",
        "rater2_a_faithfulness", "rater2_a_coverage", "rater2_a_readability",
        "rater2_b_faithfulness", "rater2_b_coverage", "rater2_b_readability"
    ]
    filled = df_ratings[rating_cols].dropna(how="all")
    if len(filled) == 0:
        print("\n" + "!" * 75)
        print("ACTION REQUIRED: Summary ratings are currently pending.")
        print("Evaluators must open 'data/summary_ratings.csv' and score")
        print("Summary A and Summary B on a 1-5 Likert scale across:")
        print("  - Faithfulness (1-5): Factual consistency with source description")
        print("  - Coverage     (1-5): Captures core problem and key details")
        print("  - Readability  (1-5): Cohesiveness, grammar, flow")
        print("Once filled, re-run this script to unblind and compute method benchmarks.")
        print("Tip: Run 'python evaluate_ratings.py --demo' to preview the full report.")
        print("!" * 75 + "\n")
        return None

    # Unblind ratings into structured tables
    scores = {"textrank": {"rater1": {c: [] for c in CRITERIA}, "rater2": {c: [] for c in CRITERIA}},
              "lead3":    {"rater1": {c: [] for c in CRITERIA}, "rater2": {c: [] for c in CRITERIA}}}

    # Track head-to-head win/tie/loss counts per criterion for TextRank vs Lead-3
    # tr_vs_l3[rater][criterion] = {"win": count, "tie": count, "loss": count}
    tr_vs_l3 = {
        "rater1": {c: {"win": 0, "tie": 0, "loss": 0} for c in CRITERIA},
        "rater2": {c: {"win": 0, "tie": 0, "loss": 0} for c in CRITERIA}
    }

    # Track which rows are rated by both raters
    both_rated_row_count = 0

    for _, row in df_ratings.iterrows():
        iid = row["issue_id"]
        if iid not in df_key.index:
            continue
        krow = df_key.loc[iid]
        method_a = krow["summary_a_method"]
        method_b = krow["summary_b_method"]
        tr_is_a = (method_a == "textrank")

        row_has_both = True

        for c in CRITERIA:
            # Rater 1
            v_a1 = pd.to_numeric(row.get(f"rater1_a_{c}"), errors="coerce")
            v_b1 = pd.to_numeric(row.get(f"rater1_b_{c}"), errors="coerce")
            if pd.notna(v_a1):
                scores[method_a]["rater1"][c].append(float(v_a1))
            if pd.notna(v_b1):
                scores[method_b]["rater1"][c].append(float(v_b1))

            if pd.notna(v_a1) and pd.notna(v_b1):
                s_tr1 = float(v_a1) if tr_is_a else float(v_b1)
                s_l31 = float(v_b1) if tr_is_a else float(v_a1)
                if s_tr1 > s_l31:
                    tr_vs_l3["rater1"][c]["win"] += 1
                elif s_tr1 == s_l31:
                    tr_vs_l3["rater1"][c]["tie"] += 1
                else:
                    tr_vs_l3["rater1"][c]["loss"] += 1

            # Rater 2
            v_a2 = pd.to_numeric(row.get(f"rater2_a_{c}"), errors="coerce")
            v_b2 = pd.to_numeric(row.get(f"rater2_b_{c}"), errors="coerce")
            if pd.notna(v_a2):
                scores[method_a]["rater2"][c].append(float(v_a2))
            if pd.notna(v_b2):
                scores[method_b]["rater2"][c].append(float(v_b2))

            if pd.notna(v_a2) and pd.notna(v_b2):
                s_tr2 = float(v_a2) if tr_is_a else float(v_b2)
                s_l32 = float(v_b2) if tr_is_a else float(v_a2)
                if s_tr2 > s_l32:
                    tr_vs_l3["rater2"][c]["win"] += 1
                elif s_tr2 == s_l32:
                    tr_vs_l3["rater2"][c]["tie"] += 1
                else:
                    tr_vs_l3["rater2"][c]["loss"] += 1

            # Check if this criterion was rated by both raters on both summaries
            if not (pd.notna(v_a1) and pd.notna(v_b1) and pd.notna(v_a2) and pd.notna(v_b2)):
                row_has_both = False

        if row_has_both:
            both_rated_row_count += 1

    has_r2 = any(len(scores["textrank"]["rater2"][c]) > 0 for c in CRITERIA)
    table_label = "[AI judge (rater1) & Human (rater2)]" if has_r2 else "[AI judge (single rater)]"

    # 1. Unblinded Quality Benchmarks
    print("\n" + "-" * 75)
    print(f"1. UNBLINDED METHOD QUALITY BENCHMARKS {table_label} (1-5 LIKERT SCALE)")
    print("-" * 75)
    print(f"{'Method':<12} {'Criterion':<14} {'Rater 1 (AI)':<14} {'Rater 2 (Human)':<16} {'Overall Mean ± SD':<18}")
    print("-" * 75)

    method_summary = {}
    for m in ["textrank", "lead3"]:
        m_display = "TextRank" if m == "textrank" else "Lead-3"
        method_summary[m] = {}
        for c in CRITERIA:
            r1_vals = scores[m]["rater1"][c]
            r2_vals = scores[m]["rater2"][c]
            all_vals = r1_vals + r2_vals
            r1_mean = f"{np.mean(r1_vals):.2f}" if r1_vals else "N/A"
            r2_mean = f"{np.mean(r2_vals):.2f}" if r2_vals else "Pending"
            pooled_mean = np.mean(all_vals) if all_vals else np.nan
            pooled_std = np.std(all_vals) if all_vals else np.nan
            method_summary[m][c] = pooled_mean
            pooled_str = f"{pooled_mean:.2f} ± {pooled_std:.2f}" if pd.notna(pooled_mean) else "N/A"
            print(f"{m_display:<12} {c.capitalize():<14} {r1_mean:<14} {r2_mean:<16} {pooled_str:<18}")
        print()

    # 2. Head-to-Head Win / Tie / Loss
    print("-" * 75)
    print(f"2. HEAD-TO-HEAD METHOD COMPARISON {table_label} (TEXTRANK VS LEAD-3)")
    print("-" * 75)

    if has_r2:
        print(f"{'Criterion':<14} {'Rater 1 (W/T/L)':<18} {'Rater 2 (W/T/L)':<18} {'Combined Win Rate':<20} {'Combined Sign Test p':<20}")
        print("-" * 75)
        for c in CRITERIA:
            w1, t1, l1 = tr_vs_l3["rater1"][c]["win"], tr_vs_l3["rater1"][c]["tie"], tr_vs_l3["rater1"][c]["loss"]
            w2, t2, l2 = tr_vs_l3["rater2"][c]["win"], tr_vs_l3["rater2"][c]["tie"], tr_vs_l3["rater2"][c]["loss"]
            r1_str = f"{w1:>2}W / {t1:>2}T / {l1:>2}L"
            r2_str = f"{w2:>2}W / {t2:>2}T / {l2:>2}L"
            total_wins = w1 + w2
            total_losses = l1 + l2
            decisive = total_wins + total_losses
            win_rate = (total_wins / decisive * 100) if decisive > 0 else 50.0
            try:
                p_val = binomtest(total_wins, decisive, 0.5, alternative="two-sided").pvalue if decisive > 0 else 1.0
                p_str = f"{p_val:.4f}"
            except Exception:
                p_str = "N/A"
            print(f"{c.capitalize():<14} {r1_str:<18} {r2_str:<18} {win_rate:>5.1f}% ({total_wins}/{decisive})   {p_str:<20}")
    else:
        print(f"{'Criterion':<14} {'TextRank Win':<14} {'Tie':<8} {'Lead-3 Win (Loss)':<18} {'TR Win Rate':<14} {'Sign Test p':<12}")
        print("-" * 75)
        for c in CRITERIA:
            w, t, l = tr_vs_l3["rater1"][c]["win"], tr_vs_l3["rater1"][c]["tie"], tr_vs_l3["rater1"][c]["loss"]
            decisive = w + l
            win_rate = (w / decisive * 100) if decisive > 0 else 50.0
            try:
                p_val = binomtest(w, decisive, 0.5, alternative="two-sided").pvalue if decisive > 0 else 1.0
                p_str = f"{p_val:.4f}"
            except Exception:
                p_str = "N/A"
            print(f"{c.capitalize():<14} {w:<14} {t:<8} {l:<18} {win_rate:>5.1f}% ({w}/{decisive}) {p_str:<12}")
    print()

    # 3. Inter-Rater Agreement (Strictly on rows rated by both raters)
    print("-" * 75)
    print("3. INTER-RATER AGREEMENT (COMPUTED ON ROWS RATED BY BOTH RATERS)")
    print("-" * 75)
    print(f"Count of rows rated by both raters: {both_rated_row_count} of {len(df_ratings)} issues")

    agreement_results = {}
    if both_rated_row_count >= 5:
        print(f"{'Criterion':<14} {'Paired n':<10} {'Exact Agree':<14} {'Within 1 Point':<16} {'Pearson r':<12} {'Weighted Kappa':<14}")
        print("-" * 75)
        for c in CRITERIA:
            r1_paired, r2_paired = [], []
            for _, row in df_ratings.iterrows():
                for slot in ["a", "b"]:
                    v1 = pd.to_numeric(row.get(f"rater1_{slot}_{c}"), errors="coerce")
                    v2 = pd.to_numeric(row.get(f"rater2_{slot}_{c}"), errors="coerce")
                    if pd.notna(v1) and pd.notna(v2):
                        r1_paired.append(int(v1))
                        r2_paired.append(int(v2))

            r1_arr = np.array(r1_paired)
            r2_arr = np.array(r2_paired)
            exact = np.mean(r1_arr == r2_arr)
            within1 = np.mean(np.abs(r1_arr - r2_arr) <= 1)
            try:
                pr, _ = pearsonr(r1_arr, r2_arr)
            except Exception:
                pr = np.nan
            try:
                wk = cohen_kappa_score(r1_arr, r2_arr, labels=[1, 2, 3, 4, 5], weights="quadratic")
            except Exception:
                wk = np.nan
            agreement_results[c] = {"exact": exact, "within1": within1, "pearson": pr, "weighted_kappa": wk}
            print(f"{c.capitalize():<14} {len(r1_arr):<10} {exact:<14.1%} {within1:<16.1%} {pr:<12.3f} {wk:<14.3f}")
    else:
        print(f"Agreement not evaluable (requires at least 5 issues rated by both raters; currently n={both_rated_row_count}).")
        print("Note: Rater 2 (human) ratings are currently pending in 'data/summary_ratings.csv'.")

    print("=" * 75 + "\n")
    return {"methods": method_summary, "agreement": agreement_results, "tr_vs_l3": tr_vs_l3}


def main():
    parser = argparse.ArgumentParser(description="Unblind and Evaluate Summary Ratings")
    parser.add_argument("--ratings", type=str, default=DEFAULT_RATINGS_FILE, help="Path to summary ratings CSV")
    parser.add_argument("--key", type=str, default=DEFAULT_KEY_FILE, help="Path to summary ratings key CSV")
    parser.add_argument("--demo", action="store_true", help="Simulate sample human ratings to preview analysis")
    args = parser.parse_args()

    evaluate_summary_ratings(ratings_file=args.ratings, key_file=args.key, demo=args.demo)


if __name__ == "__main__":
    main()
