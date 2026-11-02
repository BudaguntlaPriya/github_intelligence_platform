"""
GitHub Issue Label Recommender
 
Uses the hybrid classifier from issue_classification.py:
    1. Strong rule-based matching for obvious issues.
    2. TF-IDF + Logistic Regression for less obvious issues.
 
Usage (from the classification folder):
    python label_recommender.py                    # samples, then interactive mode
    python label_recommender.py --no-samples       # interactive mode only
    python label_recommender.py --title "App crashes" --desc "Throws an exception"
"""
 
import argparse
 
from issue_classification import classify_issue, load_bundle
 
LINE = "=" * 70
THIN = "-" * 70
 
 
def format_confidence(result):
    if result["confidence"] is None:
        return "N/A (strong rule matched)"
    return f"{result['confidence'] * 100:.2f}%"
 
 
def print_result(title, description, result):
    print(f"Title             : {title}")
    if description:
        print(f"Description       : {description}")
    print(f"Predicted Category: {result['category']}")
    print(f"Suggested Label   : {result['label']}")
    print(f"Method            : {result['method']}")
    if result.get("matched"):
        print(f"Matched Keyword   : {result['matched']}")
    print(f"Confidence        : {format_confidence(result)}")
    print(f"Confidence Level  : {result['level']}")
    if result.get("top3"):
        others = ", ".join(f"{c} {p * 100:.1f}%" for c, p in result["top3"])
        print(f"Top Predictions   : {others}")
    if result["level"] == "Low":
        print("Note              : Low confidence - please review this label manually.")
 
 
def run_samples(bundle):
    samples = [
        ("Application crashes when running request", "The application throws an exception."),
        ("Add support for new API", ""),
        ("Improve Flask documentation", ""),
        ("Security vulnerability in authentication", ""),
        ("Application is very slow", ""),
        ("How can I configure Flask?", ""),
    ]
    print("\n" + LINE)
    print("SAMPLE LABEL RECOMMENDATIONS")
    print(LINE)
    for title, desc in samples:
        print(THIN)
        print_result(title, desc, classify_issue(title, desc, bundle))
    print(THIN)
 
 
def interactive(bundle):
    print("\n" + LINE)
    print("TRY YOUR OWN GITHUB ISSUE")
    print("Leave the title empty to exit")
    print(LINE)
    while True:
        try:
            title = input("\nIssue title       : ").strip()
            if not title:
                print("No title entered. Exiting interactive mode.")
                break
            description = input("Issue description : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode.")
            break
 
        result = classify_issue(title, description, bundle)
        print("\n" + LINE)
        print("PREDICTION RESULT")
        print(LINE)
        print_result(title, description, result)
        print(LINE)
 
 
def main():
    parser = argparse.ArgumentParser(description="GitHub Issue Label Recommender")
    parser.add_argument("--title", help="Issue title (single prediction, non-interactive)")
    parser.add_argument("--desc", default="", help="Issue description")
    parser.add_argument("--no-samples", action="store_true", help="Skip the sample predictions")
    args = parser.parse_args()
 
    print(LINE)
    print("GITHUB ISSUE LABEL RECOMMENDER")
    print(LINE)
 
    bundle = load_bundle()
 
    if args.title is not None:
        result = classify_issue(args.title, args.desc, bundle)
        if result is None:
            print("No input provided - no prediction made.")
        else:
            print_result(args.title, args.desc, result)
        return
 
    if not args.no_samples:
        run_samples(bundle)
    interactive(bundle)
 
    print("\n" + LINE)
    print("LABEL RECOMMENDATION COMPLETED")
    print(LINE)
 
 
if __name__ == "__main__":
    main()