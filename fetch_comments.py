
import os
import time
import requests
import pandas as pd

# 1. GitHub authentication
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError("GitHub token not found!")

headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

owner = "pallets"
repo = "flask"

# 2. Read cleaned issues
df = pd.read_csv("cleaned_issues.csv")
df["comments"] = pd.to_numeric(df["comments"], errors="coerce").fillna(0)

issues = df[df["comments"] > 0]

print("Total issues to process:", len(issues))

# 3. Resume from previously saved comments
output_file = "raw_comments.csv"

if os.path.exists(output_file):
    old_df = pd.read_csv(output_file)

    if not old_df.empty:
        collected_ids = set(old_df["comment_id"].astype(str))
        records = old_df.to_dict("records")
    else:
        collected_ids = set()
        records = []
else:
    collected_ids = set()
    records = []

print("Previously saved comments:", len(collected_ids))

# 4. Fetch comments issue by issue
for index, issue in issues.iterrows():

    issue_number = int(issue["issue_number"])
    url = (
        f"https://api.github.com/repos/{owner}/"
        f"{repo}/issues/{issue_number}/comments"
    )

    page = 1
    issue_comments = 0

    while True:

        params = {
            "per_page": 100,
            "page": page
        }

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30
        )

        if response.status_code == 403:
            print("Rate limit reached. Saving progress.")
            pd.DataFrame(records).to_csv(output_file, index=False)
            raise SystemExit("Stopped safely. Resume after rate limit resets.")

        response.raise_for_status()

        comments = response.json()

        if not comments:
            break

        for comment in comments:

            comment_id = str(comment["id"])

            if comment_id in collected_ids:
                continue

            records.append({
                "comment_id": comment["id"],
                "issue_number": issue_number,
                "issue_id": issue["issue_id"],
                "username": (
                    comment["user"]["login"]
                    if comment["user"] else "unknown"
                ),
                "comment_body": comment["body"] or "",
                "created_at": comment["created_at"],
                "updated_at": comment["updated_at"],
                "repository": f"{owner}/{repo}"
            })

            collected_ids.add(comment_id)
            issue_comments += 1

        page += 1

    # Save after every issue
    pd.DataFrame(records).to_csv(output_file, index=False)

    print(
        f"Processed issue {issue_number} | "
        f"New comments: {issue_comments} | "
        f"Total saved: {len(records)}"
    )

    time.sleep(0.2)

print("\n========== COMMENT SUMMARY ==========")
print("Total comments collected:", len(records))
print("Saved as:", output_file)
print("=====================================")
