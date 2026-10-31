
import os
import requests
import pandas as pd
import time

# 1. Get GitHub token securely from environment variable
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError(
        "GitHub token not found! Set GITHUB_TOKEN in your terminal."
    )

print("GitHub token loaded successfully!")

# 2. GitHub API authentication headers
headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

# 3. Repository details
owner = "pallets"
repo = "flask"

url = f"https://api.github.com/repos/{owner}/{repo}/issues"

# 4. Store collected issues
records = []
page = 1

# 5. Fetch issues using pagination
while True:

    params = {
        "state": "all",
        "per_page": 100,
        "page": page
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30
    )

    # Check API response
    if response.status_code == 403:
        print("GitHub API rate limit or permission issue.")
        print("Rate limit remaining:",
              response.headers.get("X-RateLimit-Remaining"))
        print("Reset time:",
              response.headers.get("X-RateLimit-Reset"))
        break

    response.raise_for_status()

    issues = response.json()

    # Stop when no more issues are available
    if not issues:
        break

    print(f"Processing page {page}...")

    for issue in issues:

        # Skip pull requests
        if "pull_request" in issue:
            continue

        records.append({
            "issue_id": issue["id"],
            "issue_number": issue["number"],
            "title": issue["title"],
            "description": issue["body"] or "",
            "created_at": issue["created_at"],
            "updated_at": issue["updated_at"],
            "state": issue["state"],
            "comments": issue["comments"],
            "labels": ", ".join(
                label["name"] for label in issue["labels"]
            ),
            "repository": f"{owner}/{repo}"
        })

    print(f"Page {page}: {len(issues)} API records fetched")

    page += 1

    # Small delay between requests
    time.sleep(1)

# 6. Convert records into DataFrame
df = pd.DataFrame(records)

# 7. Remove duplicate issues
if not df.empty:
    df = df.drop_duplicates(subset=["issue_id"])

# 8. Save data into CSV
df.to_csv("raw_issues.csv", index=False)

# 9. Display results
print("\n========== FETCH SUMMARY ==========")
print("Total issues collected:", len(df))
print("Repository:", f"{owner}/{repo}")
print("CSV file saved: raw_issues.csv")
print("===================================")
