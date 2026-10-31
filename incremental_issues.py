import os
import requests
import pandas as pd
import psycopg2
from datetime import datetime, timedelta, timezone
from psycopg2.extras import execute_values

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

# 2. Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="github_issues_db",
    user="postgres",
    password="Priya2107"
)

cursor = conn.cursor()

# 3. Get latest updated_at timestamp
cursor.execute("""
    SELECT MAX(updated_at)
    FROM github_issues;
""")

last_updated = cursor.fetchone()[0]

# Subtract 5 minutes to avoid missing updates near the boundary
since = last_updated - timedelta(minutes=5)

print("Fetching issues updated since:", since)

# 4. Fetch updated issues from GitHub
url = f"https://api.github.com/repos/{owner}/{repo}/issues"

params = {
    "state": "all",
    "since": since.astimezone(timezone.utc).isoformat(),
    "per_page": 100,
    "page": 1
}

records = []

while True:
    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30
    )

    response.raise_for_status()
    issues = response.json()

    if not issues:
        break

    for issue in issues:

        # Skip pull requests
        if "pull_request" in issue:
            continue

        records.append((
            issue["id"],
            issue["number"],
            issue["title"],
            issue["body"] or "",
            issue["created_at"],
            issue["updated_at"],
            issue["state"],
            issue["comments"],
            ",".join(label["name"] for label in issue["labels"]),
            f"{owner}/{repo}"
        ))

    params["page"] += 1

print("New or updated issues fetched:", len(records))

# 5. Insert new issues and update existing issues
if records:

    query = """
    INSERT INTO github_issues
    (issue_id, issue_number, title, description, created_at,
     updated_at, state, comments, labels, repository)
    VALUES %s
    ON CONFLICT (issue_id)
    DO UPDATE SET
        title = EXCLUDED.title,
        description = EXCLUDED.description,
        updated_at = EXCLUDED.updated_at,
        state = EXCLUDED.state,
        comments = EXCLUDED.comments,
        labels = EXCLUDED.labels;
    """

    execute_values(cursor, query, records)

    conn.commit()

print("Incremental ingestion completed!")

cursor.close()
conn.close()
