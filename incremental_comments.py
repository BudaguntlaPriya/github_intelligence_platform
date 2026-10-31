import os
import requests
import psycopg2
from datetime import timedelta, timezone
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

# 3. Find the latest stored comment update time
cursor.execute("""
    SELECT MAX(updated_at)
    FROM github_issue_comments;
""")

last_updated = cursor.fetchone()[0]

if last_updated is None:
    raise ValueError("No existing comments found!")

since = last_updated - timedelta(minutes=5)

print("Fetching comments updated since:", since)

# 4. Fetch comments updated since that time
url = (
    f"https://api.github.com/repos/{owner}/"
    f"{repo}/issues/comments"
)

params = {
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
    comments = response.json()

    if not comments:
        break

    for comment in comments:

        issue_number = int(comment["issue_url"].split("/")[-1])

        cursor.execute(
        "SELECT issue_id FROM github_issues WHERE issue_number = %s",
        (issue_number,)
        )

        result = cursor.fetchone()

        if result is None:
            continue

        issue_id = result[0]

        records.append((
            comment["id"],
            issue_id,
            issue_number,
            comment["user"]["login"] if comment["user"] else "unknown",
            comment["body"] or "",
            comment["created_at"],
            comment["updated_at"],
            f"{owner}/{repo}"
        ))
    params["page"] += 1

print("New or updated comments fetched:", len(records))

# 5. Insert or update comments
if records:
    query = """
    INSERT INTO github_issue_comments
    (comment_id, issue_id, issue_number, username,
    comment_body, created_at, updated_at, repository)
    VALUES %s
    ON CONFLICT (comment_id)
    DO UPDATE SET
        issue_id = EXCLUDED.issue_id,
        issue_number = EXCLUDED.issue_number,
        username = EXCLUDED.username,
        comment_body = EXCLUDED.comment_body,
        updated_at = EXCLUDED.updated_at;
    """

    execute_values(cursor, query, records)
    conn.commit()

print("Incremental comment ingestion completed!")

cursor.close()
conn.close()