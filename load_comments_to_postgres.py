import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Read comments CSV
df = pd.read_csv("raw_comments.csv")

# Replace missing values
df = df.where(pd.notnull(df), None)

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="github_issues_db",
    user="postgres",
    password="Priya2107",
)

cursor = conn.cursor()

# Prepare records
records = [
    (
        row.comment_id,
        row.issue_id,
        row.issue_number,
        row.username,
        row.comment_body,
        row.created_at,
        row.updated_at,
        row.repository
    )
    for row in df.itertuples(index=False)
]

# Insert comments
query = """
INSERT INTO github_issue_comments
(comment_id, issue_id, issue_number, username,
 comment_body, created_at, updated_at, repository)
VALUES %s
ON CONFLICT (comment_id) DO NOTHING;
"""

execute_values(cursor, query, records)

conn.commit()

print("Total comments in CSV:", len(df))
print("Comments loaded successfully!")

cursor.close()
conn.close()
