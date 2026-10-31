import pandas as pd
import psycopg2

# Read cleaned CSV file
df = pd.read_csv("cleaned_issues.csv")

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="github_issues_db",
    user="postgres",
    password="Priya2107",
    port="5432"
)

cursor = conn.cursor()

# Insert records into PostgreSQL
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO github_issues (
            issue_id, issue_number, title, description,
            created_at, updated_at, state, comments,
            labels, repository
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (issue_id) DO NOTHING
    """, (
        int(row["issue_id"]),
        int(row["issue_number"]),
        row["title"],
        row["description"],
        row["created_at"],
        row["updated_at"],
        row["state"],
        int(row["comments"]),
        row["labels"],
        row["repository"]
    ))

conn.commit()
cursor.close()
conn.close()

print("Data successfully loaded into PostgreSQL!")