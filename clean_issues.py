import pandas as pd

# Read raw GitHub issues
df = pd.read_csv("raw_issues.csv")

print("Original records:", len(df))

# Remove duplicate issues
df = df.drop_duplicates(subset=["issue_id"])

# Remove missing or empty titles
df = df.dropna(subset=["title"])
df["title"] = df["title"].astype(str).str.strip()
df = df[df["title"] != ""]

# Handle missing descriptions
df["description"] = df["description"].fillna("")

# Handle missing labels
df["labels"] = df["labels"].fillna("")

# Save cleaned data
df.to_csv("cleaned_issues.csv", index=False)

print("Cleaned records:", len(df))
print("Duplicate IDs:", df["issue_id"].duplicated().sum())
print("Missing titles:", df["title"].isna().sum())
print("Saved as cleaned_issues.csv")