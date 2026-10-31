import pandas as pd

df = pd.read_csv("raw_comments.csv")

print("Total comments:", len(df))
print("Duplicate comment IDs:", df["comment_id"].duplicated().sum())
print("Missing comment bodies:", df["comment_body"].isna().sum())

print("\nFirst 5 comments:")
print(df[["issue_number", "username", "comment_body"]].head())
