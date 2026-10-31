# GitHub Issue Intelligence
## Member 1 – Data Collection and Preprocessing Documentation

### 1. Introduction

GitHub Issue Intelligence is a project designed to analyse GitHub repository issues using data processing and machine learning techniques.

The system collects GitHub issues and their comments, processes the collected information, and stores the structured data in PostgreSQL. This data is used by other team members for issue classification, priority prediction, sentiment analysis, duplicate detection, and issue summarisation.

My responsibility as Member 1 is to develop the data collection and preprocessing pipeline.

### 2. Objectives

The main objectives of my contribution are:

- Collect GitHub issues using the GitHub REST API.
- Extract issue titles, descriptions, timestamps, labels, and other details.
- Collect comments associated with GitHub issues.
- Clean and preprocess the collected datasets.
- Store issue and comment data in PostgreSQL.
- Develop incremental data ingestion to fetch new and updated records.
- Validate the stored data and maintain relationships between issues and comments.

### 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Data collection and processing |
| GitHub REST API | Fetch issues and comments |
| Requests | API communication |
| Pandas | Data cleaning and CSV processing |
| PostgreSQL 16 | Database storage |
| Psycopg2 | Python–PostgreSQL connection |
| VS Code | Development environment |

### 4. GitHub Data Collection

The public repository selected for data collection is `pallets/flask`.

The GitHub REST API was used to retrieve issue information.

The collection script is named `fetch_issues.py`.

The script collects:
- Issue ID
- Issue number
- Title
- Description
- Created timestamp
- Updated timestamp
- Issue state
- Number of comments
- Labels
- Repository name

Pagination was implemented using the API's page parameter to collect multiple pages of issues.

Pull requests were excluded because the project focuses on GitHub issues.

**Result:** 2,769 GitHub issues were collected and saved in `raw_issues.csv`.

### 5. Data Cleaning and Preprocessing

The collected issue dataset was cleaned using Pandas.

The cleaning operations included:

1. Removing duplicate issue IDs.
2. Removing records with missing titles.
3. Removing empty titles.
4. Replacing missing descriptions with empty strings.
5. Replacing missing labels with empty strings.
6. Saving the cleaned dataset as `cleaned_issues.csv`.

**Validation results:**

| Validation | Result |
|---|---:|
| Original records | 2,769 |
| Cleaned records | 2,769 |
| Duplicate issue IDs | 0 |
| Missing titles | 0 |

### 6. GitHub Comment Collection

Issue comments provide additional information about technical problems, discussions, troubleshooting, and possible solutions.

A separate script named `fetch_comments.py` was developed to collect actual comment content.

The collected fields include:

- Comment ID
- Issue ID
- Issue number
- Username
- Comment body
- Created timestamp
- Updated timestamp
- Repository name

The script uses pagination and saves progress to `raw_comments.csv`.

**Result:** 3,668 comments were collected.

Comment validation showed:
- Duplicate comment IDs: 0
- Missing comment bodies: 0

### 7. PostgreSQL Database Design

A PostgreSQL database named `github_issues_db` was created.

Two main tables were designed.

**Table 1: github_issues**

Stores the collected GitHub issue information.

Main columns:
- issue_id – Primary key
- issue_number
- title
- description
- created_at
- updated_at
- state
- comments
- labels
- repository

**Table 2: github_issue_comments**

Stores the actual comments associated with issues.

Main columns:
- comment_id – Primary key
- issue_id – Foreign key referencing github_issues
- issue_number
- username
- comment_body
- created_at
- updated_at
- repository

The foreign key relationship connects each comment to its original issue.

### 8. Loading Data into PostgreSQL

Python and Psycopg2 were used to load the cleaned CSV datasets into PostgreSQL.

The scripts developed were:

- `load_to_postgres.py`
- `load_comments_to_postgres.py`

The SQL `ON CONFLICT DO NOTHING` operation was used during initial loading to prevent duplicate records.

**Database loading results:**

| Table | Records loaded |
|---|---:|
| github_issues | 2,769 |
| github_issue_comments | 3,668 |

### 9. Incremental Data Ingestion

Incremental ingestion was implemented to avoid downloading the entire dataset every time the pipeline runs.

**Issue incremental ingestion**

The script `incremental_issues.py` retrieves the latest stored `updated_at` timestamp from PostgreSQL.

It uses the GitHub API's `since` parameter to fetch issues updated after that timestamp, with a five-minute overlap to reduce the chance of missing updates near the boundary.

New records are inserted, while existing records are updated using `ON CONFLICT DO UPDATE`.

During testing, the script fetched 3 new or updated issue records.

**Comment incremental ingestion**

The script `incremental_comments.py` retrieves the latest stored comment update timestamp.

It fetches new or updated comments using the GitHub issue comments API and the `since` parameter.

The script identifies the parent issue and updates or inserts the corresponding comment record in PostgreSQL.

During testing, 1 new or updated comment was fetched.

The total database counts remained unchanged because the fetched records could be updates to existing records.

### 10. Final Data Validation

SQL queries were used to verify the database records and relationships.

| Validation | Result |
|---|---:|
| Total GitHub issues | 2,769 |
| Total GitHub comments | 3,668 |
| Comments linked to issues | 3,668 |
| Orphan comments | 0 |

The orphan-comment check used a LEFT JOIN between `github_issue_comments` and `github_issues`.

The result of zero orphan comments confirms that all collected comments are connected to existing issues.

### 11. Project Files

The main files developed for Member 1 are:

| File | Responsibility |
|---|---|
| fetch_issues.py | Fetch GitHub issues |
| clean_issues.py | Clean issue data |
| load_to_postgres.py | Load issues into PostgreSQL |
| fetch_comments.py | Fetch issue comments |
| check_comments.py | Validate comment data |
| load_comments_to_postgres.py | Load comments into PostgreSQL |
| incremental_issues.py | Fetch new and updated issues |
| incremental_comments.py | Fetch new and updated comments |

Dataset files:
- raw_issues.csv
- cleaned_issues.csv
- raw_comments.csv

### 12. Conclusion

As Member 1, I developed the data collection and preprocessing pipeline for the GitHub Issue Intelligence project.

The pipeline collects GitHub issues and comments through the GitHub REST API, cleans the collected information, and stores it in PostgreSQL.

Incremental ingestion was also implemented to support fetching new and updated records without repeatedly downloading the complete dataset.

The final database contains 2,769 issues and 3,668 comments, with all comments successfully linked to their corresponding issues.

This prepared dataset provides the foundation for the machine learning, duplicate detection, priority prediction, and summarisation components developed by the other team members.
