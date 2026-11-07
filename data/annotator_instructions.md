# GitHub Issue Intelligence
## Priority Annotation Instructions (Second Annotator)

Welcome to the independent priority annotation study for the **GitHub Issue Intelligence** project.

Your task is to independently assign priority labels to the **40 issues** listed in `data/gold_second_annotator.csv` to calculate double-blind inter-annotator agreement (Cohen's Kappa).

---

### Key Ground Rule: Submission-Time Assessment Only

> **IMPORTANT**:
> Assign priority **strictly based on the initial issue title and description as submitted by the user**. 
> - **Do NOT** research external maintainer discussions, pull request discussions, commit fixes, or external extension repositories.
> - In real-world triage, the triage engine evaluates raw user submissions before maintainers have investigated or resolved them. We are measuring **submission-time agreement**.

---

### Priority Definitions & Criteria

You must assign exactly one of the four categories to the `teammate_priority` column for each row:

#### 1. `Critical`
- **Definition**: Catastrophic failure affecting widespread users, core integrity failure, or severe vulnerability in core Flask.
- **Inclusion Criteria**:
  - Breaks fresh installs (`pip install flask` fails to install or import).
  - Crashes immediately on startup / application boot for standard configurations.
  - Remote Code Execution (RCE), authentication bypass, or direct exploitability in core Flask.
  - Silent data loss or irreversible database/session corruption.
- **Explicit Exclusions**:
  - A bug in a third-party extension (e.g. `Flask-Session`, `Flask-SQLAlchemy`, `cachelib`) is **NOT Critical**.
  - A bug in an example demo app or tutorial snippet is **NOT Critical**.
  - A routine dependency bump request or single-user environment misconfiguration is **NOT Critical**.

#### 2. `High`
- **Definition**: Major functional breakage in core Flask functionality affecting widespread users with no straightforward workaround.
- **Inclusion Criteria**:
  - Major regression introduced in a recent release that breaks existing working applications.
  - Core routing, request dispatching, or templating logic broken with no clean workaround.
  - High-severity security issues (e.g. cookie manipulation, token leakage in default logs).
- **Explicit Exclusions**:
  - Regressions or bugs that have simple, immediate one-line workarounds (these are Medium).

#### 3. `Medium`
- **Definition**: Functional bugs with existing workarounds, minor edge-case failures, or performance degradation.
- **Inclusion Criteria**:
  - Non-fatal bugs where a workaround exists (e.g. overriding a configuration key).
  - Edge cases occurring only under uncommon parameters, special character encodings, or specific platform variants.
  - Noticeable performance bottlenecks or memory leaks under specific non-default configurations.
  - Deprecation warnings emitted on modern Python runtimes.

#### 4. `Low`
- **Definition**: Cosmetic issues, documentation, user inquiries, and non-urgent enhancements.
- **Inclusion Criteria**:
  - Documentation fixes, typo corrections in docstrings or README files.
  - User support questions, "how-do-I" inquiries, and configuration questions.
  - Feature requests, enhancement proposals, and suggestions for new APIs.
  - Cosmetic refactoring, code cleanup, typing/mypy improvements, linting nits.
  - Isolated single-user local environment issues (e.g. user forgot to activate virtualenv).

---

### Annotation File Location & Schema

1. Open `data/gold_second_annotator.csv` (or `summarization/data/gold_second_annotator.csv`).
2. The columns are:
   - `issue_id`: Unique database identifier.
   - `issue_number`: GitHub issue number.
   - `title`: User-submitted issue title.
   - `description`: Raw Markdown body submitted by the user.
   - `teammate_priority`: **Fill this column with `Low`, `Medium`, `High`, or `Critical`.**
3. Save the CSV file after annotating all 40 rows.

---

### Verification

Once you have filled the column, verify inter-annotator agreement by executing:

```bash
python summarization/compute_kappa.py
```

This will automatically unblind your labels against the primary gold labels and report:
- Raw Percent Agreement (%)
- Unweighted Cohen's Kappa ($\kappa$)
- Quadratic-Weighted Cohen's Kappa ($\kappa_w$, accounting for ordinal penalties)
- Full confusion matrix
