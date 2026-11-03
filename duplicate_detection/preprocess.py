import html
import re


_BOILERPLATE_INSTRUCTIONS = (
    "please provide a clear and concise description",
    "please describe the expected behavior",
    "please describe the actual behavior",
    "steps to reproduce the behavior",
    "before submitting this issue",
    "please check existing issues",
    "thank you for opening an issue",
    "delete this section if not applicable",
    "replace this text with your answer",
    "write your answer here",
)

_EMPTY_TEMPLATE_HEADINGS = {
    "description",
    "additional context",
    "checklist",
    "screenshots",
    "logs",
}


def remove_template_boilerplate(text):
    """Remove empty template prompts without discarding issue details."""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    cleaned_lines = []

    for line in text.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue

        heading = re.sub(r"^#{1,6}\s*", "", line_clean).strip().rstrip(":")
        if heading.casefold() in _EMPTY_TEMPLATE_HEADINGS:
            continue

        normalized_line = line_clean.casefold().strip(" -*_`:")
        if any(
            normalized_line == phrase
            or normalized_line.startswith(phrase + ":")
            for phrase in _BOILERPLATE_INSTRUCTIONS
        ):
            continue

        if re.fullmatch(
            r"[-*]?\s*(python|flask|werkzeug)\s+version\s*:\s*"
            r"(n/?a|none|unknown)?",
            line_clean,
            flags=re.IGNORECASE,
        ):
            continue

        cleaned_lines.append(line_clean)

    return "\n".join(cleaned_lines)


def clean_text(text):
    """Normalize markup and whitespace while preserving issue content."""
    if text is None:
        return ""

    text = html.unescape(str(text))
    text = remove_template_boilerplate(text)

    # Keep both the readable label and target because URLs can be issue evidence.
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r"\1 \2",
        text,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"```[\w+-]*", " ", text)
    text = text.replace("`", " ")
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # This known concatenation occurs in prose; keep the original casing.
    text = re.sub(r"\b(The)(?=Flask\b)", r"\1 ", text, flags=re.IGNORECASE)

    cleaned_text = " ".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )
    return re.sub(r"\s+", " ", cleaned_text).strip()


def build_issue_text(title, description):
    """Combine the cleaned title and description, omitting empty sections."""
    cleaned_title = clean_text(title)
    cleaned_description = clean_text(description)
    sections = []

    if cleaned_title:
        sections.append(f"Title: {cleaned_title}")
    if cleaned_description:
        sections.append(f"Description: {cleaned_description}")

    return ". ".join(sections)


if __name__ == "__main__":
    sample_title = "Flask application is not working"
    sample_description = """
### Expected Behavior

The application should start successfully.

### Actual Behavior

The Flask application crashes during startup.

### Environment

- Python version: 3.10
- Flask version: 2.2.5
"""
    print(build_issue_text(sample_title, sample_description))
