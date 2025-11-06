# Repository Agent Instructions

## Scope
These instructions apply to the entire repository unless a subdirectory overrides them
with its own `AGENTS.md` file.

## Documentation standards
- Prefer Markdown headings that form a logical outline (start at `##` when adding to
  existing documents that already have an `#` heading).
- Wrap lines at approximately 100 characters to keep diffs easy to review.
- When listing action items or steps, use ordered or unordered lists instead of dense
  paragraphs.

## Commit guidance
- Keep commit messages descriptive and in the imperative mood (e.g., "Add purged
  conformal plan").
- Group related changes together; do not mix unrelated edits in a single commit.

## Testing notes
- If you run tests or scripts, record the exact command in your PR or final report so
  reviewers can reproduce the results.
